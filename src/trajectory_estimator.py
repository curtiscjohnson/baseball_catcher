from baseball_detector import BaseballDetector
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

class TrajectoryEstimator:
    def __init__(self, display):
        self.left_detector = BaseballDetector(grayscale=False, display=display)
        self.right_detector = BaseballDetector(grayscale=False, display=display)
        self.display = display
        if self.display:
            plt.ion()

        #load parameters from file
        self.undistortRectifyMapLx = np.load(
            './calibration_params/undistortRectifyMapLx.npy')
        self.undistortRectifyMapLy = np.load(
            './calibration_params/undistortRectifyMapLy.npy')
        self.undistortRectifyMapRx = np.load(
            './calibration_params/undistortRectifyMapRx.npy')
        self.undistortRectifyMapRy = np.load(
            './calibration_params/undistortRectifyMapRy.npy')
        self.Q = np.load('./calibration_params/Q.npy')

        self.mask_points_left = np.array( [[
            [335,0], # top left
            [525,0], # top right, more aggresive was 500
            [675,480], # bottom right, more aggresive was 640
            [335,480], # bottom left
        ]], dtype=np.int32)
        self.mask_points_right = np.array( [[
            [100,0], # top left
            [300,0], # top right
            [300,480], # bottom right
            [50,480], # bottom left
        ]], dtype=np.int32)

        self.ball_loc_hist = []

    def _save_3d_point(self, left_x, left_y, right_x, right_y) -> None:
        """ 
        ! To get disparity map, I am individually detecting the ball in 
        ! each image, then using diff in x to get disparity. 
        ! They are not always on the same horizonal line however.
        """

        #calculate estimated disparity for the baseball center
        disparity = left_x - right_x

        baseball_center = np.array([left_x, left_y,
                                    disparity]).reshape(1, 1, 3)

        # use Q to get 3D point
        point = cv.perspectiveTransform(baseball_center.astype(np.float32),
                                           self.Q).squeeze()

        # point = self._transform_to_catcher_frame(point)

        if self.display:
            print(f"3D point in catcher frame: {point}")
        self.ball_loc_hist.append(point)

    def _transform_to_catcher_frame(self):
        raise NotImplementedError
    
    def _mask_frames(self, lframe, rframe):
        lmask = np.zeros(lframe.shape[:2], dtype="uint8")
        cv.fillPoly(lmask, self.mask_points_left, 255)
        lframe_masked = cv.bitwise_and(lframe, lframe, mask=lmask)

        rmask = np.zeros(rframe.shape[:2], dtype="uint8")
        cv.fillPoly(rmask, self.mask_points_right, 255)
        rframe_masked = cv.bitwise_and(rframe, rframe, mask=rmask)
        
        if self.display:
            cv.imshow("left masked", lframe_masked)
            cv.imshow("right masked", rframe_masked)
        return lframe_masked, rframe_masked

    def _undistort_and_rectify(self, left_img, right_img):
        left_img_rect = cv.remap(left_img, self.undistortRectifyMapLx,
                                 self.undistortRectifyMapLy, cv.INTER_LINEAR)

        right_img_rect = cv.remap(right_img, self.undistortRectifyMapRx,
                                  self.undistortRectifyMapRy, cv.INTER_LINEAR)
        return left_img_rect, right_img_rect

    def _fit_curves(self):
        #? this might be slow. Fix later with slicing?
        x_hist = [loc[0] for loc in self.ball_loc_hist]
        y_hist = [loc[1] for loc in self.ball_loc_hist]
        z_hist = [loc[2] for loc in self.ball_loc_hist]

        if len(self.ball_loc_hist) > 5:
            best_fit_line = np.polyfit(z_hist, x_hist, 1)
            best_fit_parabola = np.polyfit(z_hist, y_hist, 2)

            if self.display:
                # plot model estimate until z = 0
                z_hist_plot = np.linspace(z_hist[0], 0, 50)

                plt.figure("XZ Model of Ball Trajectory")
                plt.clf()
                plt.plot(z_hist, x_hist, 'x')
                plt.plot(z_hist_plot, np.polyval(best_fit_line, z_hist_plot), 'r-')
                plt.xlim([425, -20])
                plt.ylim([50, -50])
                plt.grid()
                plt.title("XZ Model of Ball Trajectory")
                plt.xlabel("z")
                plt.ylabel("x")

                plt.figure("YZ Model of Ball Trajectory")
                plt.clf()
                plt.plot(z_hist, y_hist, 'x')
                plt.plot(z_hist_plot, np.polyval(best_fit_parabola, z_hist_plot), 'r-')
                plt.xlim([425, -20])
                plt.ylim([100, -100])
                plt.grid()
                plt.title("YZ Model of Ball Trajectory")
                plt.xlabel("z")
                plt.ylabel("y")

                plt.show(block=False)

            return best_fit_line[-1], best_fit_parabola[-1]
        else:
            return 0,0

    def get_ball_3D_location(self, lframe, rframe, mask=False) -> np.ndarray:
        if mask:
            lframe, rframe = self._mask_frames(lframe, rframe)
        left_img_rect, right_img_rect = self._undistort_and_rectify(lframe, rframe)

        left_x, left_y = self.left_detector.detect(left_img_rect)
        right_x, right_y = self.right_detector.detect(right_img_rect)

        if left_x is not None and right_x is not None:
            self._save_3d_point(left_x, left_y, right_x, right_y)
            return self.ball_loc_hist[-1]
        else:
            return None

    def get_intercept(self) -> tuple:
        x_intercept, y_intercept = self._fit_curves()
        return (x_intercept, y_intercept)
    
    def get_ball_3D_location_profile(self, lframe, rframe, mask=False) -> np.ndarray:
        if mask:
            start = time.time()
            lframe, rframe = self._mask_frames(lframe, rframe)
            print(f"Masking time: {(time.time() - start)*1000} ms")
        start = time.time()
        left_img_rect, right_img_rect = self._undistort_and_rectify(lframe, rframe)
        print(f"Undistort and rectify time: {(time.time() - start)*1000} ms")

        start = time.time()
        left_x, left_y = self.left_detector.detect(left_img_rect)
        print(f"Left detect time: {(time.time() - start)*1000} ms")

        start = time.time()
        right_x, right_y = self.right_detector.detect(right_img_rect)
        print(f"Right detect time: {(time.time() - start)*1000} ms")

        if left_x is not None and right_x is not None:
            start = time.time()
            self._save_3d_point(left_x, left_y, right_x, right_y)
            print(f"Save 3D point time: {(time.time() - start)*1000} ms")
            return self.ball_loc_hist[-1]
        else:
            return None
        
    def get_intercept_profile(self) -> tuple:
        start = time.time()
        x_intercept, y_intercept = self._fit_curves()
        print(f"Fit curves time: {(time.time() - start)*1000} ms")
        return (x_intercept, y_intercept)