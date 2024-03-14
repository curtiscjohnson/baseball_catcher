from baseball_detector import BaseballDetector
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

class TrajectoryEstimator:
    def __init__(self, crop, display):
        self.cropped = crop
        if self.cropped:
            self.crop_points_xleft = np.array([290, 440])
            self.crop_points_xright = np.array([205, 350])
            self.crop_points_y = np.array([0, 145])
        else:
            self.crop_points_xleft = np.array([0, 640])
            self.crop_points_xright = np.array([0, 640])
            self.crop_points_y = np.array([0, 480])

        self.left_detector = BaseballDetector(image_height=self.crop_points_y[1] - self.crop_points_y[0],
                                              image_width=self.crop_points_xleft[1] - self.crop_points_xleft[0],
                                              grayscale=False, display=display)
        self.right_detector = BaseballDetector(image_height=self.crop_points_y[1] - self.crop_points_y[0],
                                              image_width=self.crop_points_xright[1] - self.crop_points_xright[0],
                                              grayscale=False, display=False)

        self.display = display
        self.previous_xz_estimates = []
        self.previous_yz_estimates = []
        self.ball_loc_hist = []

        if self.display:
            plt.ion()

        #load parameters from file
        calibration_path = "./calibration/calibration_params"
        self.undistortRectifyMapLx = np.load(
            f'{calibration_path}/undistortRectifyMapLx.npy')
        self.undistortRectifyMapLy = np.load(
            f'{calibration_path}/undistortRectifyMapLy.npy')
        self.undistortRectifyMapRx = np.load(
            f'{calibration_path}/undistortRectifyMapRx.npy')
        self.undistortRectifyMapRy = np.load(
            f'{calibration_path}/undistortRectifyMapRy.npy')
        self.Q = np.load(f'{calibration_path}/Q.npy')

        self.P1 = np.load(f'{calibration_path}/P1.npy')
        self.P2 = np.load(f'{calibration_path}/P2.npy')

        print(f"P1: {self.P1}")
        print(f"P2: {self.P2}")



    def _save_3d_point(self, left_x, left_y, right_x, right_y) -> None:
        """ 
        ! To get disparity map, I am individually detecting the ball in 
        ! each image, then using diff in x to get disparity. 
        ! They are not always on the same horizonal line however.
        """
        if self.cropped:
            left_x += self.crop_points_xleft[0]
            left_y += self.crop_points_y[0]
            right_x += self.crop_points_xright[0]
            right_y += self.crop_points_y[0]

        # point = self.get_3d_point(left_x, left_y, right_x, right_y)
        point = self.triangulate_point(left_x, left_y, right_x, right_y)
        print(f"Camera frame: {point}")
        point = self._transform_to_catcher_frame(point)
        print(f"Catcher frame: {point}")

        self.ball_loc_hist.append(point)

    def get_3d_point(self, left_x, left_y, right_x, right_y):

        #calculate estimated disparity for the baseball center
        disparity = left_x - right_x

        baseball_center = np.array([left_x, left_y,
                                    disparity]).reshape(1, 1, 3)

        # use Q to get 3D point
        point = cv.perspectiveTransform(baseball_center.astype(np.float32),
                                           self.Q).squeeze()

        return point

    def triangulate_point(self, left_x, left_y, right_x, right_y):
        left_pts = np.array([left_x, left_y]).reshape(-1,1).astype(np.float32)
        right_pts = np.array([right_x, right_y]).reshape(-1,1).astype(np.float32)


        # print(f"P1:\n{self.P1}")
        # print(f"P2:\n{self.P2}")
        print(f"Left:\n{left_pts}")
        print(f"Right:\n{right_pts}")

        point_4d = cv.triangulatePoints(self.P1, self.P2, left_pts, right_pts)
        point3d = point_4d[:3]/point_4d[-1]
        print(point3d)
        return point3d.squeeze()        

    def _transform_to_catcher_frame(self, point_3d):
        x, y, z = point_3d
        delta_x = 11 #seems about right. making this smaller shifts the catcher to the left. I think
        delta_y = 23 #TODO: tune me and delta_z
        delta_z = 20
        z_offset = 0 #-30
        #432

        return np.array([x - delta_x, 
                         delta_y - y, 
                         delta_z - z - z_offset])
        
    def _mask_frames(self, lframe, rframe):
        lframe_masked = lframe[self.crop_points_y[0]: self.crop_points_y[1], self.crop_points_xleft[0]:self.crop_points_xleft[1]]
        rframe_masked = rframe[self.crop_points_y[0]: self.crop_points_y[1], self.crop_points_xright[0]:self.crop_points_xright[1]]

        if self.display:
            # cv.imshow("left masked", lframe_masked)
            # cv.imshow("right masked", rframe_masked)
            pass
        return lframe_masked, rframe_masked

    def _undistort_and_rectify(self, left_img, right_img):
        left_img_rect = cv.remap(left_img, self.undistortRectifyMapLx,
                                 self.undistortRectifyMapLy, cv.INTER_LINEAR)

        right_img_rect = cv.remap(right_img, self.undistortRectifyMapRx,
                                  self.undistortRectifyMapRy, cv.INTER_LINEAR)
        return left_img_rect, right_img_rect

    def _fit_curves(self, num_detections_to_wait):
        #? this might be slow. Fix later with slicing?
        x_hist = [loc[0] for loc in self.ball_loc_hist]
        y_hist = [loc[1] for loc in self.ball_loc_hist]
        z_hist = [loc[2] for loc in self.ball_loc_hist]
        print(f"Total ball detections: {len(self.ball_loc_hist)}")
        if len(self.ball_loc_hist) > 50:
            print("stopped updating")
            return self.previous_xz_estimates[-1], self.previous_yz_estimates[-1]

        if len(self.ball_loc_hist) > num_detections_to_wait:
            best_fit_line = np.polyfit(z_hist[6:], x_hist[6:], 1)
            best_fit_parabola = np.polyfit(z_hist[6:], y_hist[6:], 2)

            self.previous_xz_estimates.append(best_fit_line[-1])
            self.previous_yz_estimates.append(best_fit_parabola[-1])
            if self.display:
                print(f"Number of detections: {len(self.ball_loc_hist)}")
                print(f"XY Intercept: {best_fit_line[-1], best_fit_parabola[-1]}")
                # plot model estimate until z = 0
                z_hist_plot = np.linspace(z_hist[0], 0, 50)

                plt.figure("XZ Model of Ball Trajectory")
                plt.clf()
                plt.plot(z_hist, x_hist, 'x')
                plt.plot(z_hist_plot, np.polyval(best_fit_line, z_hist_plot), 'r-')
                plt.plot([0]*len(self.previous_xz_estimates), self.previous_xz_estimates, 'go')
                plt.xlim([-500, 20])
                # plt.ylim([50, -50])
                plt.grid()
                plt.title("XZ Model of Ball Trajectory")
                plt.xlabel("z")
                plt.ylabel("x")
                plt.legend(["Data", "Model", "Intercept"])

                plt.figure("YZ Model of Ball Trajectory")
                plt.clf()
                plt.plot(z_hist, y_hist, 'x')
                plt.plot(z_hist_plot, np.polyval(best_fit_parabola, z_hist_plot), 'r-')
                plt.plot([0]*len(self.previous_yz_estimates), self.previous_yz_estimates, 'go')
                plt.xlim([-500, 20])
                # plt.ylim([100, -100])
                plt.grid()
                plt.title("YZ Model of Ball Trajectory")
                plt.xlabel("z")
                plt.ylabel("y")
                plt.legend(["Data", "Model", "Intercept"])

                plt.figure("XY Model of Ball Trajectory")
                plt.clf()
                plt.scatter(self.previous_xz_estimates, self.previous_yz_estimates, c=list(range(len(self.previous_yz_estimates))), cmap='viridis')
                plt.colorbar(label='time frame (from num_detections_to_wait)')
                plt.grid()
                # plt.xlim([0, 30])
                # plt.ylim([0, 70])
                plt.title("XY Model of Ball Trajectory")

                plt.show(block=False)

            return best_fit_line[-1], best_fit_parabola[-1]
        else:
            return 0,0

    def get_ball_3D_location(self, lframe, rframe) -> np.ndarray:
        left_img_rect, right_img_rect = self._undistort_and_rectify(lframe, rframe)

        if self.cropped:
            lframe, rframe = self._mask_frames(left_img_rect, right_img_rect)

        left_x, left_y = self.left_detector.detect(lframe)
        right_x, right_y = self.right_detector.detect(rframe)

        if left_x is not None and right_x is not None:
            self._save_3d_point(left_x, left_y, right_x, right_y)
            return self.ball_loc_hist[-1]
        else:
            return None

    def get_intercept(self, num_detections_to_wait=15) -> tuple:
        x_intercept, y_intercept = self._fit_curves(num_detections_to_wait)
        return (x_intercept, y_intercept)
    
    def get_ball_3D_location_profile(self, lframe, rframe, mask=False) -> np.ndarray:
        if mask:
            start = time.time()
            lframe, rframe = self._mask_frames(lframe, rframe)
            print(f"\n\nMasking time: {(time.time() - start)*1000} ms")
        start = time.time()
        left_img_rect, right_img_rect = self._undistort_and_rectify(lframe, rframe)
        print(f"Undistort and rectify time: {(time.time() - start)*1000} ms")

        start = time.time()
        left_x, left_y = self.left_detector.detect(left_img_rect, profile=True)
        print(f"Left detect time: {(time.time() - start)*1000} ms")

        start = time.time()
        right_x, right_y = self.right_detector.detect(right_img_rect,profile=True)
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
