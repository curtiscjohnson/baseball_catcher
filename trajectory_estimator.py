from baseball_detector import BaseballDetector
import numpy as np
import cv2 as cv


class TrajectoryEstimator:
    def __init__(self):
        self.left_detector = BaseballDetector()
        self.right_detector = BaseballDetector()

        #load parameters from file
        self.undistortRectifyMapLx = np.load(
            './calibration/undistortRectifyMapLx.npy')
        self.undistortRectifyMapLy = np.load(
            './calibration/undistortRectifyMapLy.npy')
        self.undistortRectifyMapRx = np.load(
            './calibration/undistortRectifyMapRx.npy')
        self.undistortRectifyMapRy = np.load(
            './calibration/undistortRectifyMapRy.npy')
        self.Q = np.load('./calibration/Q.npy')

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

        #use Q to get 3D point
        points3d = cv.perspectiveTransform(baseball_center.astype(np.float32),
                                           self.Q).squeeze()

        point = self._transform_to_catcher_frame(points3d)
        self.ball_loc_hist.append(point)

    def _transform_to_catcher_frame(self):
        raise NotImplementedError

    def _undistort_and_rectify(self, left_img, right_img):
        left_img_rect = cv.remap(left_img, self.undistorself.tRectifyMapLx,
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

        return best_fit_line[-1], best_fit_parabola[-1]

    def get_intercept(self, left_image, right_image) -> np.ndarray:
        left_img_rect, right_img_rect = self._undistort_and_rectify(
            left_image, right_image)

        left_x, left_y = self.left_detector.detect_ball(left_image)
        right_x, right_y = self.right_detector.detect_ball(right_image)

        if left_x is not None and right_x is not None:
            self._save_3d_point(left_x, left_y, right_x, right_y)

        x_intercept, y_intercept = self._fit_curves()

        return (x_intercept, y_intercept)
