from trajectory_estimator import TrajectoryEstimator
from Flea2Camera2 import FleaCam
from roboteq_wrapper import RoboteqWrapper
from roboteq_constants import *
import numpy as np
import cv2 as cv
PORT = '/dev/ttyUSB0'  



if __name__ == "__main__":

    camera = FleaCam() #this is both cameras

    """
    Camera API notes
    lframe, rframe = camera.getFrame()
    """

    """
    #API NOTES
    catcher.SetToDefault()

    Set to open loop and speed to 0 in case the catcher hits the safety stop
    catcher.MoveAtSpeed(0, 0) xspeed, yspeed. +x is towards right camera, +y is up

    catcher.setHome()

    catcher.MoveToXY(x,y)
    """
    estimator = TrajectoryEstimator(display=False)
    # catcher = RoboteqWrapper(PORT)


    def mask_image(image, points):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv.fillPoly(mask, points, 255)
        masked = cv.bitwise_and(image, image, mask=mask)
        return masked

    points_left = np.array( [[
        [160,10], # top left
        [350,10], # top right
        [500,450], # bottom right
        [50,450], # bottom left
    ]],
    dtype=np.int32)
    points_right = np.array( [[
        [270,10], # top left
        [500,10], # top right
        [615,450], # bottom right
        [150,450], # bottom left
    ]],
    dtype=np.int32)


    while True:
        lframe, rframe = camera.getFrame()

        cv.imshow("left_mask_image", mask_image(lframe, points_left))
        cv.imshow("right_mask_image", mask_image(rframe, points_right))
        cv.waitKey(0)

        x, y = estimator.get_intercept(lframe, rframe)

        print(x,y)

        # if x is not None and y is not None:
        #     catcher.MoveToXY(x, y)
