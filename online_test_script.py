import numpy as np
import cv2 as cv
import os
import time

from Flea2Camera2 import FleaCam
from trajectory_estimator import TrajectoryEstimator


if __name__ == "__main__":

    camera = FleaCam()
    height, width, channel = camera.frame_shape

    estimator = TrajectoryEstimator(display=True)

    print("running")


    while True:
        
        rframe, lframe = camera.getFrame()
        start = time.time()

        #WE HAVE 16 ms to finish the rest of this loop

        # get current 3D point of the ball
        point_3D = estimator.get_ball_3D_location(lframe, rframe, mask=True)

        # # # get intercept estimate from trajectory estimator
        if point_3D is not None:
            x, y = estimator.get_intercept()
            print(f"Intercept: {x},{y}\n\n")

        # print(time.time() - start)
        # print(f"Loop Time w/o reading image: {(time.time() - start)*1000} ms")
        # cv.imshow("left", lframe)
        # cv.imshow("right", rframe)
        # cv.waitKey(0)

    # camera = FleaCam() #this is both cameras

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