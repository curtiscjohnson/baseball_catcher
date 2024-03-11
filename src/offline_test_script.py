import numpy as np
import cv2 as cv
import os
import time

# from Flea2Camera2 import FleaCam
from trajectory_estimator import TrajectoryEstimator


if __name__ == "__main__":
    # dataset = "20240215112959"
    # dataset = "20240215113025" # ball is cropped out with the mask, but this looks like one of the datasets where the thrower was set too high (ball leaves the frame)
    # dataset = "20240215113049" # leaves image a bit at the end, but I think that is after we want to have a final catcher position anyways. 
    # dataset = "20240215113115" # stays in image great
    dataset = "20240215113139" # stays in image great
    # dataset = "20240215113202"
    # dataset = "20240215113225"

    path = f"data/{dataset}/"
    l_image_paths = os.listdir(path+"L")
    r_image_paths = os.listdir(path+"R")

    # camera = FleaCam()
    # height, width, channel = camera.frame_shape

    # Sort the image paths so they are in ascending order (sorted() puts 10 in front of 2)
    l_image_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    r_image_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    assert len(l_image_paths) == len(r_image_paths)

    estimator = TrajectoryEstimator(display=True)


    for i in range(35, 100):
    # while True:
        
        print(f"\n\nProcessing image {i+1}/{len(l_image_paths)}")
        lframe = cv.imread(f"{path}/L/{l_image_paths[i]}")
        rframe = cv.imread(f"{path}/R/{r_image_paths[i]}")
        # rframe, lframe = camera.getFrame()

        start = time.time()
        # get current 3D point of the ball
        point_3D = estimator.get_ball_3D_location(lframe, rframe, mask=True)

        # # get intercept estimate from trajectory estimator
        if point_3D is not None:
            x, y = estimator.get_intercept()
            print(f"Intercept: {x},{y}\n\n")

        # print(f"Loop Time w/o reading image: {(time.time() - start)*1000} ms")
        cv.imshow("left", lframe)
        cv.imshow("right", rframe)
        cv.waitKey(0)

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