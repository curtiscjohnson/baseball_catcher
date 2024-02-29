from trajectory_estimator import TrajectoryEstimator
from Flea2Camera2 import FleaCam
from roboteq_wrapper import RoboteqWrapper
from roboteq_constants import *
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
    catcher.MoveAtSpeed(0, 0) xspeed, yspeed. +x is left, +y is up

    catcher.setHome()

    catcher.MoveToXY(x,y)
    """
    catcher = RoboteqWrapper(PORT)

    estimator = TrajectoryEstimator()


    while True:
        lframe, rframe = camera.getFrame()
        x, y = estimator.get_intercept(lframe, rframe)

        if x is not None and y is not None:
            catcher.MoveToXY(x, y)
