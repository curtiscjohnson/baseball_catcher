import cv2 as cv
import os
import numpy as np

def detect_corners(path, size, display=False):
    img = cv.imread(path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray_img, size, None)
    if ret:
        refined_corners = cv.cornerSubPix(gray_img, corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 30, 0.001))

        if display:
            refined_corners_img = img.copy()
            cv.drawChessboardCorners(refined_corners_img, size, refined_corners, ret)

            cv.imshow("Refined Corners", refined_corners_img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return refined_corners
    else:
        print(f"No corners detected in {path}")

def calibrate_single_camera(data_folder):
    size = (10,7)
    objp = np.zeros((size[0]*size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1,2)
    objpoints = []
    imgpoints = []

    data = sorted(os.listdir(data_folder))
    for img in data:
        img_path = os.path.join(data_folder, img)
        corners = detect_corners(img_path, size, display=False)

        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (640,480), None, None)

    return mtx, dist

def get_img_points(data_path, size=(10,7)):
    img_points = []

    for img in sorted(os.listdir(data_path)):
        img_path = os.path.join(data_path, img)
        img = cv.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, size, None)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_points.append(corners2)
        else:
            print(f"No corners detected in {img_path}")
    return img_points

def calibrate_stereo_camera(left_intrinsic_matrix, left_distortion_matrix, 
                            right_intrinsic_matrix, right_distortion_matrix, stereo_img_path):
    size = (10,7)
    objp = np.zeros((size[0]*size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1,2) * 3.88 #TODO what size should this be???? 3.88?

    left_img_points = get_img_points(stereo_img_path+"/L")
    right_img_points = get_img_points(stereo_img_path+"/R")
    objpoints = [objp]*len(left_img_points)

    ret, _, _, _, _, R, T, E, F = cv.stereoCalibrate(objpoints, left_img_points, right_img_points, left_intrinsic_matrix, \
                                                     left_distortion_matrix, right_intrinsic_matrix, right_distortion_matrix, (640,480), \
                                                     criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001), flags=cv.CALIB_FIX_INTRINSIC)

    return R, T, E, F


if __name__ == "__main__":
    save_path = "calibration/calibration_params"
    data_folder_left = "data/calibration_imgs/LeftOnly/L"
    data_folder_right = "data/calibration_imgs/RightOnly/R"
    stereo_path = "data/calibration_imgs/Stereo"

    left_intrinsic_matrix, left_disortion_matrix = calibrate_single_camera(data_folder_left)
    right_intrinsic_matrix, right_distortion_matrix = calibrate_single_camera(data_folder_right)

    R, T, E, F = calibrate_stereo_camera(left_intrinsic_matrix, left_disortion_matrix, 
                                         right_intrinsic_matrix, right_distortion_matrix,
                                         stereo_path)

    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(left_intrinsic_matrix, left_disortion_matrix, right_intrinsic_matrix, right_distortion_matrix, (640,480), R, T)
    undistortRectifyMapLx, undistortRectifyMapLy = cv.initUndistortRectifyMap(left_intrinsic_matrix, left_disortion_matrix, R1, P1, (640,480), cv.CV_32FC1)
    undistortRectifyMapRx, undistortRectifyMapRy = cv.initUndistortRectifyMap(right_intrinsic_matrix, right_distortion_matrix, R2, P2, (640,480), cv.CV_32FC1)

    os.makedirs(save_path, exist_ok=True)

    np.save(f"{save_path}/Q.npy", Q)
    np.save(f"{save_path}/undistortRectifyMapLx.npy", undistortRectifyMapLx)
    np.save(f"{save_path}/undistortRectifyMapLy.npy", undistortRectifyMapLy)
    np.save(f"{save_path}/undistortRectifyMapRx.npy", undistortRectifyMapRx)
    np.save(f"{save_path}/undistortRectifyMapRy.npy", undistortRectifyMapRy)

    # save R, T, E, and F to a yaml
    fs = cv.FileStorage(f"{save_path}/calibration_params.yml", cv.FILE_STORAGE_WRITE)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("E", E)
    fs.write("F", F)
    fs.write("Q", Q)
    fs.write("Left Intrinsic Matrix", left_intrinsic_matrix)
    fs.write("Left Distortion Matrix", left_disortion_matrix)
    fs.write("Right Intrinsic Matrix", right_intrinsic_matrix)
    fs.write("Right Distortion Matrix", right_distortion_matrix)

    fs.release()
