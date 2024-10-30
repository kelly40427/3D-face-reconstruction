import cv2
import numpy as np
import glob
import os

class StereoCal:

    def CameraCalibrate(self, image_files_left, image_files_middle, image_files_right, chessboard_size=(9, 6), square_size=10):
    
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        objpoints = [] 
        imgpoints_left = []  
        imgpoints_middle = []  
        imgpoints_right = []  


        for fname_left, fname_middle, fname_right in zip(image_files_left, image_files_middle, image_files_right):

            img_left = cv2.imread(fname_left)
            img_middle = cv2.imread(fname_middle)
            img_right = cv2.imread(fname_right)
            
            # Covert color into gray
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_middle = cv2.cvtColor(img_middle, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
            ret_middle, corners_middle = cv2.findChessboardCorners(gray_middle, chessboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)


            if ret_left and ret_middle and ret_right:
                objpoints.append(objp)

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left_refined = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_middle_refined = cv2.cornerSubPix(gray_middle, corners_middle, (11, 11), (-1, -1), criteria)
                corners_right_refined = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

                imgpoints_left.append(corners_left_refined)
                imgpoints_middle.append(corners_middle_refined)
                imgpoints_right.append(corners_right_refined)

        # Camera calibration for each camera
        _, camera_matrix_left, dist_coeffs_left, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
        _, camera_matrix_middle, dist_coeffs_middle, _, _ = cv2.calibrateCamera(objpoints, imgpoints_middle, gray_middle.shape[::-1], None, None)
        _, camera_matrix_right, dist_coeffs_right, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

        return camera_matrix_left, dist_coeffs_left, camera_matrix_middle, dist_coeffs_middle, camera_matrix_right, dist_coeffs_right, objpoints, imgpoints_left, imgpoints_middle, imgpoints_right


    def stereo_calibrate_and_rectify(self, objpoints, imgpoints1, imgpoints2, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, image_size):

        # Stereo calibration
        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F= cv2.stereoCalibrate(
            objpoints, imgpoints1, imgpoints2, 
            camera_matrix1, dist_coeffs1,
            camera_matrix2, dist_coeffs2, 
            image_size, flags=cv2.CALIB_FIX_INTRINSIC 
        )

        # Stereo rectification
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix1, distCoeffs1,
                                                    cameraMatrix2, distCoeffs2,
                                                    image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)

        # Compute rectification maps
        map1_x, map1_y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, image_size, cv2.CV_32FC1)
        map2_x, map2_y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, image_size, cv2.CV_32FC1)

        return R1, R2, P1, P2, Q, map1_x, map1_y, map2_x, map2_y, T

    def rectify_images(self, image1, image2, map1_x, map1_y, map2_x, map2_y):
        """
        Rectify images while preserving alpha channel if present
        """

        if len(image1.shape) == 3 and image1.shape[2] == 4:  # BGRA format
            # Respectively Remap The BGR and A channels
            rectified_bgr1 = cv2.remap(image1[:,:,:3], map1_x, map1_y, cv2.INTER_LINEAR)
            rectified_alpha1 = cv2.remap(image1[:,:,3], map1_x, map1_y, cv2.INTER_LINEAR)
            rectified_image1 = cv2.merge([rectified_bgr1, rectified_alpha1[:,:,np.newaxis]])
        else:
            print("NOT IN BGRA FORMAT")
            rectified_image1 = cv2.remap(image1, map1_x, map1_y, cv2.INTER_LINEAR)

        if len(image2.shape) == 3 and image2.shape[2] == 4:  # BGRA format
            rectified_bgr2 = cv2.remap(image2[:,:,:3], map2_x, map2_y, cv2.INTER_LINEAR)
            rectified_alpha2 = cv2.remap(image2[:,:,3], map2_x, map2_y, cv2.INTER_LINEAR)
            rectified_image2 = cv2.merge([rectified_bgr2, rectified_alpha2[:,:,np.newaxis]])
        else:
            print("NOT IN BGRA FORMAT")
            rectified_image2 = cv2.remap(image2, map2_x, map2_y, cv2.INTER_LINEAR)

        return rectified_image1, rectified_image2

