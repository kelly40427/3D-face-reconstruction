import cv2 as cv
import numpy as np
import glob
import os

class StereoCal:

    def adjust_brightness_contrast(self, image, alpha=1.5, beta=50):
        """
        modify the contrast and brightness of checkerboard
        alpha: control contrast (1.0 keep the same，>1 increase contrast)
        beta: control brightness (0 keep the same，>0 increase brightness)
        """
        adjusted = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted

    # Camera calibration
    def stereo_calibration (self, calib_path1, calib_path2, chessboard_size=(9, 6)):
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        #prepare object points
        objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints_calib1 = [] # 2d points in image plane.
        imgpoints_calib2 = [] # 2d points in image plane.
        img_shape = None  # store image size
        
        images_calib1 = glob.glob(f'{calib_path1}/*.jpg')
        images_calib2 = glob.glob(f'{calib_path2}/*.jpg')
        
        assert len(images_calib1) == len(images_calib2)

        #in pairs
        for frame1, frame2 in zip(images_calib1, images_calib2):
            img1 =  cv.imread(frame1)
            img2 =  cv.imread(frame2)
            # ajust the brightness and contrast
            img1 = self.adjust_brightness_contrast(img1)
            img2 = self.adjust_brightness_contrast(img2)
            
            gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

            ret1, corners1 = cv.findChessboardCorners(gray1, chessboard_size, None)
            ret2, corners2 = cv.findChessboardCorners(gray2, chessboard_size, None)

            if ret1 == True and ret2 == True:
                objpoints.append(objp)

                corners1 = cv.cornerSubPix(gray1, corners1,(11,11), (-1,-1), criteria)
                corners2 = cv.cornerSubPix(gray2, corners2,(11,11), (-1,-1), criteria)
                
                imgpoints_calib1.append(corners1)
                imgpoints_calib2.append(corners2)

                img_shape = gray1.shape[::-1]
            else:
                print(f"Chessboard corners not found in images: {frame1}, {frame2}")
                continue  # skip this pair images
        
        num_valid_images = len(objpoints)
        print(f"Number of valid image pairs found: {num_valid_images}")
        stereocalib_criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5)

        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_calib1, imgpoints_calib2,
                                                                            None, None, None, None, img_shape, 
                                                                            criteria = stereocalib_criteria, flags=cv.CALIB_FIX_PRINCIPAL_POINT )
            
        return CM1, dist1, CM2, dist2, R, T

        
    def stereo_rectification(self, CM1, dist1, CM2, dist2, R, T, images_calib1, images_calib2):

        img_1 = cv.imread(images_calib1)
        img_2 = cv.imread(images_calib2)
        h, w = img_1.shape[:2]
        img_shape = img_1.shape[:2][::-1]
        # check if read the image successfully
        if img_1 is None or img_2 is None:
            print(f"Error: Failed to load images: {images_calib1}, {images_calib2}")
            return None, None, None

        #Undistortion

        # stereo rectification
        R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(CM1, dist1, CM2, dist2, img_shape, R, T, flags=cv.CALIB_ZERO_TANGENT_DIST)

        # prepare to remap (undistortion)
        maps1 = cv.initUndistortRectifyMap(CM1, dist1, R1, P1, img_shape, cv.CV_32FC1)
        maps2 = cv.initUndistortRectifyMap(CM2, dist2, R2, P2, img_shape, cv.CV_32FC1)

        rectified_1 = cv.remap(img_1, maps1[0], maps1[1], cv.INTER_LINEAR)
        rectified_2 = cv.remap(img_2, maps2[0], maps2[1], cv.INTER_LINEAR)

        # Show rectified images
        cv.imshow('Rectified Image 1', rectified_1)
        cv.imshow('Rectified Image 2', rectified_2)
        cv.waitKey(0)
        cv.destroyAllWindows()

        return rectified_1, rectified_2, Q


