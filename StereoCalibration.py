import cv2
import numpy as np
import glob
import os

class StereoCal:

    # def adjust_brightness_contrast(self, image, alpha=1.5, beta=50):
    #     """
    #     modify the contrast and brightness of checkerboard
    #     alpha: control contrast (1.0 keep the same，>1 increase contrast)
    #     beta: control brightness (0 keep the same，>0 increase brightness)
    #     """
    #     adjusted = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    #     return adjusted

    # # Camera calibration
    # def stereo_calibration (self, calib_path1, calib_path2, chessboard_size=(9, 6)):
    #     # termination criteria
    #     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #     #prepare object points
    #     objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    #     objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    #     # Arrays to store object points and image points from all the images.
    #     objpoints = [] # 3d point in real world space
    #     imgpoints_calib1 = [] # 2d points in image plane.
    #     imgpoints_calib2 = [] # 2d points in image plane.
    #     img_shape = None  # store image size
        
    #     images_calib1 = glob.glob(f'{calib_path1}/*.jpg')
    #     images_calib2 = glob.glob(f'{calib_path2}/*.jpg')
        
    #     assert len(images_calib1) == len(images_calib2)

    #     #in pairs
    #     for frame1, frame2 in zip(images_calib1, images_calib2):
    #         img1 =  cv.imread(frame1)
    #         img2 =  cv.imread(frame2)
    #         # ajust the brightness and contrast
    #         img1 = self.adjust_brightness_contrast(img1)
    #         img2 = self.adjust_brightness_contrast(img2)
            
    #         gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    #         gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    #         ret1, corners1 = cv.findChessboardCorners(gray1, chessboard_size, None)
    #         ret2, corners2 = cv.findChessboardCorners(gray2, chessboard_size, None)

    #         if ret1 == True and ret2 == True:
    #             objpoints.append(objp)

    #             corners1 = cv.cornerSubPix(gray1, corners1,(11,11), (-1,-1), criteria)
    #             corners2 = cv.cornerSubPix(gray2, corners2,(11,11), (-1,-1), criteria)
                
    #             imgpoints_calib1.append(corners1)
    #             imgpoints_calib2.append(corners2)

    #             img_shape = gray1.shape[::-1]
    #         else:
    #             print(f"Chessboard corners not found in images: {frame1}, {frame2}")
    #             continue  # skip this pair images
        
    #     num_valid_images = len(objpoints)
    #     print(f"Number of valid image pairs found: {num_valid_images}")
    #     stereocalib_criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5)

    #     ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_calib1, imgpoints_calib2,
    #                                                                         None, None, None, None, img_shape, 
    #                                                                         criteria = stereocalib_criteria, flags=cv.CALIB_FIX_PRINCIPAL_POINT )
            
    #     return CM1, dist1, CM2, dist2, R, T

        
    # def stereo_rectification(self, CM1, dist1, CM2, dist2, R, T, images_calib1, images_calib2):

    #     img_1 = cv.imread(images_calib1)
    #     img_2 = cv.imread(images_calib2)
    #     h, w = img_1.shape[:2]
    #     img_shape = img_1.shape[:2][::-1]
    #     # check if read the image successfully
    #     if img_1 is None or img_2 is None:
    #         print(f"Error: Failed to load images: {images_calib1}, {images_calib2}")
    #         return None, None, None

    #     #Undistortion

    #     # stereo rectification
    #     R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(CM1, dist1, CM2, dist2, img_shape, R, T, flags=cv.CALIB_ZERO_DISPARITY, alpha= -1)

    #     # prepare to remap (undistortion)
    #     maps1 = cv.initUndistortRectifyMap(CM1, dist1, R1, P1, img_shape, cv.CV_16SC2)
    #     maps2 = cv.initUndistortRectifyMap(CM2, dist2, R2, P2, img_shape, cv.CV_16SC2)

    #     rectified_1 = cv.remap(img_1, maps1[0], maps1[1], cv.INTER_LINEAR)
    #     rectified_2 = cv.remap(img_2, maps2[0], maps2[1], cv.INTER_LINEAR)

    #     return rectified_1, rectified_2, Q
    

    def CameraCalibrate(self, image_files_left, image_files_middle, image_files_right, chessboard_size=(9, 6), square_size=10):
    
        # 準備棋盤格的 3D 世界座標
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        # 儲存 3D 點和 2D 點的列表
        objpoints = []  # 3D 點
        imgpoints_left = []  # 左視角的2D 點
        imgpoints_middle = []  # 中視角的2D 點
        imgpoints_right = []  # 右視角的2D 點

        # 遍歷所有的校正影像
        for fname_left, fname_middle, fname_right in zip(image_files_left, image_files_middle, image_files_right):
            # 讀取左、中、右影像
            img_left = cv2.imread(fname_left)
            img_middle = cv2.imread(fname_middle)
            img_right = cv2.imread(fname_right)
            
            # 轉換為灰階
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_middle = cv2.cvtColor(img_middle, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            # 尋找棋盤格角點
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
            ret_middle, corners_middle = cv2.findChessboardCorners(gray_middle, chessboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

            # 如果三個視角都找到角點，則進行進一步處理
            if ret_left and ret_middle and ret_right:
                objpoints.append(objp)

                # 對角點進行亞像素精度調整
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left_refined = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_middle_refined = cv2.cornerSubPix(gray_middle, corners_middle, (11, 11), (-1, -1), criteria)
                corners_right_refined = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

                imgpoints_left.append(corners_left_refined)
                imgpoints_middle.append(corners_middle_refined)
                imgpoints_right.append(corners_right_refined)

        # 分別進行左、中、右三個相機的校正
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

