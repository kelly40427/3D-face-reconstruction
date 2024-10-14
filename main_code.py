
import cv2 as cv
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image

from ColourNormalisation import ColourNorm
from StereoCalibration import StereoCal

colournorm=ColourNorm()
stereocal=StereoCal()

# subject_paths = [
#     'D:/MyCode/Image Processing/project/ipcv_project3/subject1',
#     'D:/MyCode/Image Processing/project/ipcv_project3/subject2',
#     'D:/MyCode/Image Processing/project/ipcv_project3/subject4'
# ]

subject_paths = [
    'D:/MyCode/Image Processing/project/ipcv_project3/subject1',
    "C:\Users\lobke\OneDrive - University of Twente\Documents\BME5\image processing and computer vision\project\ipcv_project3\subject1"
    'D:/MyCode/Image Processing/project/ipcv_project3/subject2',
    'D:/MyCode/Image Processing/project/ipcv_project3/subject4'
]

calib_left = "D:/MyCode/Image Processing/project/ipcv_project3/Calibratie 1/calibrationLeft"
calib_middle = "D:/MyCode/Image Processing/project/ipcv_project3/Calibratie 1/calibrationMiddle"
calib_right = "D:/MyCode/Image Processing/project/ipcv_project3/Calibratie 1/calibrationRight"

CM1, dist1, CM2, dist2, R_left_middle, T_left_middle = stereocal.stereo_calibration(calib_left, calib_middle)
CM_middle, dist_middle, CM_right, dist_right, R_middle_right, T_middle_right = stereocal.stereo_calibration(calib_middle, calib_right)

scale_percent = 50 
for subject_path in subject_paths:
    left_images = sorted(glob.glob(os.path.join(subject_path, 'subject*Left', '*.jpg')))
    middle_images = sorted(glob.glob(os.path.join(subject_path, 'subject*Middle', '*.jpg')))
    right_images = sorted(glob.glob(os.path.join(subject_path, 'subject*Right', '*.jpg')))

    if len(left_images) != len(middle_images) or len(middle_images) != len(right_images):
        print(f"Error: Number of images do not match in Left, Middle, or Right folders in {subject_path}")
        continue
    
    # process each pair
    for left_img, middle_img, right_img in zip(left_images, middle_images, right_images):
        rectified_left, rectified_middle, Q_left_middle = stereocal.stereo_rectification(CM1, dist1, CM2, dist2, R_left_middle, T_left_middle, left_img, middle_img)

        # showing the image (left, middle)
        if rectified_left is not None and rectified_middle is not None:
            cv.imshow('Rectified Left', rectified_left)
            cv.imshow('Rectified Middle', rectified_middle)
            cv.waitKey(0)  
            cv.destroyAllWindows()
        else:
            print(f"Rectification failed for images: {left_img}, {middle_img}")

        # process (middle, right)
        rectified_middle2, rectified_right, Q_middle_right = stereocal.stereo_rectification(CM_middle, dist_middle, CM_right, dist_right, R_middle_right, T_middle_right, middle_img, right_img)

        # show the image (middle, right)
        if rectified_middle2 is not None and rectified_right is not None:
            cv.imshow('Rectified Middle (middle-right)', rectified_middle2)
            cv.imshow('Rectified Right (middle-right)', rectified_right)
            cv.waitKey(0) 
            cv.destroyAllWindows()
        else:
            print(f"Rectification failed for images: {middle_img}, {right_img}")