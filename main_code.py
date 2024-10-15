import cv2 as cv
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
import os

# Import the classes
from colour_normalisation import ColourNorm
from StereoCalibration import StereoCal
from backgroundRemoval import background
from stereoMatching import stereoMatching

# Define the classes
colournorm=ColourNorm() 
stereocal=StereoCal() 
background_removal=background()

# Paths to calibration images
calib_left = 'D:/MyCode/Image Processing/project/ipcv_project3/Calibratie 1/calibrationLeft'
calib_middle = 'D:/MyCode/Image Processing/project/ipcv_project3/Calibratie 1/calibrationMiddle'
calib_right = 'D:/MyCode/Image Processing/project/ipcv_project3/Calibratie 1/calibrationRight'

CM1, dist1, CM2, dist2, R_left_middle, T_left_middle = stereocal.stereo_calibration(calib_left, calib_middle)
CM_middle, dist_middle, CM_right, dist_right, R_middle_right, T_middle_right = stereocal.stereo_calibration(calib_middle, calib_right)

subject_paths = [
    'D:/MyCode/Image Processing/project/ipcv_project3/subject1',
    'D:/MyCode/Image Processing/project/ipcv_project3/subject2',
    'D:/MyCode/Image Processing/project/ipcv_project3/subject4'
]

for subject_path in subject_paths:
    # Load images from 'Left', 'Middle', and 'Right' folders
    left_img_path = glob.glob(os.path.join(subject_path, 'subject*Left', '*.jpg'))[0]
    middle_img_path = glob.glob(os.path.join(subject_path, 'subject*Middle', '*.jpg'))[0]
    right_img_path = glob.glob(os.path.join(subject_path, 'subject*Right', '*.jpg'))[0]
    
    # Open the images
    left_img = cv.imread(left_img_path)
    middle_img = cv.imread(middle_img_path)
    right_img = cv.imread(right_img_path)

    # Background removal on original images
    left_coords = background_removal.get_coordinates(left_img)
    middle_coords = background_removal.get_coordinates(middle_img)
    right_coords = background_removal.get_coordinates(right_img)

    left_img_bg_removed = background_removal.remove_background(left_img, left_coords)
    middle_img_bg_removed = background_removal.remove_background(middle_img, middle_coords)
    right_img_bg_removed = background_removal.remove_background(right_img, right_coords)

    # Save background removed images for further use
    bg_removed_folder = os.path.join(subject_path, 'bg_removed')
    if not os.path.exists(bg_removed_folder):
        os.makedirs(bg_removed_folder)

    left_img_bg_removed_path = os.path.join(bg_removed_folder, 'left_bg_removed.jpg')
    middle_img_bg_removed_path = os.path.join(bg_removed_folder, 'middle_bg_removed.jpg')
    right_img_bg_removed_path = os.path.join(bg_removed_folder, 'right_bg_removed.jpg')

    cv.imwrite(left_img_bg_removed_path, left_img_bg_removed)
    cv.imwrite(middle_img_bg_removed_path, middle_img_bg_removed)
    cv.imwrite(right_img_bg_removed_path, right_img_bg_removed)

    # Stereo rectification on background-removed images (Left, Middle)
    rectified_left, rectified_middle, Q_left_middle = stereocal.stereo_rectification(
        CM1, dist1, CM2, dist2, R_left_middle, T_left_middle, left_img_bg_removed_path, middle_img_bg_removed_path
    )
    
    # Save rectified images (Left, Middle)
    rectified_folder = os.path.join(subject_path, 'rectified')
    if not os.path.exists(rectified_folder):
        os.makedirs(rectified_folder)

    rectified_left_path = os.path.join(rectified_folder, 'rectified_left.jpg')
    rectified_middle_path = os.path.join(rectified_folder, 'rectified_middle.jpg')

    if rectified_left is not None and rectified_middle is not None:
        cv.imwrite(rectified_left_path, rectified_left)
        cv.imwrite(rectified_middle_path, rectified_middle)
        
        # Plot the rectified images
        # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # axs[0].imshow(cv.cvtColor(rectified_left, cv.COLOR_BGR2RGB))
        # axs[0].set_title('Rectified Left (BG Removed)')
        # axs[0].axis('off')
        # axs[1].imshow(cv.cvtColor(rectified_middle, cv.COLOR_BGR2RGB))
        # axs[1].set_title('Rectified Middle (BG Removed)')
        # axs[1].axis('off')
        # plt.show()

        # Stereo matching on rectified images (Left, Middle)
        DMap = stereoMatching(rectified_left, rectified_middle, 16, 16)
        DMap_clipped = np.clip(DMap, 0, 255)

        # Plot and save disparity map
        plt.figure(figsize=(10, 5))
        plt.imshow(DMap_clipped, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        disparity_map_path = os.path.join(rectified_folder, 'disparity_map_left_middle.png')
        plt.savefig(disparity_map_path)
        plt.title('Disparity Map (Left, Middle)')
        plt.show()
    else:
        print(f"Rectification failed for images: {left_img_path}, {middle_img_path}")

    # Stereo rectification on background-removed images (Middle, Right)
    rectified_middle2, rectified_right, Q_middle_right = stereocal.stereo_rectification(
        CM_middle, dist_middle, CM_right, dist_right, R_middle_right, T_middle_right, middle_img_bg_removed_path, right_img_bg_removed_path
    )

    # Save rectified images (Middle, Right)
    rectified_middle2_path = os.path.join(rectified_folder, 'rectified_middle2.jpg')
    rectified_right_path = os.path.join(rectified_folder, 'rectified_right.jpg')

    if rectified_middle2 is not None and rectified_right is not None:
        cv.imwrite(rectified_middle2_path, rectified_middle2)
        cv.imwrite(rectified_right_path, rectified_right)
        
        # Plot the rectified images
        # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # axs[0].imshow(cv.cvtColor(rectified_middle2, cv.COLOR_BGR2RGB))
        # axs[0].set_title('Rectified Middle (BG Removed)')
        # axs[0].axis('off')
        # axs[1].imshow(cv.cvtColor(rectified_right, cv.COLOR_BGR2RGB))
        # axs[1].set_title('Rectified Right (BG Removed)')
        # axs[1].axis('off')
        # plt.show()

        # Stereo matching on rectified images (Middle, Right)
        DMap = stereoMatching(rectified_middle2, rectified_right, 16, 16)
        DMap_clipped = np.clip(DMap, 0, 255)

        # Plot and save disparity map
        plt.figure(figsize=(10, 5))
        plt.imshow(DMap_clipped, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        disparity_map_path = os.path.join(rectified_folder, 'disparity_map_middle_right.png')
        plt.savefig(disparity_map_path)
        plt.title('Disparity Map (Middle, Right)')
        plt.show()
    else:
        print(f"Rectification failed for images: {middle_img_path}, {right_img_path}")
