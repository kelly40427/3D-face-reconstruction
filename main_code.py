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
from DepthMapCreator import DepthMapCreator
from MeshGenerator import MeshGenerator  

# Define the classes
colournorm = ColourNorm()
stereocal = StereoCal()
background_removal = background()
stereo_matching = stereoMatching()
depth_map_creator = DepthMapCreator()
mesh_generator = MeshGenerator()  

# Paths to calibration images
calib_left = 'ipcv_project3/Calibratie 1/calibrationLeft'
calib_middle = 'ipcv_project3/Calibratie 1/calibrationMiddle'
calib_right = 'ipcv_project3/Calibratie 1/calibrationRight'

CM1, dist1, CM2, dist2, R_left_middle, T_left_middle = stereocal.stereo_calibration(calib_left, calib_middle)
CM_middle, dist_middle, CM_right, dist_right, R_middle_right, T_middle_right = stereocal.stereo_calibration(calib_middle, calib_right)

# Paths to subjects
subject_paths = [
    'ipcv_project3/subject1',
    'ipcv_project3/subject2',
    'ipcv_project3/subject4'
]

for subject_path in subject_paths:
    left_img_path = glob.glob(os.path.join(subject_path, 'subject*Left', '*.jpg'))[0]
    middle_img_path = glob.glob(os.path.join(subject_path, 'subject*Middle', '*.jpg'))[0]

    left_img = cv.imread(left_img_path)
    middle_img = cv.imread(middle_img_path)

    left_img_bg_removed = background_removal.remove_background(left_img)
    middle_img_bg_removed = background_removal.remove_background(middle_img)

    bg_removed_folder = os.path.join(subject_path, 'bg_removed')
    if not os.path.exists(bg_removed_folder):
        os.makedirs(bg_removed_folder)

    left_img_bg_removed_path = os.path.join(bg_removed_folder, 'left_bg_removed.jpg')
    middle_img_bg_removed_path = os.path.join(bg_removed_folder, 'middle_bg_removed.jpg')

    cv.imwrite(left_img_bg_removed_path, left_img_bg_removed)
    cv.imwrite(middle_img_bg_removed_path, middle_img_bg_removed)

    rectified_left, rectified_middle, Q_left_middle = stereocal.stereo_rectification(
        CM1, dist1, CM2, dist2, R_left_middle, T_left_middle, left_img_bg_removed_path, middle_img_bg_removed_path
    )

    if rectified_left is not None and rectified_middle is not None:
        rectified_folder = os.path.join(subject_path, 'rectified')
        if not os.path.exists(rectified_folder):
            os.makedirs(rectified_folder)

        rectified_left_path = os.path.join(rectified_folder, 'rectified_left.jpg')
        rectified_middle_path = os.path.join(rectified_folder, 'rectified_middle.jpg')

        cv.imwrite(rectified_left_path, rectified_left)
        cv.imwrite(rectified_middle_path, rectified_middle)

        # Stereo matching
        disparity_map = stereo_matching.stereoMatchingBM(rectified_left, rectified_middle)

        disparity_map_clipped = np.clip(disparity_map, 0, 255)
        plt.imshow(disparity_map, cmap='viridis')
        plt.colorbar()
        plt.title('Disparity Map')
        plt.show()

        # Depth map creation
        depth_map = depth_map_creator.create_depth_map(disparity_map, Q_left_middle)
        depth_map_output_path = os.path.join(rectified_folder, 'depth_map_left_middle.png')
        depth_map_creator.plot_depth_map(depth_map, depth_map_output_path)

        # 3D Mesh generation
        mesh_output_path = os.path.join(rectified_folder, '3d_mesh.png')
        mesh_generator.generate_mesh(depth_map, mesh_output_path)
    else:
        print(f"Rectification failed for images: {left_img_path}, {middle_img_path}")

for subject_path in subject_paths:
    right_img_path = glob.glob(os.path.join(subject_path, 'subject*Right', '*.jpg'))[0]
    middle_img_path = glob.glob(os.path.join(subject_path, 'subject*Middle', '*.jpg'))[0]

    right_img = cv.imread(right_img_path)
    middle_img = cv.imread(middle_img_path)

    right_img_bg_removed = background_removal.remove_background(right_img)
    middle_img_bg_removed = background_removal.remove_background(middle_img)

    bg_removed_folder = os.path.join(subject_path, 'bg_removed')
    if not os.path.exists(bg_removed_folder):
        os.makedirs(bg_removed_folder)

    right_img_bg_removed_path = os.path.join(bg_removed_folder, 'right_bg_removed.jpg')
    middle_img_bg_removed_path = os.path.join(bg_removed_folder, 'middle_bg_removed.jpg')

    cv.imwrite(right_img_bg_removed_path, right_img_bg_removed)
    cv.imwrite(middle_img_bg_removed_path, middle_img_bg_removed)

    rectified_right, rectified_middle, Q_right_middle = stereocal.stereo_rectification(
        CM1, dist1, CM2, dist2, R_middle_right, T_middle_right, right_img_bg_removed_path, middle_img_bg_removed_path
    )

    if rectified_right is not None and rectified_middle is not None:
        rectified_folder = os.path.join(subject_path, 'rectified')
        if not os.path.exists(rectified_folder):
            os.makedirs(rectified_folder)

        rectified_right_path = os.path.join(rectified_folder, 'rectified_right.jpg')
        rectified_middle_path = os.path.join(rectified_folder, 'rectified_middle.jpg')

        cv.imwrite(rectified_right_path, rectified_right)
        cv.imwrite(rectified_middle_path, rectified_middle)

        # Stereo matching
        disparity_map = stereo_matching.stereoMatchingBM(rectified_right, rectified_middle)

        disparity_map_clipped = np.clip(disparity_map, 0, 255)
        plt.imshow(disparity_map_clipped, cmap='viridis')
        plt.title('Disparity Map')
        plt.colorbar()
        plt.show()

        # Depth map creation
        depth_map = depth_map_creator.create_depth_map(disparity_map, Q_right_middle)
        depth_map_output_path = os.path.join(rectified_folder, 'depth_map_right_middle.png')
        depth_map_creator.plot_depth_map(depth_map, depth_map_output_path)

        # 3D Mesh generation
        mesh_output_path = os.path.join(rectified_folder, '3d_mesh.png')
        mesh_generator.generate_mesh(depth_map, mesh_output_path)
    else:
        print(f"Rectification failed for images: {left_img_path}, {middle_img_path}")
