import cv2 as cv
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
import os
import open3d as o3d
import copy


# Import the classes
from colour_normalisation import ColourNorm
from StereoCalibration import StereoCal
from backgroundRemoval import background
from stereoMatching import stereoMatching
from DepthMapCreator import DepthMapCreator
from MeshGenerator import MeshGenerator  
from ICPCombine import ICPCombine

# Define the classes
colournorm = ColourNorm()
stereocal = StereoCal()
background_removal = background()
stereo_matching = stereoMatching()
depth_map_creator = DepthMapCreator()
mesh_generator = MeshGenerator()  
icp_combine = ICPCombine()

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
pcd_left_middle_list = []
pcd_middle_right_list = []

for subject_path in subject_paths:
    left_img_path = glob.glob(os.path.join(subject_path, 'subject*Left', '*.jpg'))[0]
    middle_img_path = glob.glob(os.path.join(subject_path, 'subject*Middle', '*.jpg'))[0]

    left_img = cv.imread(left_img_path)
    middle_img = cv.imread(middle_img_path)

    left_img_bg_removed = background_removal.remove_background(left_img)
    middle_img_bg_removed = background_removal.remove_background(middle_img)

    # convert color to RGB
    left_img_bg_removed_RGB = cv.cvtColor(left_img_bg_removed, cv.COLOR_BGR2RGB)
    middle_img_bg_removed_RGB = cv.cvtColor(middle_img_bg_removed, cv.COLOR_BGR2RGB)

    bg_removed_folder = os.path.join(subject_path, 'bg_removed')
    if not os.path.exists(bg_removed_folder):
        os.makedirs(bg_removed_folder)

    left_img_bg_removed_path = os.path.join(bg_removed_folder, 'left_bg_removed.jpg')
    middle_img_bg_removed_path = os.path.join(bg_removed_folder, 'middle_bg_removed.jpg')

    cv.imwrite(left_img_bg_removed_path, left_img_bg_removed_RGB)
    cv.imwrite(middle_img_bg_removed_path, middle_img_bg_removed_RGB)

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
        #disparity_map = stereo_matching.stereoMatching(rectified_left, rectified_middle,15,15)
        unreliable_disparity_map = stereo_matching.unreliable_disparity_mask(disparity_map)
        filtered_disparity_map = stereo_matching.filter_disparity(disparity_map,unreliable_disparity_map)

        disparity_map_clipped = np.clip(disparity_map, 0, 255)
        plt.imshow(filtered_disparity_map, cmap='viridis')
        plt.colorbar()
        plt.title('Disparity Map')
        plt.show()

        # Depth map creation
        focal_length_middle = (CM2[0,0] + CM2[1,1]) / 2
        baseline_left_middle = np.linalg.norm(T_left_middle)

        depth_map = depth_map_creator.create_depth_map(disparity_map, Q_left_middle, focal_length_middle, baseline_left_middle)
        depth_map_output_path = os.path.join(rectified_folder, 'depth_map_left_middle.png')
        depth_map_creator.plot_depth_map(depth_map, depth_map_output_path)

        # 3D Mesh generation
        mesh_output_path = os.path.join(rectified_folder, '3d_mesh.png')
        mesh_generator.generate_mesh(depth_map, mesh_output_path)

        # pcd creation
        #points_3d, valid_mask = depth_map_creator.create_3d_points(filtered_disparity_map, Q_left_middle)
        #colors = rectified_middle[valid_mask].reshape(-1, 3) 
        #pcd_left_middle = depth_map_creator.create_3dpoint_cloud(points_3d, colors)
        pcd_left_middle = depth_map_creator.create_3dpoint_cloud2(depth_map, rectified_middle, CM2)
        o3d.visualization.draw_geometries([pcd_left_middle], window_name="Colored Point Cloud with Normals")
        pcd_left_middle_list.append(pcd_left_middle)

    else:
        print(f"Rectification failed for images: {left_img_path}, {middle_img_path}")

for subject_path in subject_paths:
    right_img_path = glob.glob(os.path.join(subject_path, 'subject*Right', '*.jpg'))[0]
    middle_img_path = glob.glob(os.path.join(subject_path, 'subject*Middle', '*.jpg'))[0]

    right_img = cv.imread(right_img_path)
    middle_img = cv.imread(middle_img_path)

    right_img_bg_removed = background_removal.remove_background(right_img)
    middle_img_bg_removed = background_removal.remove_background(middle_img)

    # convert color to RGB
    middle_img_bg_removed_RGB2 = cv.cvtColor(middle_img_bg_removed, cv.COLOR_BGR2RGB)
    right_img_bg_removed_RGB2 = cv.cvtColor(right_img_bg_removed, cv.COLOR_BGR2RGB)

    bg_removed_folder = os.path.join(subject_path, 'bg_removed')
    if not os.path.exists(bg_removed_folder):
        os.makedirs(bg_removed_folder)

    right_img_bg_removed_path = os.path.join(bg_removed_folder, 'right_bg_removed.jpg')
    middle_img_bg_removed_path = os.path.join(bg_removed_folder, 'middle_bg_removed.jpg')

    cv.imwrite(right_img_bg_removed_path, right_img_bg_removed_RGB2)
    cv.imwrite(middle_img_bg_removed_path, middle_img_bg_removed_RGB2)

    rectified_right, rectified_middle, Q_right_middle = stereocal.stereo_rectification(
        CM_middle, dist_middle, CM_right, dist_right, R_middle_right, T_middle_right, right_img_bg_removed_path, middle_img_bg_removed_path
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
        disparity_map = stereo_matching.stereoMatchingBM(rectified_right,rectified_middle)
        unreliable_disparity_map = stereo_matching.unreliable_disparity_mask(disparity_map)
        filtered_disparity_map = stereo_matching.filter_disparity(disparity_map,unreliable_disparity_map)

        disparity_map_clipped = np.clip(disparity_map, 0, 255)
        plt.imshow(disparity_map_clipped, cmap='viridis')
        plt.title('Disparity Map')
        plt.colorbar()
        plt.show()

        # Depth map creation
        focal_length_middle2 = (CM_middle[0,0] + CM_middle[1,1]) / 2
        baseline_right_middle = np.linalg.norm(T_middle_right)

        depth_map = depth_map_creator.create_depth_map(disparity_map, Q_right_middle, focal_length_middle2, baseline_right_middle)
        depth_map_output_path = os.path.join(rectified_folder, 'depth_map_right_middle.png')
        depth_map_creator.plot_depth_map(depth_map, depth_map_output_path)

        # 3D Mesh generation
        mesh_output_path = os.path.join(rectified_folder, '3d_mesh.png')
        mesh_generator.generate_mesh(depth_map, mesh_output_path)

        # pcd creation
        #points_3d, valid_mask = depth_map_creator.create_3d_points(filtered_disparity_map, Q_left_middle)
        #colors = rectified_middle[valid_mask].reshape(-1, 3) 
        #pcd_middle_right = depth_map_creator.create_3dpoint_cloud(points_3d, colors)
        pcd_middle_right = depth_map_creator.create_3dpoint_cloud2(depth_map, rectified_middle, CM_middle)
        o3d.visualization.draw_geometries([pcd_middle_right], window_name="Colored Point Cloud with Normals")
        pcd_middle_right_list.append(pcd_middle_right)

    else:
        print(f"Rectification failed for images: {left_img_path}, {middle_img_path}")

for pcd_left, pcd_right in zip(pcd_left_middle_list, pcd_middle_right_list):
    threshold = 0.05
    # trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
    #                     [-0.139, 0.967, -0.215, 0.7],
    #                     [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    trans_init = np.eye(4)
    reg_p2p = icp_combine.point_to_point_icp(pcd_right, pcd_left, threshold, trans_init)
    print("Transformation Matrix:")
    print(reg_p2p)

    # 視覺化結果
    icp_combine.draw_registration_result(pcd_right, pcd_left, reg_p2p)


