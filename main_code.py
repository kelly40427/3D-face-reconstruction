import cv2
import cv2 as cv
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
import os
import open3d as o3d
import copy
from tqdm import tqdm

# Import the classes
from colour_normalisation import ColourNorm
from StereoCalibration import StereoCal
from stereoMatching import stereoMatching
from DepthMapCreator import DepthMapCreator
from MeshGenerator import MeshGenerator  
from ICPCombine import ICPCombine
from BackgroundRemoval import ImprovedBackground

# Define the classes
colournorm = ColourNorm()
stereocal = StereoCal()
background_removal = ImprovedBackground()
stereo_matching = stereoMatching()  
depth_map_creator = DepthMapCreator()
mesh_generator = MeshGenerator()  
icp_combine = ICPCombine()

# Paths to calibration images
calib_left = 'ipcv_project3/Calibratie 1/calibrationLeft'
calib_middle = 'ipcv_project3/Calibratie 1/calibrationMiddle'
calib_right = 'ipcv_project3/Calibratie 1/calibrationRight'

# Paths to subjects
subject_paths = [
    'ipcv_project3/subject1',
    'ipcv_project3/subject2',
    'ipcv_project3/subject4'
]

images_left = glob.glob(os.path.join(calib_left, '*.jpg'))
images_middle = glob.glob(os.path.join(calib_middle, '*.jpg'))
images_right = glob.glob(os.path.join(calib_right, '*.jpg'))

chessboard_size = (9, 6)
square_size = 10

sample_image = cv.imread(images_left[0])
image_size = (sample_image.shape[1], sample_image.shape[0])

# Camera calibration
camera_matrix_left, dist_coeffs_left, camera_matrix_middle, dist_coeffs_middle, \
camera_matrix_right, dist_coeffs_right, objpoints, imgpoints_left, \
imgpoints_middle, imgpoints_right = stereocal.CameraCalibrate(
    images_left, images_middle, images_right, chessboard_size, square_size
)

# Left-middle calibration
R1, R2, P1, P2, Q, map1_x, map1_y, map2_x, map2_y, T_left_middle = stereocal.stereo_calibrate_and_rectify(
    objpoints, imgpoints_left, imgpoints_middle, camera_matrix_left, dist_coeffs_left,
    camera_matrix_middle, dist_coeffs_middle, image_size)

# Middle-right calibration
R2_new, R3, P2_new, P3, Q2, map2_x_new, map2_y_new, map3_x, map3_y, T_middle_right = stereocal.stereo_calibrate_and_rectify(
    objpoints, imgpoints_middle, imgpoints_right, camera_matrix_middle, dist_coeffs_middle,
    camera_matrix_right, dist_coeffs_right, image_size)

cnt = 0

for subject_path in subject_paths:
    cnt += 1

    print(f"Processing {subject_path}...")

    # Get all images without limiting to first one
    images_left_person = glob.glob(os.path.join(subject_path, 'subject*Left', '*.jpg'))
    images_middle_person = glob.glob(os.path.join(subject_path, 'subject*Middle', '*.jpg'))
    images_right_person = glob.glob(os.path.join(subject_path, 'subject*Right', '*.jpg'))

    # Sort the images to ensure matching order
    images_left_person.sort()
    images_middle_person.sort()
    images_right_person.sort()

    # Create output folders
    bg_removed_folder = os.path.join(subject_path, 'bg_removed')
    rectified_folder = os.path.join(subject_path, 'rectified')
    for folder in [bg_removed_folder, rectified_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # for img_left_path, img_middle_path, img_right_path in zip(images_left_person, images_middle_person, images_right_person):
    for idx, (img_left_path, img_middle_path, img_right_path) in enumerate(
            zip(images_left_person, images_middle_person, images_right_person)):
        base_number = os.path.basename(img_left_path).split('_')[-1].split('.')[0]  # Extract number from filename
        print(f"Processing image set {idx + 1}/{len(images_left_person)} (number {base_number})")

        img_left = cv.imread(img_left_path)
        img_middle = cv.imread(img_middle_path)
        img_right = cv.imread(img_right_path)

        # Normalization
        images_for_normalization = [img_left, img_middle, img_right]
        normalized_images = colournorm.normalize_images(images_for_normalization)
        img_left, img_middle, img_right = normalized_images[0], normalized_images[1], normalized_images[2]

        # Determine if the subject has special conditions
        is_bald = 'subject4' in subject_path
        is_left = True  # Always true for left images

        # Remove background
        left_img_bg_removed = background_removal.remove_background(img_left, is_bald=is_bald, is_left=is_left)
        middle_img_bg_removed = background_removal.remove_background(img_middle, is_bald=is_bald)
        right_img_bg_removed = background_removal.remove_background(img_right, is_bald=is_bald)

        # Generate numbered output filenames
        left_img_bg_removed_path = os.path.join(bg_removed_folder, f'left_bg_removed_{base_number}.png')
        middle_img_bg_removed_path = os.path.join(bg_removed_folder, f'middle_bg_removed_{base_number}.png')
        right_img_bg_removed_path = os.path.join(bg_removed_folder, f'right_bg_removed_{base_number}.png')

        # Save background removed images
        cv2.imwrite(left_img_bg_removed_path, left_img_bg_removed)
        cv2.imwrite(middle_img_bg_removed_path, middle_img_bg_removed)
        cv2.imwrite(right_img_bg_removed_path, right_img_bg_removed)

        print(f"Saved background removed images for set {base_number}")

        #left_img_bg_removed_BGR = cv2.cvtColor(left_img_bg_removed, cv2.COLOR_RGB2BGR)
        #middle_img_bg_removed_BGR = cv2.cvtColor(middle_img_bg_removed, cv2.COLOR_RGB2BGR)
        #right_img_bg_removed_BGR = cv2.cvtColor(right_img_bg_removed, cv2.COLOR_RGB2BGR)
        

       # Stereo rectification
        rectified_left, rectified_middle = stereocal.rectify_images(
            left_img_bg_removed, middle_img_bg_removed, map1_x, map1_y, map2_x, map2_y)
        rectified_middle_new, rectified_right = stereocal.rectify_images(
            middle_img_bg_removed, right_img_bg_removed, map2_x_new, map2_y_new, map3_x, map3_y)

        if rectified_left is not None and rectified_middle is not None and rectified_right is not None:
            # Save rectified images with alpha channel
            cv2.imwrite(os.path.join(rectified_folder, f'rectified_left_{base_number}.png'), rectified_left)
            cv2.imwrite(os.path.join(rectified_folder, f'rectified_middle_{base_number}.png'), rectified_middle)
            cv2.imwrite(os.path.join(rectified_folder, f'rectified_right_{base_number}.png'), rectified_right)


            if cnt == 1:
                mindisparity1 = 276
                mindisparity2 = 276
                numdisparity1 = 16 * 4
                numdisparity2 = 16 * 4
                blockSize1 = 5
                blockSize2 = 5
                uniquenessRatio = 5
            elif cnt == 2:
                mindisparity1 = 292
                mindisparity2 = 292
                numdisparity1 = 16 * 5
                numdisparity2 = 16 * 5
                blockSize1 = 11
                blockSize2 = 9
                uniquenessRatio = 1
            elif cnt == 3:
                mindisparity1 = 318
                mindisparity2 = 346
                numdisparity1 = 16 * 7
                numdisparity2 = 16 * 5
                blockSize1 = 15
                blockSize2 = 17
                uniquenessRatio = 5

            # Stereo matching
            # diparity left-middle
            total_disparity_left_middle, disparity_map_left_middle = stereo_matching.stereoMatchingBM(rectified_left,
                                                                                                      rectified_middle,
                                                                                                      mindisparity1,
                                                                                                      numdisparity1,
                                                                                                    blockSize1,
                                                                                                      uniquenessRatio)
            # disparity_map = stereo_matching.stereoMatching(rectified_left, rectified_middle,15,15)
            unreliable_disparity_map_left_middle = stereo_matching.unreliable_disparity_mask(disparity_map_left_middle)
            filtered_disparity_map_left_middle = stereo_matching.filter_disparity(disparity_map_left_middle,
                                                                                  unreliable_disparity_map_left_middle)

            # disparity middle-right
            total_disparity_middle_right, disparity_map_middle_right = stereo_matching.stereoMatchingBM(
                rectified_middle_new, rectified_right, mindisparity2, numdisparity2, blockSize2, uniquenessRatio)
            unreliable_disparity_map_middle_right = stereo_matching.unreliable_disparity_mask(disparity_map_middle_right)
            filtered_disparity_map_middle_right = stereo_matching.filter_disparity(disparity_map_middle_right,
                                                                                   unreliable_disparity_map_middle_right)


            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(disparity_map_left_middle, cmap='jet')
            plt.colorbar()
            plt.title('Disparity Map: Left-Middle')

            plt.subplot(1, 2, 2)
            plt.imshow(disparity_map_middle_right, cmap='jet')
            plt.colorbar()
            plt.title('Disparity Map: Middle-Right')

            plt.show()

            # Depth map creation
            focal_length = (camera_matrix_middle[0, 0] + camera_matrix_middle[1, 1]) / 2
            baseline_left_middle = np.linalg.norm(T_left_middle)
            baseline_middle_right = np.linalg.norm(T_middle_right)

            # depth map left-middle
            depth_map_left_middle = depth_map_creator.create_depth_map(disparity_map_left_middle, Q, focal_length,
                                                                       baseline_left_middle)
            depth_map_output_path = os.path.join(rectified_folder, f'depth_map_left_middle_{base_number}.png')
            depth_map_creator.plot_depth_map(depth_map_left_middle, depth_map_output_path)

            depth_map_middle_right = depth_map_creator.create_depth_map(disparity_map_middle_right, Q2, focal_length,
                                                                        baseline_middle_right)
            depth_map_output_path = os.path.join(rectified_folder, f'depth_map_middle_right_{base_number}.png')
            depth_map_creator.plot_depth_map(depth_map_middle_right, depth_map_output_path)

            # pcd creation
            # pcd_left_middle
            pcd_left_middle = depth_map_creator.create_3dpoint_cloud2(depth_map_left_middle, rectified_middle,
                                                                      camera_matrix_middle, total_disparity_left_middle)
            o3d.visualization.draw_geometries([pcd_left_middle], window_name="Colored Point Cloud with Normals: pcd_left_middle")
            output_mesh_path = os.path.join(rectified_folder, f'left_middle_mesh_{base_number}.ply')
            mesh_generator.surface_reconstruction(pcd_left_middle, output_mesh_path)

            # pcd_middle_right
            pcd_middle_right = depth_map_creator.create_3dpoint_cloud2(depth_map_middle_right, rectified_right,
                                                                       camera_matrix_middle, total_disparity_middle_right)
            o3d.visualization.draw_geometries([pcd_middle_right], window_name="Colored Point Cloud with Normals: pcd_middle_right")
            output_mesh_path = os.path.join(rectified_folder, f'middle_right_mesh_{base_number}.ply')
            mesh_generator.surface_reconstruction(pcd_middle_right, output_mesh_path)

            # ICP combine
            threshold = 0.05
            beta = np.radians(0)
            trans_init1 = np.array(
                [[np.cos(beta), 0, np.sin(beta), -47],
                 [0, 1, 0, 0],
                 [-np.sin(beta), 0, np.cos(beta), 0],
                 [0, 0, 0, 1]])
            beta = np.radians(0)
            trans_init2 = np.array(
                [[np.cos(beta), 0, np.sin(beta), 0],
                 [0, 1, 0, 0],
                 [-np.sin(beta), 0, np.cos(beta), 0],
                 [0, 0, 0, 1]])
            trans_init = trans_init1 @ trans_init2
            reg_p2p = icp_combine.point_to_point_icp(pcd_middle_right, pcd_left_middle, threshold, trans_init)
            print("Transformation Matrix:")
            print(reg_p2p)

            # visualize
            pcd_middle_right.transform(reg_p2p) # apply transformation
            newpointcloud = pcd_middle_right + pcd_left_middle # combine point clouds
            o3d.visualization.draw_geometries([newpointcloud], window_name="new point cloud") # visualize
            mesh_path = os.path.join(rectified_folder, f'mesh_icp_{base_number}.ply') # save point cloud
            o3d.io.write_point_cloud(mesh_path, newpointcloud) # save point cloud

            output_mesh_path = os.path.join(rectified_folder, f'combined_mesh_{base_number}.ply') # save mesh
            mesh_generator.surface_reconstruction(newpointcloud, output_mesh_path) # uses Poisson surface reconstruction to make continues mesh
            
            print(f"Completed processing image set {base_number}")
