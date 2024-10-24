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
from backgroundRemoval import background
from stereoMatching import stereoMatching
from DepthMapCreator import DepthMapCreator
from MeshGenerator import MeshGenerator  
from ICPCombine import ICPCombine
from updated_bg_removal import ImprovedBackground

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

# CM1, dist1, CM2, dist2, R_left_middle, T_left_middle = stereocal.stereo_calibration(calib_left, calib_middle)
# CM_middle, dist_middle, CM_right, dist_right, R_middle_right, T_middle_right = stereocal.stereo_calibration(calib_middle, calib_right)


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

        images_left_person = glob.glob(os.path.join(subject_path, 'subject*Left', '*.jpg'))[:1]
        images_middle_person = glob.glob(os.path.join(subject_path, 'subject*Middle', '*.jpg'))[:1]
        images_right_person = glob.glob(os.path.join(subject_path, 'subject*Right', '*.jpg'))[:1]


        for img_left_path, img_middle_path, img_right_path in zip(images_left_person, images_middle_person, images_right_person):

            img_left = cv.imread(img_left_path)
            img_middle = cv.imread(img_middle_path)
            img_right = cv.imread(img_right_path)

            # make the image colour correct
            img_left = cv.cvtColor(img_left, cv.COLOR_BGR2RGB)
            img_middle = cv.cvtColor(img_middle, cv.COLOR_BGR2RGB)
            img_right = cv.cvtColor(img_right, cv.COLOR_BGR2RGB)

            # Normalization
            images_for_normalization = [img_left, img_middle, img_right]
            normalized_images = colournorm.normalize_images(images_for_normalization)
            img_left, img_middle, img_right = normalized_images[0], normalized_images[1], normalized_images[2]

            cv2.imwrite('ipcv_project3/subject1/subject1Left/subject1_Left_1_normalized.png', normalized_images[0])

            # Determine if the subject has special conditions
            is_bald = 'subject4' in subject_path
            is_left = 'Left' in img_left_path

            # Remove background
            left_img_bg_removed = background_removal.remove_background(img_left, is_bald=is_bald, is_left=is_left)
            middle_img_bg_removed = background_removal.remove_background(img_middle, is_bald=is_bald)
            right_img_bg_removed = background_removal.remove_background(img_right,is_bald=is_bald)

            bg_removed_folder = os.path.join(subject_path, 'bg_removed')
            if not os.path.exists(bg_removed_folder):
                os.makedirs(bg_removed_folder)

            left_img_bg_removed_path = os.path.join(bg_removed_folder, 'left_bg_removed.png')
            middle_img_bg_removed_path = os.path.join(bg_removed_folder, 'middle_bg_removed.png')
            right_img_bg_removed_path = os.path.join(bg_removed_folder, 'right_bg_removed.png')

            cv.imwrite(left_img_bg_removed_path, left_img_bg_removed)
            cv.imwrite(middle_img_bg_removed_path, middle_img_bg_removed)
            cv.imwrite(right_img_bg_removed_path, right_img_bg_removed)

            #convert color to BGR
            left_img_bg_removed_BGR = cv.cvtColor(left_img_bg_removed, cv.COLOR_BGRA2BGR)
            middle_img_bg_removed_BGR = cv.cvtColor(middle_img_bg_removed, cv.COLOR_BGRA2BGR)
            right_img_bg_removed_BGR = cv.cvtColor(right_img_bg_removed, cv.COLOR_BGRA2BGR)

            #Stereo rectification
            rectified_left, rectified_middle = stereocal.rectify_images(left_img_bg_removed_BGR, middle_img_bg_removed_BGR, map1_x, map1_y, map2_x, map2_y)
            rectified_middle_new, rectified_right = stereocal.rectify_images(middle_img_bg_removed_BGR, right_img_bg_removed_BGR, map2_x_new, map2_y_new, map3_x, map3_y)

            if rectified_left is not None and rectified_middle is not None and rectified_right is not None:

                rectified_folder = os.path.join(subject_path, 'rectified')
                if not os.path.exists(rectified_folder):
                    os.makedirs(rectified_folder)

                rectified_left_path = os.path.join(rectified_folder, 'rectified_left.jpg')
                rectified_middle_path = os.path.join(rectified_folder, 'rectified_middle.jpg')
                rectified_right_path = os.path.join(rectified_folder, 'rectified_right.jpg')

                cv.imwrite(rectified_left_path, rectified_left)
                cv.imwrite(rectified_middle_path, rectified_middle)
                cv.imwrite(rectified_right_path, rectified_right)

                if cnt == 1:
                    mindisparity1 = 276
                    mindisparity2 = 276
                    numdisparity1 = 16*4
                    numdisparity2 = 16*4
                    blockSize1 = 5
                    blockSize2 = 5
                    uniquenessRatio = 5
                elif cnt == 2:
                    mindisparity1 = 292
                    mindisparity2 = 292
                    numdisparity1 = 16*5
                    numdisparity2 = 16*5
                    blockSize1 = 11
                    blockSize2 = 9
                    uniquenessRatio = 1
                elif cnt == 3:
                    mindisparity1 = 318
                    mindisparity2 = 346
                    numdisparity1 = 16*7
                    numdisparity2 = 16*5
                    blockSize1 = 15
                    blockSize2 = 17
                    uniquenessRatio = 5

                # Stereo matching
                # diparity left-middle
                total_disparity_left_middle, disparity_map_left_middle = stereo_matching.stereoMatchingBM(rectified_left, rectified_middle, mindisparity1, numdisparity1,blockSize1,uniquenessRatio)
                #disparity_map = stereo_matching.stereoMatching(rectified_left, rectified_middle,15,15)
                unreliable_disparity_map_left_middle = stereo_matching.unreliable_disparity_mask(disparity_map_left_middle)
                filtered_disparity_map_left_middle = stereo_matching.filter_disparity(disparity_map_left_middle,unreliable_disparity_map_left_middle)
                
                # disparity middle-right
                total_disparity_middle_right, disparity_map_middle_right = stereo_matching.stereoMatchingBM(rectified_middle_new, rectified_right, mindisparity2, numdisparity2, blockSize2, uniquenessRatio)
                unreliable_disparity_map_middle_right = stereo_matching.unreliable_disparity_mask(disparity_map_middle_right)
                filtered_disparity_map_middle_right = stereo_matching.filter_disparity(disparity_map_middle_right, unreliable_disparity_map_middle_right)

                #disparity_map_left_middle_clipped = np.clip(disparity_map_left_middle, 0, 255)
                #disparity_map_middle_right_clipped = np.clip(disparity_map_middle_right, 0, 255)

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
                focal_length = (camera_matrix_middle[0,0] + camera_matrix_middle[1,1]) / 2
                baseline_left_middle = np.linalg.norm(T_left_middle)          
                baseline_middle_right = np.linalg.norm(T_middle_right)

                # depth map left-middle
                depth_map_left_middle = depth_map_creator.create_depth_map(disparity_map_left_middle, Q, focal_length, baseline_left_middle)
                depth_map_output_path = os.path.join(rectified_folder, 'depth_map_left_middle.png')
                depth_map_creator.plot_depth_map(depth_map_left_middle, depth_map_output_path)

                
                depth_map_middle_right = depth_map_creator.create_depth_map(disparity_map_middle_right, Q2, focal_length, baseline_middle_right)
                depth_map_output_path = os.path.join(rectified_folder, 'depth_map_middle_right.png')
                depth_map_creator.plot_depth_map(depth_map_middle_right, depth_map_output_path)


                # pcd creation
                # pcd_left_middle
                pcd_left_middle = depth_map_creator.create_3dpoint_cloud2(depth_map_left_middle, rectified_middle, camera_matrix_middle, total_disparity_left_middle)
                o3d.visualization.draw_geometries([pcd_left_middle], window_name="Colored Point Cloud with Normals")

                # pcd_middle_right
                pcd_middle_right = depth_map_creator.create_3dpoint_cloud2(depth_map_middle_right, rectified_right, camera_matrix_middle, total_disparity_middle_right)
                o3d.visualization.draw_geometries([pcd_middle_right], window_name="Colored Point Cloud with Normals")

                #ICP combine
                threshold = 0.05
                beta = np.radians(0)
                trans_init1 = np.array(
                            [[np.cos(beta),   0,  np.sin(beta),   -47],
                            [0,              1,  0,              0],
                            [-np.sin(beta),  0,  np.cos(beta),   0],
                            [0,              0,  0,              1]])
                beta = np.radians(0)
                trans_init2 = np.array(
                            [[np.cos(beta),   0,  np.sin(beta),  0],
                            [0,              1,  0,              0],
                            [-np.sin(beta),  0,  np.cos(beta),   0],
                            [0,              0,  0,              1]])
                trans_init = trans_init1 @ trans_init2
                reg_p2p = icp_combine.point_to_point_icp(pcd_middle_right, pcd_left_middle, threshold, trans_init)
                print("Transformation Matrix:")
                print(reg_p2p)
                # 視覺化結果
                pcd_middle_right.transform(reg_p2p)
                newpointcloud = pcd_middle_right + pcd_left_middle 
                o3d.visualization.draw_geometries([newpointcloud])
                mesh_path = os.path.join(rectified_folder, 'mesh_icp.ply')
                o3d.io.write_point_cloud(mesh_path, newpointcloud)

                output_mesh_path = os.path.join(rectified_folder, "combined_mesh.ply")
                mesh_generator.surface_reconstruction(newpointcloud, output_mesh_path)
