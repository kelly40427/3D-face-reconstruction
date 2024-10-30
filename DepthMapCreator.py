import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

class DepthMapCreator:

    def create_depth_map(self, disparity_map, Q, focal_length, baseline):
        """
        Create a depth map by reprojecting the disparity map into 3D space using the Q matrix.
        """
        min_disparity = 10
        max_disparity = min_disparity+1024
        valid_mask = (disparity_map > min_disparity) & (disparity_map < max_disparity)
        # Reproject points to 3D space (Z axis corresponds to depth)
        #points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
        
        # Extract the Z coordinate (depth) from the 3D points
        #depth_map = points_3D[:, :, 2]
        #depth_map = np.where(valid_mask, depth_map, 0)
        
        #Z = fB/d
        depth_map = (focal_length * baseline) / disparity_map

        # Replace inf or very high values with a large, finite value
        depth_map[depth_map == np.inf] = 0
        depth_map[depth_map < 0 ] = 0
        depth_map[depth_map > 10000] = 10000  # Set a reasonable threshold for large depth values
        
        # Normalize depth map for better visualization
        # depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        # depth_map_normalized = np.uint8(depth_map_normalized)
        
        # Define a kernel for morphological operations
        kernel = np.ones((25, 25), np.uint8)

        # Apply morphological closing
        depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, kernel)
        
        # Apply bilateral filter to smooth disparity map
        depth_map = cv2.bilateralFilter(depth_map, 9, 5, 5)

        return depth_map


    def plot_depth_map(self, depth_map, output_path):
        """
        Save and plot the depth map using matplotlib.
        """
        plt.figure(figsize=(10, 5))
        plt.imshow(depth_map, cmap='magma')  # Use 'magma' colormap for depth visualization
        plt.colorbar()
        plt.title('Depth Map')
        plt.savefig(output_path)
        plt.show()

    def create_3d_points(self, disparity_map, Q):

        points_3d = cv2.reprojectImageTo3D(disparity_map, Q)

        # 建立有效視差範圍的遮罩
        min_disparity = 0
        max_disparity = min_disparity+1024
        valid_mask = (disparity_map > min_disparity) & (disparity_map < max_disparity)

        # 同時使用遮罩篩選 points_3D
        points_3d = points_3d[valid_mask]
          
        return points_3d, valid_mask
    
    def create_3dpoint_cloud(self, points_3d, colors):
        """
        Create an Open3D point cloud object from 3D points and colors.
        """

        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d.reshape(-1, 3))  # Reshape points into Nx3
        # Normalize the color values (if not already) and assign to point cloud
        pcd.colors = o3d.utility.Vector3dVector(colors/ 255.0)  # Reshape and normalize colors
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        print("Number of points in point cloud:", np.asarray(pcd.points).shape[0])
    
        return pcd

    def create_3dpoint_cloud2(self, depth_map, image, K, total_disparity):
        """
        通过深度图和RGB图像生成带颜色的点云
        depth_map: 深度图 (H x W)
        image: 对齐的RGB图像 (H x W x 3)
        K: 相机内参矩阵
        """
        h, w = depth_map.shape

        # 相机内参
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 创建空的点云列表
        point_cloud = []

        # 遍历每个像素
        for v in range(h):
            for u in range(w):
                Z = depth_map[v, u]
                
                # 跳过无效深度
                if Z <= 0:
                    continue
                
                # 计算三维坐标
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                
                # 获取对应像素的颜色
                if u-total_disparity<0:
                    continue
                    color = [0,0,0]
                else:
                    ucolor = u-total_disparity
                    color = image[v, ucolor]
                    if color[3]==0:
                        continue
                        color = [0,0,0]
                
                # 将三维点和颜色组合起来
                # point_cloud.append([X, Y, Z, color[2], color[1], color[1]])
                point_cloud.append([X, Y, Z, color[2], color[1], color[0]])

        point_cloud = np.array(point_cloud)
        points = point_cloud[:, :3]
        colors = point_cloud[:, 3:] / 255.0  # 将颜色归一化到 [0, 1]

        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        print("Number of points in point cloud:", np.asarray(pcd.points).shape[0])
    
        return pcd
