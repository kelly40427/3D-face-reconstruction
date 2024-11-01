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
        
        #Z = fB/d
        depth_map = (focal_length * baseline) / disparity_map

        # Replace inf or very high values with a large, finite value
        depth_map[depth_map == np.inf] = 0
        depth_map[depth_map < 0 ] = 0
        depth_map[depth_map > 10000] = 10000  # Set a reasonable threshold for large depth values
        
        
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


    def create_3dpoint_cloud2(self, depth_map, image, K, total_disparity):
        """
        Generate colored point-cloud from depth map and image
        depth_map: (H x W)
        image: rectified RGBA image (H x W x 4)
        K: intrinsic matrix
        total_disparity: Offset along the X axis of image
        """
        h, w = depth_map.shape

        # intrinsic parameters to get focal length and principal point
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # empty point cloud
        point_cloud = []

        # Traversal oer the pixels
        for v in range(h):
            for u in range(w):
                Z = depth_map[v, u]
                
                # skip invalid depth(negative)
                if Z <= 0:
                    continue
                
                # Calculate 3D coordinate
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                
                # getting correspond color
                if u-total_disparity<0:
                    continue
                    color = [0,0,0]
                else:
                    ucolor = u-total_disparity
                    color = image[v, ucolor]
                    if color[3]==0:
                        continue
                        color = [0,0,0]
                
                # combine points with color
                point_cloud.append([X, Y, Z, color[2], color[1], color[0]])

        point_cloud = np.array(point_cloud)
        points = point_cloud[:, :3]
        colors = point_cloud[:, 3:] / 255.0  # normalize color to [0, 1]

        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        print("Number of points in point cloud:", np.asarray(pcd.points).shape[0])
    
        return pcd
