import cv2
import numpy as np
import matplotlib.pyplot as plt

class DepthMapCreator:

    def create_depth_map(self, disparity_map, Q):
        """
        Create a depth map by reprojecting the disparity map into 3D space using the Q matrix.
        """
        min_disparity=10
        max_disparity=64
        valid_mask = (disparity_map > min_disparity) & (disparity_map < max_disparity)
        # Reproject points to 3D space (Z axis corresponds to depth)
        points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
        
        # Extract the Z coordinate (depth) from the 3D points
        depth_map = points_3D[:, :, 2]
        depth_map = np.where(valid_mask, depth_map, 0)
        # Replace inf or very high values with a large, finite value
        depth_map[depth_map == np.inf] = 0
        depth_map[depth_map > 10000] = 10000  # Set a reasonable threshold for large depth values
        
        # Normalize depth map for better visualization
        # depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        # depth_map_normalized = np.uint8(depth_map_normalized)
        
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
