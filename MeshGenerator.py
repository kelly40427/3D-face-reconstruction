# MeshGenerator.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 
import open3d as o3d

class MeshGenerator:
    def __init__(self):
        pass

    def generate_mesh(self, depth_map, output_path=None, image2D = None):
        """
        Generate and plot a 3D mesh from the depth map.
        
        Args:
        - depth_map: 2D numpy array representing the depth map.
        - output_path: Optional path to save the 3D mesh plot as an image.
        - image2D : the original 2d image to give color to the mesh
        """
        # Generate mesh grid for the X, Y coordinates   
        height, width = depth_map.shape
        x = np.linspace(0, width, width)
        y = np.linspace(0, height, height)
        X, Y = np.meshgrid(x, y)
        Z = depth_map

        #flippedImage2D = cv2.flip(image2D,1)
        image2D = cv2.cvtColor(image2D,cv2.COLOR_BGR2RGB)
        image_data = np.array(image2D)

        # Plotting the 3D surface mesh
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, facecolors=image_data/255.0, rstride=1, cstride=1)

        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Depth (Z)')
        ax.set_title('3D Mesh from Depth Map')

        ax.set_xticks(np.arange(0,1100,100))
        ax.set_yticks(np.arange(0,1100,100))
        ax.set_zticks(np.arange(0,1100,100))
        
        '''
        points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        o3d.io.write_point_cloud("point_cloud.pcd", point_cloud)
        '''
        # Show or save the plot
        if output_path:
            plt.savefig(output_path)
        plt.show()

    





