# MeshGenerator.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MeshGenerator:
    def __init__(self):
        pass

    def generate_mesh(self, depth_map, output_path=None):
        """
        Generate and plot a 3D mesh from the depth map.
        
        Args:
        - depth_map: 2D numpy array representing the depth map.
        - output_path: Optional path to save the 3D mesh plot as an image.
        """
        # Generate mesh grid for the X, Y coordinates
        height, width = depth_map.shape
        x = np.linspace(0, width, width)
        y = np.linspace(0, height, height)
        X, Y = np.meshgrid(x, y)
        Z = depth_map

        # Plotting the 3D surface mesh
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')

        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Depth (Z)')
        ax.set_title('3D Mesh from Depth Map')

        # Show or save the plot
        if output_path:
            plt.savefig(output_path)
        plt.show()

    





