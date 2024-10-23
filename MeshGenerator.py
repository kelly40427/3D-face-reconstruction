# MeshGenerator.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import os

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
    
    def surface_reconstruction(self, pcd, file_path):
        
        # 1. 預處理：估算法向量
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # 2. 對齊法向量方向，使法向量指向一致
        pcd.orient_normals_consistent_tangent_plane(k=10)

        # 3. 執行泊松重建
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, width=0, scale=1.2, linear_fit=False)
        
        # 4. 移除低密度的頂點來清理網格
        vertices_to_remove = densities < np.quantile(densities, 0.02)  # 保留高密度的頂點
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # 5. 裁剪操作：使用點雲的對齊邊界框來裁剪泊松生成的網格
        bbox = pcd.get_axis_aligned_bounding_box()  # 獲取點雲的軸對齊邊界框
        cropped_mesh = mesh.crop(bbox)  # 使用邊界框裁剪網格

        # 6. 可視化裁剪後的網格
        o3d.visualization.draw_geometries([cropped_mesh], window_name="Cropped Poisson Mesh", width=800, height=600)

        o3d.io.write_triangle_mesh(file_path, cropped_mesh)
        
        return cropped_mesh

    





