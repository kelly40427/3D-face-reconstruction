import copy
import open3d as o3d

class ICPCombine:
    
    # Initialize functions
    def draw_registration_result(self, source, target, transformation):
        """
        param: source - source point cloud
        param: target - target point cloud
        param: transformation - 4 X 4 homogeneous transformation matrix
        """
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)

        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp], zoom=0.4459, front=[0.9288, -0.2951, -0.2242], lookat=[1.6784, 2.0612, 1.4451], up=[-0.3402, -0.9189, -0.1996])

    
    def point_to_point_icp(self, source, target, threshold, trans_init):
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        return reg_p2p.transformation

    def point_to_plane_icp(self, source, target, threshold, trans_init):
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        return reg_p2l.transformation