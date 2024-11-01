import copy
import open3d as o3d

class ICPCombine:
    def point_to_point_icp(self, source, target, threshold, trans_init):
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        return reg_p2p.transformation
