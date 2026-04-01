import numpy as np
import open3d as o3d


class GroundPlane:
    def __init__(self, ply_path):
        print("Loading ground plane:", ply_path)
        self.pcd = o3d.io.read_point_cloud(ply_path)

        pts = np.asarray(self.pcd.points)
        self.points = pts[:, :2]  # (x, y)
        self.z = pts[:, 2]

        # KD-tree για nearest neighbor
        self.kd = o3d.geometry.KDTreeFlann(self.pcd)

        print("Loaded points:", len(self.points))

    def query_height(self, x, y):
        """
        βρίσκει το κοντινότερο σημείο στο ground
        """
        query = np.array([x, y, 0.0])

        _, idx, _ = self.kd.search_knn_vector_3d(query, 1)
        i = idx[0]

        return float(self.z[i])
