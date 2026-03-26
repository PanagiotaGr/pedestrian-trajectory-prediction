import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

pcd = o3d.io.read_point_cloud("data/map.ply")
print(pcd)

points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)


n = min(50000, len(points))
idx = np.random.choice(len(points), n, replace=False)
pts = points[idx]
cols = colors[idx]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.savefig("map_preview.png", dpi=300)
plt.show()
