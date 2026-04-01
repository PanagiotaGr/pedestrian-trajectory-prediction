import os
from ground_plane_utils import GroundPlane

path = os.path.expanduser("~/imptc_project/ground_plane/xung_ground_plane_02.ply")
gp = GroundPlane(path)

z = gp.query_height(0.0, 0.0)
print("Ground height at (0,0):", z)
