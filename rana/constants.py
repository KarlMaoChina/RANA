"""Shared geometric constants used by feature construction and training."""

NORM_MM = 64.0
SPAR_MM = 40.0
DT_CLAMP = 1.5
PLANE_RADII_MM = (60.0, 120.0, None)
MIN_PLANE_PTS = 100
MIN_PV_VOXELS = 5
ANCHOR_ORGANS = ("gallbladder", "liver", "portal_vein_and_splenic_vein")
CHANNEL_NAMES = ("sdf", "t1", "t2", "dtube")
