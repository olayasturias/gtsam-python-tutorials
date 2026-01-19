import os
import json
import numpy as np
from pathlib import Path

def skew(p):
    return np.array([
        [0.0, -p[2], p[1]],
        [p[2], 0.0, -p[0]],
        [-p[1], p[0], 0.0]
    ], dtype=np.float64)

def convert_to_gtsam_coords(points):
    """Convert points to GTSAM coordinates"""
    gtsam_points = []
    for pt in points:
        pt_gtsam = np.array([pt[1], pt[2], pt[0]], dtype=np.float64).reshape(3, )
        gtsam_points.append(pt_gtsam)
    return gtsam_points

def parse_graph_file(filename: str):
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir.parent / "data" / filename

    with data_path.open("r") as f:
        values = json.load(f)

    from_idx = values["from_idx"] # index 1 in corresponding variables
    frame_idx = values["frame_idx"] # index 2 in corresponding variables

    # read values
    obs_Tc1 = values["obs_Tc_1"]
    obs1_covTc = values["obs1_covTc"]
    obs_Tc2 = values["obs_Tc_2"]
    obs2_covTc = values["obs2_covTc"]

    # convert values to gtsam coords
    obs_Tc1= convert_to_gtsam_coords(obs_Tc1)
    obs_Tc2 = convert_to_gtsam_coords(obs_Tc2)

    return from_idx,frame_idx, obs_Tc1, obs_Tc2, obs1_covTc, obs2_covTc