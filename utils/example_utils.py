import os
import json
import numpy as np
import rerun as rr
from pathlib import Path

def skew(p):
    return np.array([
        [0.0, -p[2], p[1]],
        [p[2], 0.0, -p[0]],
        [-p[1], p[0], 0.0]
    ], dtype=np.float64)

def residuals_for_pose(graph, values, pose_key, landmark_keys, obs_list):
    norms = []
    for i, lk in enumerate(landmark_keys):
        pose = values.atPose3(pose_key)
        L = values.atPoint3(lk)
        pred = pose.transformTo(L)
        r = np.asarray(pred) - np.asarray(obs_list[i]).reshape(3,)
        norms.append(np.linalg.norm(r))
    return np.array(norms)

def mahalanobis_norm(r, C):
    # r: (3,), C: (3,3)
    return float(r.T @ np.linalg.inv(C) @ r)

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

    pixel1_uv = values["pixel1_uv"]
    pixel2_uv = values["pixel2_uv"]
    pixel1_uv_cov = values["pixel1_uv_cov"]
    pixel2_uv_cov = values["pixel2_uv_cov"]

    return from_idx,frame_idx, obs_Tc1, obs_Tc2, obs1_covTc, obs2_covTc, pixel1_uv, pixel2_uv, pixel1_uv_cov, pixel2_uv_cov

def rerun_viz(from_idx, frame_idx,
              pose_1_ini, pose_2_ini,
              pose_1_opt, pose_2_opt,
              landmark_1, landmark_2,
              landmark_covar_1, landmark_covar_2,
              landmark_positions
              ):

    rr.set_time("step", sequence=0)
    pose_1_ini_rot = pose_1_ini.rotation().toQuaternion()
    rr.log(
        "logs",
        rr.TextLog(f"Pose 1 before optimization: {pose_1_ini}")
    )
    pose_2_ini_rot = pose_2_ini.rotation().toQuaternion()
    rr.log(
        "logs",
        rr.TextLog(f"Pose 2 before optimization: {pose_2_ini}")
    )

    landmark_w_1 = [pose_1_ini.transformFrom(np.array(landmark_1_i)) for landmark_1_i in landmark_1]
    landmark_w_2 = [pose_2_ini.transformTo(np.array(landmark_2_i)) for landmark_2_i in landmark_2]
    for i, (w_1, w_2) in enumerate(zip(landmark_w_1, landmark_w_2)):
        rr.log(f"world/arrows/{i}",
               rr.LineStrips3D(
                   [w_1, w_2],
                   colors=[0, 0, 255],
               ))

    pose_1_q = pose_1_opt.rotation().toQuaternion()
    pose_1_t = pose_1_opt.translation()

    pose_2_q = pose_2_opt.rotation().toQuaternion()
    pose_2_t = pose_2_opt.translation()

    # Log optimized poses
    rr.set_time("step", sequence=1)
    rr.log(
        "logs",
        rr.TextLog(f"Pose 1 after optimization: {pose_1_opt}")
    )

    rr.log(
        "logs",
        rr.TextLog(f"Pose 2 after optimization: {pose_2_opt}")
    )

    # camera
    rr.set_time("step", sequence=0)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log("world/cam/{}".format(from_idx),
           rr.Transform3D(
               translation=pose_1_ini.translation(),
               quaternion=[pose_1_ini_rot.x(), pose_1_ini_rot.y(), pose_1_ini_rot.z(), pose_1_ini_rot.w()],
               axis_length=1.0,
           ))

    rr.log("world/cam/{}".format(frame_idx),
           rr.Transform3D(
               translation=pose_2_ini.translation(),
               quaternion=[pose_2_ini_rot.x(), pose_2_ini_rot.y(), pose_2_ini_rot.z(), pose_2_ini_rot.w()],
               axis_length=1.0
           ))
    
    landmark_covar_1 = [np.diag(np.array(landmark_covar_i))/2 for landmark_covar_i in landmark_covar_1]
    landmark_covar_2 = [np.diag(np.array(landmark_covar_i))/2 for landmark_covar_i in landmark_covar_2]

    rr.log(
        "world/cam/{}/points_w_covar".format(from_idx),
        rr.Ellipsoids3D(
            centers=landmark_1,
            half_sizes=landmark_covar_1,
            colors=[255, 165, 0],
        )
    )

    rr.log(
        "world/cam/{}/points_w_covar".format(frame_idx),
        rr.Ellipsoids3D(
            centers=landmark_2,
            half_sizes=landmark_covar_2,
            colors=[245, 66, 191],
        )
    )

    rr.set_time("step", sequence=1)
    rr.log("world/cam/{}".format(frame_idx),
           rr.Transform3D(
               translation=pose_2_t,
               quaternion=[pose_2_q.x(), pose_2_q.y(), pose_2_q.z(), pose_2_q.w()],
               axis_length=1.0
           ))


    landmark_w_1 = [pose_1_opt.transformFrom(np.array(landmark_1_i)) for landmark_1_i in landmark_1]
    landmark_w_2 = [pose_2_opt.transformFrom(np.array(landmark_2_i)) for landmark_2_i in landmark_2]
    for i, (w_1, w_2) in enumerate(zip(landmark_w_1, landmark_w_2)):
        rr.log(f"world/arrows/{i}",
               rr.LineStrips3D(
                   [w_1, w_2],
                   colors=[0, 0, 255],
               ))

    rr.log(
        "world/optimized/points",
        rr.Points3D(landmark_positions, colors=[0, 255, 0], radii=[.02])
    )
    rr.log("logs", rr.TextLog("Optimization complete."))