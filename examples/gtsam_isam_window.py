import gtsam
import numpy as np
import rerun as rr
from utils.example_utils import parse_graph_file, rerun_viz
from typing import List, Optional
from gtsam_pose_to_point import make_pose_to_point_factor
from collections import defaultdict
from typing import Dict, Tuple, List
import sys
sys.stdout.flush()

PosePixelMap = Dict[int, Dict[Tuple[float, float], int]]

def next_landmark_index(values: gtsam.Values) -> int:
    max_idx = -1
    for k in values.keys():
        s = gtsam.Symbol(k)
        if s.chr() == ord('l'):
            max_idx = max(max_idx, s.index())
    return max_idx + 1


def connect_pixels(prev_pixel2_uv, current_pixel1_uv):
    indexes = []
    for i in range(len(current_pixel1_uv)):
        uv1 = current_pixel1_uv[i]
        mask = np.isclose(prev_pixel2_uv, uv1, atol=1.).all(axis=-1)
        if mask.any():
            idx = np.where(mask)[0][0]
            indexes.append((idx, i))
    return indexes


def test_gtsam_pose_to_point_isam2():
    rr.init("gtsam_pose2point_isam2", spawn=True)

    # ---- iSAM2 setup ----
    isam_params = gtsam.ISAM2Params()
    isam_params.setRelinearizeThreshold(0.01)
    isam_params.relinearizeSkip = 1
    isam = gtsam.ISAM2(isam_params)

    # Use ONE gauge-fixing prior (recommended for SLAM / iSAM2).
    # If you *really* want to clamp pose_0 & pose_1 each iteration (like your LM version),
    # move the priors inside the loop.
    gauge_prior_added = False

    # Robust kernel
    m_huber = gtsam.noiseModel.mEstimator.Huber.Create(0.1)

    pose_pixel_landmarks: PosePixelMap = defaultdict(dict)

    for index in range(1, 28):
        prev_from_idx, prev_frame_idx, \
            prev_from_pose, prev_init_motion, \
            prev_obs_landmark_c1, prev_obs_landmark_c2, \
            prev_obs_cov_landmark_c1, prev_obs_cov_landmark_c2, \
            prev_pixel1_uv, prev_pixel2_uv, prev_pixel1_cov_uv, prev_pixel2_cov_uv = \
            parse_graph_file("graph_data_dump_long.json", frame_idx=index)

        current_from_idx, current_frame_idx, \
            current_from_pose, current_init_motion, \
            current_obs_landmark_c1, current_obs_landmark_c2, \
            current_obs_cov_landmark_c1, current_obs_cov_landmark_c2, \
            current_pixel1_uv, current_pixel2_uv, current_pixel1_cov_uv, current_pixel2_cov_uv = \
            parse_graph_file("graph_data_dump_long.json", frame_idx=index + 1)

        connected_indexes = connect_pixels(prev_pixel2_uv, current_pixel1_uv)

        # Keys
        pose_0_key = gtsam.symbol('p', prev_from_idx)
        pose_1_key = gtsam.symbol('p', current_from_idx)
        pose_2_key = gtsam.symbol('p', current_frame_idx)

        P0 = prev_from_pose
        P1 = current_from_pose
        P2 = current_init_motion

        # Only add *new* factors and *new* initial guesses each step
        new_factors = gtsam.NonlinearFactorGraph()
        new_values = gtsam.Values()

        # Current estimate (for "exists" checks)
        est = isam.calculateEstimate() #if isam.size() > 0 else gtsam.Values()

        # Insert pose initials if missing
        if not est.exists(pose_0_key):
            new_values.insert(pose_0_key, P0)
        if not est.exists(pose_1_key):
            new_values.insert(pose_1_key, P1)
        if not est.exists(pose_2_key):
            new_values.insert(pose_2_key, P2)

        # Gauge fix: add ONE strong prior (or two if you insist).
        if not gauge_prior_added:
            sigmas = np.array([1e-4] * 6, dtype=np.float64)
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
            new_factors.add(gtsam.PriorFactorPose3(pose_0_key, P0, prior_noise))
            # Optional: also pin pose_1 (your original did). Usually not necessary.
            new_factors.add(gtsam.PriorFactorPose3(pose_1_key, P1, prior_noise))
            gauge_prior_added = True

        # Landmarks
        landmark_keys = []
        cov_landmark_1 = []
        cov_landmark_2 = []

        covar_scale = 1.0
        eps = 1e-6

        start_l_idx = next_landmark_index(isam.calculateEstimate())

        for i in range(len(current_obs_landmark_c1)):
            # Window: connect previous observation to same landmark if pixel matched
            if any(i == conn[1] for conn in connected_indexes):
                prev_i = [conn[0] for conn in connected_indexes if conn[1] == i][0]
                uv0 = (int(prev_pixel1_uv[prev_i][0]), int(prev_pixel1_uv[prev_i][1]))
                landmark_key = pose_pixel_landmarks.get(pose_0_key, {}).get(uv0, None)
                if landmark_key is None:
                    # This should not happen
                    continue

                obs_c0 = prev_obs_landmark_c1[prev_i]
                cov_c0 = np.array(prev_obs_cov_landmark_c1[prev_i], dtype=np.float64) * covar_scale
                cov_c0 = cov_c0 + eps * np.eye(3)
                nm0 = gtsam.noiseModel.Robust.Create(m_huber, gtsam.noiseModel.Gaussian.Covariance(cov_c0))
                new_factors.add(make_pose_to_point_factor(pose_0_key, landmark_key, obs_c0, nm0))

            else:
                landmark_key = gtsam.symbol('l', start_l_idx + i)
                landmark_keys.append(landmark_key)

            uv1 = (int(current_pixel1_uv[i][0]), int(current_pixel1_uv[i][1]))
            uv2 = (int(current_pixel2_uv[i][0]), int(current_pixel2_uv[i][1]))
            pose_pixel_landmarks.setdefault(pose_1_key, {}).setdefault(uv1, landmark_key)
            pose_pixel_landmarks.setdefault(pose_2_key, {}).setdefault(uv2, landmark_key)

            obs_c1 = current_obs_landmark_c1[i]
            obs_c2 = current_obs_landmark_c2[i]

            cov_c1 = np.array(current_obs_cov_landmark_c1[i], dtype=np.float64) * covar_scale
            cov_c2 = np.array(current_obs_cov_landmark_c2[i], dtype=np.float64) * covar_scale
            cov_c1 = cov_c1 + eps * np.eye(3)
            cov_c2 = cov_c2 + eps * np.eye(3)

            cov_landmark_1.append(cov_c1)
            cov_landmark_2.append(cov_c2)

            nm1 = gtsam.noiseModel.Robust.Create(m_huber, gtsam.noiseModel.Gaussian.Covariance(cov_c1))
            nm2 = gtsam.noiseModel.Robust.Create(m_huber, gtsam.noiseModel.Gaussian.Covariance(cov_c2))

            # Factors for current frame pair
            new_factors.add(make_pose_to_point_factor(pose_1_key, landmark_key, obs_c1, nm1))
            new_factors.add(make_pose_to_point_factor(pose_2_key, landmark_key, obs_c2, nm2))

            # Initial for landmark if missing: use cam1 observation lifted into world with P1
            if not est.exists(landmark_key) and not new_values.exists(landmark_key):
                pw_init = P1.transformFrom(np.asarray(obs_c1, dtype=np.float64).reshape(3,))
                new_values.insert(landmark_key, pw_init)

        # ---- iSAM2 update ----
        isam.update(new_factors, new_values)
        result = isam.calculateEstimate()

        # Visualization values (keep parity with your original output)
        pose_1_ini = P1
        pose_2_ini = P2
        pose_1_opt = result.atPose3(pose_1_key)
        pose_2_opt = result.atPose3(pose_2_key)
        landmark_positions = [result.atPoint3(k) for k in landmark_keys]

        rerun_viz(
            current_from_idx, current_frame_idx,
            pose_1_ini, pose_2_ini,
            pose_1_opt, pose_2_opt,
            current_obs_landmark_c1, current_obs_landmark_c2,
            cov_landmark_1, cov_landmark_2,
            landmark_positions
        )


if __name__ == "__main__":
    test_gtsam_pose_to_point_isam2()
