import gtsam
import numpy as np
import rerun as rr
from typing import List, Optional
from utils.example_utils import parse_graph_file, rerun_viz
from gtsam import Cal3_S2
from gtsam_pose_to_point import make_pose_to_point_factor

def test_gtsam_generic_projection_with_3d_priors():
    rr.init("gtsam_reprojection_pose2point", spawn=True)

    from_idx, frame_idx, \
        obs_landmark_c1, obs_landmark_c2, \
        obs_cov_landmark_c1, obs_cov_landmark_c2, \
        pixel1_uv, pixel2_uv, pixel1_cov_uv, pixel2_cov_uv = \
        parse_graph_file("graph_data_dump_curved_plane.json")

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # World frame = camera 1 frame
    pose_1_key = gtsam.symbol('p', 1)
    pose_2_key = gtsam.symbol('p', 2)

    P1 = gtsam.Pose3.Identity()
    P2 = gtsam.Pose3(gtsam.Rot3(np.eye(3)*0.9),
                     gtsam.Point3(np.array([0.05, 0.05, 0.05], dtype=np.float64)))

    initial.insert(pose_1_key, P1)
    initial.insert(pose_2_key, P2)

    # Strong (not hard) prior on pose 1 to fix gauge
    sigmas = np.array([1e-4]*6, dtype=np.float64)
    graph.add(gtsam.PriorFactorPose3(
        pose_1_key, P1, gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    ))

    # Calibration
    K = Cal3_S2(459.2732, 459.2732, 0, 345.8487, 349.7954)

    landmark_keys = []

    for i in range(len(pixel1_uv)):
        lm_key = gtsam.symbol('l', i)
        landmark_keys.append(lm_key)

        # Model pixel noise from covariances
        pixel_covar_1 = np.diag(pixel1_cov_uv[i][:2])
        pixel_covar_2 = np.diag(pixel2_cov_uv[i][:2])
        pixel_noise_1 = gtsam.noiseModel.Gaussian.Covariance(np.array(pixel_covar_1, dtype=np.float64))
        pixel_noise_2 = gtsam.noiseModel.Gaussian.Covariance(np.array(pixel_covar_2, dtype=np.float64))
        pix_huber = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
        pixel_noise_1 = gtsam.noiseModel.Robust.Create(pix_huber, pixel_noise_1)
        pixel_noise_2 = gtsam.noiseModel.Robust.Create(pix_huber, pixel_noise_2)

        # Model 3D landmark noise from covariances
        lm_covar_1 = obs_cov_landmark_c1[i]
        lm3d_noise_1 = gtsam.noiseModel.Gaussian.Covariance(np.array(lm_covar_1, dtype=np.float64))
        l_huber = gtsam.noiseModel.mEstimator.Huber.Create(.1)
        lm3d_noise_1 = gtsam.noiseModel.Robust.Create(l_huber, lm3d_noise_1)
        lm_covar_2 = obs_cov_landmark_c2[i]
        lm3d_noise_2 = gtsam.noiseModel.Gaussian.Covariance(np.array(lm_covar_2, dtype=np.float64))
        lm3d_noise_2 = gtsam.noiseModel.Robust.Create(l_huber, lm3d_noise_2)

        # Add projection factors (2D) for both cameras
        z1 = gtsam.Point2(float(pixel1_uv[i][0]), float(pixel1_uv[i][1]))
        z2 = gtsam.Point2(float(pixel2_uv[i][0]), float(pixel2_uv[i][1]))

        graph.add(gtsam.GenericProjectionFactorCal3_S2(z1, pixel_noise_1, pose_1_key, lm_key, K))
        graph.add(gtsam.GenericProjectionFactorCal3_S2(z2, pixel_noise_2, pose_2_key, lm_key, K))

        # Add 3D pose-to-point constraint for camera 2 using obs_landmark_c2:
        # transformTo maps world point into camera-2 coordinates.
        p2 = np.asarray(obs_landmark_c2[i], dtype=np.float64).reshape(3,)
        graph.add(make_pose_to_point_factor(pose_2_key, lm_key, p2, lm3d_noise_1))

        # Optional: enforce cam1 3D consistency
        p1 = np.asarray(obs_landmark_c1[i], dtype=np.float64).reshape(3,)
        graph.add(make_pose_to_point_factor(pose_1_key, lm_key, p1, lm3d_noise_2))

        # Initialize landmark in WORLD (world == cam1) using obs_landmark_c1
        obs_landmark_w_i = P1.transformFrom(obs_landmark_c1[i])
        initial.insert(lm_key, obs_landmark_w_i)

    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()

    pose_1_ini = initial.atPose3(pose_1_key)
    pose_2_ini = initial.atPose3(pose_2_key)

    pose_1_opt = result.atPose3(pose_1_key)
    pose_2_opt = result.atPose3(pose_2_key)
    landmark_positions = [result.atPoint3(k) for k in landmark_keys]

    rerun_viz(from_idx, frame_idx,
              pose_1_ini, pose_2_ini,
              pose_1_opt, pose_2_opt,
              obs_landmark_c1, obs_landmark_c2,
              obs_cov_landmark_c1, obs_cov_landmark_c2,
              landmark_positions)

if __name__ == "__main__":
    test_gtsam_generic_projection_with_3d_priors()
