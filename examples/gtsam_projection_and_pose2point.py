import gtsam
import numpy as np
import rerun as rr
from gtsam import Cal3_S2
from gtsam_pose_to_point import make_pose_to_point_factor
from utils.example_utils import parse_graph_file, rerun_viz


def test_gtsam_pose_to_point():
    rr.init("gtsam_reprojection_pose2point", spawn=True)

    # Read and parse graph values
    from_idx,frame_idx, \
        obs_landmark_c1, obs_landmark_c2, \
        obs_cov_landmark_c1, obs_cov_landmark_c2,\
        pixel1_uv, pixel2_uv, pixel1_cov_uv, pixel2_cov_uv = \
        parse_graph_file("graph_data_dump_curved_plane.json")

    ################## GTSAM POSE GRAPH SETUP ####################
    # Create  a factor graph container
    graph = gtsam.NonlinearFactorGraph()

    ##### Step 1: Set factors for poses

    # Create keys por pose 1 and 2 (e.g. p1 and p2)
    pose_1_key = gtsam.symbol('p', 1)
    pose_2_key = gtsam.symbol('p', 2)

    # Initial estimate: pose 1 at identity. Pose 2 overlaps pose 1
    initial_estimate = gtsam.Values()
    P1 = gtsam.Pose3.Identity()
    P2 = gtsam.Pose3.Identity()
    initial_estimate.insert(pose_1_key, P1)
    initial_estimate.insert(pose_2_key, P2)

    # Prior: set pose 1 at origin (identity).
    # Fix pose 1 there with a constrained noise model (i.e. no noise in the estimate)
    # Strong (not hard) prior on pose 1 to fix gauge
    sigmas = np.array([1e-4]*6, dtype=np.float64)
    graph.add(gtsam.PriorFactorPose3(
        pose_1_key, P1, gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    ))

    ##### Step 2: Set factors for landmarks

    landmark_keys = []
    cov_landmark_1 = []
    cov_landmark_2 = []
    # Calibration
    K = Cal3_S2(459.2732, 459.2732, 0, 345.8487, 349.7954)
    for i in range(len(obs_landmark_c1)):
        # Read observation and covariance
        obs_landmark_c1_i = obs_landmark_c1[i]
        obs_landmark_c2_i = obs_landmark_c2[i]
        covar_scale = 1.
        obs_cov_landmark_c1_i = np.array(obs_cov_landmark_c1[i])*covar_scale
        obs_cov_landmark_c2_i = np.array(obs_cov_landmark_c2[i])*covar_scale

        eps = 1e-6
        obs_cov_landmark_c1_i = obs_cov_landmark_c1_i + eps * np.eye(3)
        obs_cov_landmark_c2_i = obs_cov_landmark_c2_i + eps * np.eye(3)

        # obs_cov_landmark_c1_i = np.eye(3)*0.1
        # obs_cov_landmark_c2_i = np.eye(3)*0.1

        cov_landmark_1.append(obs_cov_landmark_c1_i)
        cov_landmark_2.append(obs_cov_landmark_c2_i)

        # Create noise model
        noise_model_1 = gtsam.noiseModel.Gaussian.Covariance(obs_cov_landmark_c1_i)
        noise_model_2 = gtsam.noiseModel.Gaussian.Covariance(obs_cov_landmark_c2_i)
        noise_model_1 = gtsam.noiseModel.Isotropic.Sigma(3, .1)
        noise_model_2 = gtsam.noiseModel.Isotropic.Sigma(3, .1)

        m_huber = gtsam.noiseModel.mEstimator.Huber.Create(.1)
        noise_model_1 = gtsam.noiseModel.Robust.Create(
            m_huber,
            noise_model_1
        )
        noise_model_2 = gtsam.noiseModel.Robust.Create(
            m_huber,
            noise_model_2
        )
        # Create landmark key
        landmark_key = gtsam.symbol('l', i)
        landmark_keys.append(landmark_key)

        ####### Create pose2point factors
        factor1 = make_pose_to_point_factor(pose_1_key, landmark_key, obs_landmark_c1_i, noise_model_1)
        factor2 = make_pose_to_point_factor(pose_2_key, landmark_key, obs_landmark_c2_i, noise_model_2)
        # Add factors to graph
        graph.add(factor1)
        graph.add(factor2)
        # Add initial estimate for landmarks (in world coordinates)
        # We will just set the initial estimate as the points overlapping with the points seen from cam 1
        obs_landmark_w_i = P1.transformFrom(obs_landmark_c1_i)
        initial_estimate.insert(landmark_key, obs_landmark_w_i)

        ####### Add projection factors (2D) for both cameras
        # Model pixel noise from covariances
        pixel_covar_1 = np.diag(pixel1_cov_uv[i][:2])
        pixel_covar_2 = np.diag(pixel2_cov_uv[i][:2])
        pixel_noise_1 = gtsam.noiseModel.Gaussian.Covariance(np.array(pixel_covar_1, dtype=np.float64))
        pixel_noise_2 = gtsam.noiseModel.Gaussian.Covariance(np.array(pixel_covar_2, dtype=np.float64))
        pix_huber = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
        pixel_noise_1 = gtsam.noiseModel.Robust.Create(pix_huber, pixel_noise_1)
        pixel_noise_2 = gtsam.noiseModel.Robust.Create(pix_huber, pixel_noise_2)

        z1 = gtsam.Point2(float(pixel1_uv[i][0]), float(pixel1_uv[i][1]))
        z2 = gtsam.Point2(float(pixel2_uv[i][0]), float(pixel2_uv[i][1]))

        graph.add(gtsam.GenericProjectionFactorCal3_S2(z1, pixel_noise_1, pose_1_key, landmark_key, K))
        graph.add(gtsam.GenericProjectionFactorCal3_S2(z2, pixel_noise_2, pose_2_key, landmark_key, K))

    ##### Step 3: optimization of the graph

    pose_1_ini = (initial_estimate.atPose3(pose_1_key) if initial_estimate.exists(pose_1_key) else "Not in graph")
    pose_2_ini = (initial_estimate.atPose3(pose_2_key) if initial_estimate.exists(pose_2_key) else "Not in graph")

    # Set the optimizer and verbosity level
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    # Run optimization
    result = optimizer.optimize()

    # Read result from graph using the keys we've assigned to the variables
    pose_1_opt = result.atPose3(pose_1_key)
    pose_2_opt = result.atPose3(pose_2_key)

    landmark_positions = [result.atPoint3(k) for k in landmark_keys]

    rerun_viz(from_idx, frame_idx,
              pose_1_ini, pose_2_ini,
              pose_1_opt, pose_2_opt,
              obs_landmark_c1, obs_landmark_c2,
              cov_landmark_1, cov_landmark_2,
              landmark_positions
              )



if __name__ == "__main__":
    test_gtsam_pose_to_point()