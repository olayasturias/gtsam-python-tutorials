import gtsam
import numpy as np
import rerun as rr
from typing import List, Optional
from utils.example_utils import parse_graph_file, residuals_for_pose, mahalanobis_norm, rerun_viz
from gtsam import SmartProjectionPose3Factor, SmartProjectionParams, Cal3_S2

def test_gtsam_smart_projection_pose():
    rr.init("gtsam_test_custom_factor_rel", spawn=True)

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
    # Give pose 2 a non-degenerate initial guess
    P2 = gtsam.Pose3(gtsam.Rot3(np.eye(3)*.9), gtsam.Point3(np.array([0.05, 0.05, 0.05])))
    initial_estimate.insert(pose_1_key, P1)
    initial_estimate.insert(pose_2_key, P2)

    # Prior: set pose 1 at origin (identity).
    # Hard constraints can make optimization brittle when other factors are initially inconsistent. Add sigmas.
    sigmas = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])  # rad, rad, rad, m, m, m
    ini_estimate_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    graph.add(
        gtsam.PriorFactorPose3(
            pose_1_key,
            P1,  # fixed at identity
            ini_estimate_noise
        ))

    ##### Step 2: Set factors for landmarks

    landmark_keys = []
    for i in range(len(obs_landmark_c1)):
        # Initialize the smart factor
        smart_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        smart_params = SmartProjectionParams()  # Use default params
        K = Cal3_S2(459.2732, 459.2732, 0, 345.8487, 349.7954)  # Fixed calibration
        # tensor([[459.2732,   0.0000,   0.0000],
        #         [  0.0000, 459.2732,   0.0000],
        #         [345.8487, 349.7954,   1.0000]])

        smart_factor = SmartProjectionPose3Factor(smart_noise, K, smart_params)
        smart_factor.add(gtsam.Point2(pixel1_uv[i][0], pixel1_uv[i][1]), pose_1_key)
        smart_factor.add(gtsam.Point2(pixel2_uv[i][0], pixel2_uv[i][1]), pose_2_key)

        # Add factors to graph
        graph.add(smart_factor)


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

    landmark_positions = [result.atPoint3(landmark_keys[i]) for i in range(len(landmark_keys))]

    rerun_viz(from_idx, frame_idx,
              pose_1_ini, pose_2_ini,
              pose_1_opt, pose_2_opt,
              obs_landmark_c1, obs_landmark_c2,
              obs_cov_landmark_c1, obs_cov_landmark_c2,
              landmark_positions
              )



if __name__ == "__main__":
    test_gtsam_smart_projection_pose()