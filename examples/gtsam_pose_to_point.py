import gtsam
import numpy as np
import rerun as rr
from typing import List, Optional
from utils.example_utils import parse_graph_file


def make_pose_to_point_factor(camera_pose_key, landmark_key, obs_landmark_ck_i, noise_model):
    obs_landmark_ck_i = np.asarray(obs_landmark_ck_i, dtype=np.float64).reshape(3, )

    keys = [camera_pose_key, landmark_key]

    def error_func(this_factor, values, jacobian: Optional[List[np.ndarray]]):
        pose_ck: gtsam.Pose3 = values.atPose3(this_factor.keys()[0])
        landmark_w_k: gtsam.Point3 = values.atPoint3(this_factor.keys()[1])

        if jacobian is not None:
            jacobian[0] = np.zeros((3, 6), dtype=np.float64)
            jacobian[1] = np.zeros((3, 3), dtype=np.float64)

            pred_landmark_ck_i = pose_ck.transformTo(landmark_w_k, jacobian[0], jacobian[1])

        else:
            pred_landmark_ck_i = pose_ck.transformTo(landmark_w_k)  # (3,)

        r = pred_landmark_ck_i - obs_landmark_ck_i  # (3,)
        return r

    return gtsam.CustomFactor(noise_model, keys, error_func)


def test_gtsam_pose_to_point():
    rr.init("gtsam_test_custom_factor_rel", spawn=True)

    # Read and parse graph values
    from_idx,frame_idx, \
        obs_landmark_c1, obs_landmark_c2, \
        obs_cov_landmark_c1, obs_cov_landmark_c2 = \
        parse_graph_file("graph_data_dump.json")

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
    ini_estimate_noise = gtsam.noiseModel.Constrained.All(6)
    graph.add(
        gtsam.PriorFactorPose3(
            pose_1_key,
            P1,  # fixed at identity
            ini_estimate_noise
        ))

    ##### Step 2: Set factors for landmarks

    landmark_keys = []
    for i in range(len(obs_landmark_c1)):
        # Read observation and covariance
        obs_landmark_c1_i = obs_landmark_c1[i]
        obs_landmark_c2_i = obs_landmark_c2[i]
        obs_cov_landmark_c1_i = obs_cov_landmark_c1[i]
        obs_cov_landmark_c2_i = obs_cov_landmark_c2[i]

        # Create noise model
        noise_model_1 = gtsam.noiseModel.Gaussian.Covariance(obs_cov_landmark_c1_i)
        noise_model_2 = gtsam.noiseModel.Gaussian.Covariance(obs_cov_landmark_c2_i)
        m_huber = gtsam.noiseModel.mEstimator.Huber.Create(0.1)
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
        # Create factors
        factor1 = make_pose_to_point_factor(pose_1_key, landmark_key, obs_landmark_c1_i, noise_model_1)
        factor2 = make_pose_to_point_factor(pose_2_key, landmark_key, obs_landmark_c2_i, noise_model_2)
        # Add factors to graph
        graph.add(factor1)
        graph.add(factor2)
        # Add initial estimate for landmarks (in world coordinates)
        # We will just set the initial estimate as the points overlapping with the points seen from cam 1
        obs_landmark_w_i = P1.transformFrom(obs_landmark_c1_i)
        initial_estimate.insert(landmark_key, obs_landmark_w_i)

    ##### Step 3: optimization of the graph

    rr.set_time("step", sequence=0)

    rr.log("logs", rr.TextLog("Starting optimization..."))
    pose_1_ini = (initial_estimate.atPose3(pose_1_key) if initial_estimate.exists(pose_1_key) else "Not in graph")
    pose_1_ini_rot = pose_1_ini.rotation().toQuaternion()
    rr.log(
        "logs",
        rr.TextLog(f"Pose 1 before optimization: {pose_1_ini}")
    )
    pose_2_ini = (initial_estimate.atPose3(pose_2_key) if initial_estimate.exists(pose_2_key) else "Not in graph")
    pose_2_ini_rot = pose_2_ini.rotation().toQuaternion()
    rr.log(
        "logs",
        rr.TextLog(f"Pose 2 before optimization: {pose_2_ini}")
    )

    # Set the optimizer and verbosity level
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    # Run optimization
    result = optimizer.optimize()

    # Read result from graph using the keys we've assigned to the variables
    pose_1 = result.atPose3(pose_1_key)
    pose_1_q = pose_1.rotation().toQuaternion()
    pose_1_t = pose_1.translation()
    pose_2 = result.atPose3(pose_2_key)
    pose_2_q = pose_2.rotation().toQuaternion()
    pose_2_t = pose_2.translation()
    landmark_positions = [result.atPoint3(landmark_keys[i]) for i in range(len(landmark_keys))]

    # Log optimized poses
    rr.set_time("step", sequence=1)
    rr.log(
        "logs",
        rr.TextLog(f"Pose 1 after optimization: {pose_1}")
    )

    rr.log(
        "logs",
        rr.TextLog(f"Pose 2 after optimization: {pose_2}")
    )

    ############### RERUN VISUALIZATION ###############
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
    rr.log(
        "world/cam/{}/points".format(from_idx),
        rr.Points3D(obs_landmark_c1, colors=[255, 165, 0], radii=[.02])
    )
    rr.log(
        "world/cam/{}/points".format(frame_idx),
        rr.Points3D(obs_landmark_c2, colors=[245, 66, 191], radii=[.02])
    )

    rr.set_time("step", sequence=1)
    rr.log("world/cam/{}".format(frame_idx),
           rr.Transform3D(
               translation=pose_2_t,
               quaternion=[pose_2_q.x(), pose_2_q.y(), pose_2_q.z(), pose_2_q.w()],
               axis_length=1.0
           ))

    rr.log(
        "world/optimized/points",
        rr.Points3D(landmark_positions, colors=[0, 255, 0], radii=[.02])
    )
    rr.log("logs", rr.TextLog("Optimization complete."))


if __name__ == "__main__":
    test_gtsam_pose_to_point()