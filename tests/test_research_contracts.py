import unittest

import numpy as np

from maestro_research.contracts import (
    GraspCandidate,
    Observation,
    RobotState,
    SceneSnapshot,
    ObjectEstimate,
    matrix_to_quat_xyzw,
    pose_from_pos_quat,
    quat_xyzw_to_matrix,
    transform_points,
    validate_pose,
)


class TransformTests(unittest.TestCase):
    def test_quaternion_matrix_roundtrip(self):
        quats = [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.7071067811865476, 0.0, 0.7071067811865476],
            [0.2, -0.3, 0.4, 0.85],
        ]
        for q in quats:
            matrix = quat_xyzw_to_matrix(q)
            back = matrix_to_quat_xyzw(matrix)
            np.testing.assert_allclose(quat_xyzw_to_matrix(back), matrix, atol=1e-8)

    def test_transform_points_and_inverse_pose(self):
        pose = pose_from_pos_quat([0.3, -0.2, 0.8], [0.0, 0.0, 0.3826834, 0.9238795])
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        world = transform_points(pose, points)
        np.testing.assert_allclose(world[0], [0.3, -0.2, 0.8], atol=1e-8)
        self.assertAlmostEqual(np.linalg.norm(world[1] - world[0]), 1.0, places=8)

    def test_reflection_and_non_finite_pose_rejected(self):
        bad = np.eye(4)
        bad[0, 0] = -1.0
        with self.assertRaises(ValueError):
            validate_pose(bad)
        nan = np.eye(4)
        nan[0, 3] = np.nan
        with self.assertRaises(ValueError):
            validate_pose(nan)


class ContractValidationTests(unittest.TestCase):
    def make_state(self):
        return RobotState(
            eef_position=[0.1, 0.2, 0.9],
            eef_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            gripper_qpos=[0.04, 0.04],
        )

    def test_observation_requires_matching_depth_shape(self):
        with self.assertRaises(ValueError):
            Observation(
                rgb=np.zeros((8, 8, 3), np.uint8),
                depth=np.zeros((4, 4)),
                intrinsics=np.eye(3),
                camera_pose=np.eye(4),
                robot_state=self.make_state(),
                calibration_id="test",
                frame_id="f0",
                acquisition_time=0.0,
            )

    def test_scene_staleness(self):
        obj = ObjectEstimate("cube", [0.1, 0.2, 0.05], [0, 0, 0, 1], None, "oracle", 0.0)
        scene = SceneSnapshot(1, 0.0, "world", {"cube": obj}, observation_age=0.2)
        self.assertFalse(scene.stale(0.5))
        self.assertTrue(scene.stale(0.1))

    def test_grasp_candidate_normalizes_axis_and_rejects_bad_width(self):
        pose = np.eye(4)
        candidate = GraspCandidate("c0", "cube", pose, pose, [0, 0, -2.0], 0.1, 0.04, 1.0, 0.0)
        np.testing.assert_allclose(candidate.approach_axis, [0, 0, -1], atol=1e-9)
        with self.assertRaises(ValueError):
            GraspCandidate("c0", "cube", pose, pose, [0, 0, -1], 0.1, 0.0, 1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
