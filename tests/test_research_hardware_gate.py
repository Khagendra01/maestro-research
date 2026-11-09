import os
import unittest
from unittest.mock import patch

from maestro_research.hardware_gate import (
    PHYSICAL_ENV_VAR,
    HardwareBackend,
    PhysicalExecutionNotAuthorized,
    SimulationOnlyGuard,
    guard_observation_source,
    require_physical_authorization,
)


class HardwareGateTests(unittest.TestCase):
    def test_physical_execution_requires_both_flag_and_env(self):
        with patch.dict(os.environ, {PHYSICAL_ENV_VAR: "0"}):
            with self.assertRaises(PhysicalExecutionNotAuthorized):
                require_physical_authorization(True)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(PhysicalExecutionNotAuthorized):
                require_physical_authorization(False)

    def test_hardware_backend_cannot_be_constructed_in_sim(self):
        with patch.dict(os.environ, {PHYSICAL_ENV_VAR: "0"}):
            with self.assertRaises(PhysicalExecutionNotAuthorized):
                HardwareBackend(allow_physical=False)
            with self.assertRaises(PhysicalExecutionNotAuthorized):
                HardwareBackend(allow_physical=True)

    def test_simulation_guard_refuses_motion(self):
        guard = SimulationOnlyGuard("sim")
        guard.assert_simulation()
        for name in ("connect", "home", "scan", "move", "close_gripper", "open_gripper"):
            with self.subTest(name=name), self.assertRaises(PhysicalExecutionNotAuthorized):
                getattr(guard, name)()

    def test_non_sim_observation_source_is_rejected(self):
        guard_observation_source("sim")
        with self.assertRaises(PhysicalExecutionNotAuthorized):
            guard_observation_source("hardware")


if __name__ == "__main__":
    unittest.main()
