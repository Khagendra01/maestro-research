"""Hardware boundary.

Simulation/replay is the default. Any path that could connect to a robot must
pass through this module. Physical execution requires both an explicit flag in
code and the environment variable MAESTRO_ALLOW_PHYSICAL=1.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

PHYSICAL_ENV_VAR = "MAESTRO_ALLOW_PHYSICAL"
DEFAULT_MODE = "sim"


class PhysicalExecutionNotAuthorized(RuntimeError):
    pass


def physical_authorized() -> bool:
    return os.environ.get(PHYSICAL_ENV_VAR) == "1"


def require_physical_authorization(allow_physical: bool) -> None:
    if not allow_physical:
        raise PhysicalExecutionNotAuthorized(
            "physical execution is disabled: pass allow_physical=True and set "
            f"{PHYSICAL_ENV_VAR}=1 after explicit authorization"
        )
    if not physical_authorized():
        raise PhysicalExecutionNotAuthorized(
            f"physical execution is disabled: set {PHYSICAL_ENV_VAR}=1 after explicit authorization"
        )


def refuse_droid_import() -> None:
    """Fail fast if any code path imports the DROID hardware client in sim mode."""
    if "droid" in sys.modules and not physical_authorized():
        raise PhysicalExecutionNotAuthorized(
            "the droid hardware client was imported while physical execution is disabled"
        )


class SimulationOnlyGuard:
    """Context object that refuses hardware connection, homing, scanning and motion."""

    def __init__(self, backend_mode: str = DEFAULT_MODE):
        self.backend_mode = backend_mode

    def assert_simulation(self) -> None:
        if self.backend_mode != "sim":
            raise PhysicalExecutionNotAuthorized(
                f"expected a simulation backend, got mode={self.backend_mode!r}"
            )

    def connect(self, *_args, **_kwargs):
        raise PhysicalExecutionNotAuthorized("hardware connection is disabled in simulation mode")

    home = scan = move = close_gripper = open_gripper = connect


class HardwareBackend:
    """Placeholder hardware backend. Constructing it without authorization fails."""

    mode = "hardware"

    def __init__(self, allow_physical: bool = False):
        require_physical_authorization(allow_physical)
        from droid.robot_env import RobotEnv  # noqa: F401  (unreachable without authorization)

        raise NotImplementedError(
            "The hardware backend is not implemented in this milestone. Autonomous motion "
            "requires calibration verification: arm/gripper model, joint limits, TCP and "
            "camera transforms, controller limits, workspace bounds and stop behavior."
        )

    def close(self) -> None:
        pass


def guard_observation_source(source: str, expected: str = "sim") -> None:
    allowed = {"sim", "replay", "fixture"}
    if expected == "sim" and source not in allowed:
        raise PhysicalExecutionNotAuthorized(
            f"observation source {source!r} is not permitted for a simulation episode"
        )
