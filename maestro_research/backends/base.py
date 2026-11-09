"""Backend protocol. A backend owns the physics/action interface and the official
success oracle; the executor never reads the oracle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from ..contracts import Observation


@dataclass(frozen=True)
class BackendInfo:
    name: str
    mode: str
    robot: str
    gripper: str
    control_hz: float
    action_dim: int
    action_semantics: str
    notes: str = ""


class SimulationBackend(Protocol):
    info: BackendInfo
    mode: str

    def reset(self, task, initial_state_index: int) -> Observation:
        ...

    def step(self, action) -> "StepResult":
        ...

    def observe(self) -> Observation:
        ...

    def robot_state(self):
        ...

    def official_success(self) -> bool:
        ...

    def close(self) -> None:
        ...


@dataclass
class StepResult:
    observation: Optional[Observation]
    done: bool
    info: Dict[str, Any]
