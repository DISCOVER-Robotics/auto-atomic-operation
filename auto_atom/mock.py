"""Mock runtime components used for development and examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .framework import (
    AutoAtomConfig,
    EefControlConfig,
    PoseControlConfig,
    TaskFileConfig,
)
from .runtime import (
    ControlResult,
    ControlSignal,
    ObjectHandler,
    OperatorHandler,
    SimulatorBackend,
)


@dataclass
class MockObjectHandler(ObjectHandler):
    """Mock object handle."""

    kind: str = "mock_object"


@dataclass
class MockOperatorHandler(OperatorHandler):
    """Mock operator that completes each primitive action in two update ticks."""

    operator_name: str
    role: str = "generic"
    _command_key: str = ""
    _progress: int = 0

    @property
    def name(self) -> str:
        return self.operator_name

    def move_to_pose(
        self,
        pose: PoseControlConfig,
        simulator: SimulatorBackend,
        target: Optional[ObjectHandler],
    ) -> ControlResult:
        command_key = f"pose:{_serialize_param(pose)}:{target.name if target else ''}"
        self._prepare_command(command_key)
        self._progress += 1
        if self._progress == 1:
            return ControlResult(
                signal=ControlSignal.RUNNING,
                details={
                    "event": "moving",
                    "operator": self.name,
                    "role": self.role,
                    "target": target.name if target else "",
                    "pose": _serialize_param(pose),
                },
            )
        return ControlResult(
            signal=ControlSignal.REACHED,
            details={
                "event": "pose_reached",
                "operator": self.name,
                "target": target.name if target else "",
                "pose": _serialize_param(pose),
            },
        )

    def control_eef(
        self,
        eef: EefControlConfig,
        simulator: SimulatorBackend,
    ) -> ControlResult:
        command_key = f"eef:{eef.close}:{eef.joint_positions}"
        self._prepare_command(command_key)
        self._progress += 1
        if self._progress == 1:
            return ControlResult(
                signal=ControlSignal.RUNNING,
                details={
                    "event": "eef_moving",
                    "operator": self.name,
                    "eef": _serialize_param(eef),
                },
            )
        return ControlResult(
            signal=ControlSignal.REACHED,
            details={
                "event": "eef_reached",
                "operator": self.name,
                "eef": _serialize_param(eef),
            },
        )

    def _prepare_command(self, command_key: str) -> None:
        if self._command_key != command_key:
            self._command_key = command_key
            self._progress = 0


@dataclass
class MockSimulatorBackend(SimulatorBackend):
    """Simple simulator backend with in-memory object and operator handlers."""

    simulator_name: str
    env_path: str
    operators: Dict[str, MockOperatorHandler] = field(default_factory=dict)
    objects: Dict[str, MockObjectHandler] = field(default_factory=dict)
    lifecycle_events: List[str] = field(default_factory=list)

    def setup(self, config: AutoAtomConfig) -> None:
        self.lifecycle_events.append(
            f"setup(simulator={self.simulator_name}, env={config.env_path}, seed={config.seed})"
        )

    def reset(self) -> None:
        self.lifecycle_events.append("reset()")
        for operator in self.operators.values():
            operator._progress = 0
            operator._command_key = ""

    def teardown(self) -> None:
        self.lifecycle_events.append("teardown()")

    def get_operator_handler(self, name: str) -> MockOperatorHandler:
        try:
            return self.operators[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.operators)) or "<empty>"
            raise KeyError(f"Unknown operator '{name}'. Known operators: {known}") from exc

    def get_object_handler(self, name: str) -> Optional[MockObjectHandler]:
        if not name:
            return None
        try:
            return self.objects[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.objects)) or "<empty>"
            raise KeyError(f"Unknown object '{name}'. Known objects: {known}") from exc


def build_mock_backend(task_file: TaskFileConfig) -> MockSimulatorBackend:
    config = task_file.task
    operators = {
        operator.name: MockOperatorHandler(
            operator_name=operator.name,
            role=operator.model_extra.get("role", "generic") if operator.model_extra else "generic",
        )
        for operator in task_file.operators
    }
    object_names = {stage.object for stage in config.stages if stage.object}
    objects = {
        object_name: MockObjectHandler(name=object_name)
        for object_name in object_names
    }
    return MockSimulatorBackend(
        simulator_name=config.simulator,
        env_path=str(config.env_path),
        operators=operators,
        objects=objects,
    )


def _serialize_param(param: Any) -> Any:
    if hasattr(param, "model_dump"):
        return param.model_dump(mode="json")
    return param
