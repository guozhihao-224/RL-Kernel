# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Executable production/RL-Kernel routes for the module debug matrix."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from rl_engine.contracts.diagnostics.modules import module_debug_matrix
from rl_engine.runtime.performance.schema import ExecutionPhase, OperatorWorkloadKey
from rl_engine.runtime.semantic_registry import (
    OperatorInstanceProvenance,
    OperatorInstantiationError,
    OperatorRequirements,
    _JsonRecord,
)

if TYPE_CHECKING:
    from rl_engine.runtime.semantic_registry import OperatorSession
    from rl_engine.runtime.selector.cache import CachedSelection


class Implementation(str, Enum):
    PRODUCTION = "production"
    RL_KERNEL = "rl_kernel"


@dataclass(frozen=True)
class OperatorAblationCase:
    module: str
    case_id: str
    training: Implementation
    rollout: Implementation
    purpose: str
    diagnostic_axes: tuple[str, ...]

    def implementation_for(self, target: str) -> Implementation:
        normalized = target.strip().lower()
        if normalized == "training":
            return self.training
        if normalized == "rollout":
            return self.rollout
        raise ValueError("target must be 'training' or 'rollout'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "case_id": self.case_id,
            "training_implementation": self.training.value,
            "rollout_implementation": self.rollout.value,
            "purpose": self.purpose,
            "diagnostic_axes": list(self.diagnostic_axes),
        }


_CASE_DEFINITIONS = (
    ("P/P", Implementation.PRODUCTION, Implementation.PRODUCTION, "native baseline"),
    ("R/R", Implementation.RL_KERNEL, Implementation.RL_KERNEL, "RL-Kernel control"),
    (
        "P/R",
        Implementation.PRODUCTION,
        Implementation.RL_KERNEL,
        "rollout-only mismatch",
    ),
    (
        "R/P",
        Implementation.RL_KERNEL,
        Implementation.PRODUCTION,
        "training-only mismatch",
    ),
)


def operator_ablation_cases(module: str) -> tuple[OperatorAblationCase, ...]:
    normalized = module.strip().lower()
    matrix = module_debug_matrix()
    try:
        axes = tuple(str(axis["id"]) for axis in matrix["modules"][normalized]["axes"])
    except KeyError as exc:
        raise ValueError(f"unknown ablation module {module!r}") from exc
    return tuple(
        OperatorAblationCase(normalized, case_id, training, rollout, purpose, axes)
        for case_id, training, rollout, purpose in _CASE_DEFINITIONS
    )


def operator_ablation_case(module: str, case_id: str) -> OperatorAblationCase:
    normalized = case_id.strip().upper()
    for case in operator_ablation_cases(module):
        if case.case_id == normalized:
            return case
    raise ValueError(f"unknown ablation case {case_id!r}")


@dataclass(frozen=True)
class BoundOperatorSelection(_JsonRecord):
    """An exact operator selection portable across framework worker processes."""

    module: str
    workload: OperatorWorkloadKey
    requirements: OperatorRequirements
    backend_id: str
    descriptor_fingerprint: str
    implementation_fingerprint: str | None
    selection_reason: str
    cache_reason: str

    def __post_init__(self) -> None:
        module = self.module.strip().lower()
        if module not in {"attention", "ffn", "logp", "moe", "speculative"}:
            raise ValueError(f"unsupported bound selection module {self.module!r}")
        object.__setattr__(self, "module", module)

    @property
    def target(self) -> str:
        return "training" if self.workload.phase is ExecutionPhase.TRAIN else "rollout"

    @classmethod
    def from_cached(cls, module: str, selection: CachedSelection) -> BoundOperatorSelection:
        decision = selection.decision
        return cls(
            module=module,
            workload=replace(decision.workload, shape=selection.requested_shape),
            requirements=decision.resolution.requirements,
            backend_id=decision.selected_backend_id,
            descriptor_fingerprint=decision.resolution.descriptor.descriptor_fingerprint,
            implementation_fingerprint=decision.implementation_fingerprint,
            selection_reason=decision.reason,
            cache_reason=selection.cache_reason,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BoundOperatorSelection:
        data = dict(payload)
        data["workload"] = OperatorWorkloadKey.from_dict(data["workload"])
        data["requirements"] = OperatorRequirements(**data["requirements"])
        return cls(**data)

    def matches_provenance(self, value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        if value.get("backend_id") == self.backend_id and "descriptor_fingerprint" in value:
            return value["descriptor_fingerprint"] == self.descriptor_fingerprint and (
                self.implementation_fingerprint is None
                or value.get("implementation_fingerprint") == self.implementation_fingerprint
            )
        return any(self.matches_provenance(item) for item in value.values())

    def instantiate(
        self,
        session: OperatorSession,
        *,
        factory_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[Any, OperatorInstanceProvenance]:
        resolution = session.resolve(
            semantic_op=self.workload.semantic_op,
            requested_backend=self.backend_id,
            target=self.target,
            requirements=self.requirements,
            strict=True,
        )
        if resolution.descriptor.descriptor_fingerprint != self.descriptor_fingerprint:
            raise OperatorInstantiationError(f"bound operator {self.backend_id!r} changed")
        instance = session.instantiate(resolution, factory_kwargs=factory_kwargs)
        provenance = session.instance_provenance(resolution, instance)
        if self.implementation_fingerprint is not None and (
            provenance.implementation_fingerprint != self.implementation_fingerprint
        ):
            raise OperatorInstantiationError(
                f"implementation of bound operator {self.backend_id!r} changed"
            )
        return instance, provenance


@dataclass(frozen=True)
class IntegrationPlan:
    """One independently selectable P/R case for Attention, FFN, and Logp."""

    cases: Mapping[str, OperatorAblationCase]
    bound_selections: tuple[BoundOperatorSelection, ...] = ()

    def __post_init__(self) -> None:
        normalized = dict(self.cases)
        if set(normalized) != {"attention", "ffn", "logp"}:
            raise ValueError("integration plan must define attention, ffn, and logp")
        for module, case in normalized.items():
            if not isinstance(case, OperatorAblationCase) or case.module != module:
                raise ValueError(f"invalid integration case for {module!r}")
        object.__setattr__(self, "cases", MappingProxyType(normalized))
        bindings = tuple(self.bound_selections)
        if len({(item.module, item.target) for item in bindings}) != len(bindings):
            raise ValueError("operator selections must be unique per module and framework side")
        object.__setattr__(self, "bound_selections", bindings)

    def with_bound_selections(self, *selections: BoundOperatorSelection) -> "IntegrationPlan":
        return IntegrationPlan(self.cases, self.bound_selections + tuple(selections))

    @classmethod
    def from_case_ids(
        cls,
        *,
        attention: str = "P/P",
        ffn: str = "P/P",
        logp: str = "P/P",
    ) -> "IntegrationPlan":
        return cls(
            {
                "attention": operator_ablation_case("attention", attention),
                "ffn": operator_ablation_case("ffn", ffn),
                "logp": operator_ablation_case("logp", logp),
            }
        )

    def binding_for(self, module: str, target: str) -> BoundOperatorSelection | None:
        return next(
            (
                item
                for item in self.bound_selections
                if item.module == module and item.target == target
            ),
            None,
        )

    def implementation_for(self, module: str, target: str) -> Implementation:
        module = module.strip().lower()
        target = target.strip().lower()
        if self.binding_for(module, target) is not None:
            return Implementation.RL_KERNEL
        try:
            case = self.cases[module]
        except KeyError as exc:
            raise ValueError(f"unknown integration module {module!r}") from exc
        return case.implementation_for(target)

    def to_dict(self) -> dict[str, Any]:
        matrix = module_debug_matrix()
        result: dict[str, Any] = {
            "schema_version": matrix["schema_version"],
            "cases": {module: case.to_dict() for module, case in self.cases.items()},
        }
        if self.bound_selections:
            result["bound_selections"] = [item.to_dict() for item in self.bound_selections]
        return result


def integration_plan_from_environment() -> IntegrationPlan:
    """Materialize the one plan inherited by framework worker processes."""

    plan = IntegrationPlan.from_case_ids(
        attention=os.getenv("RL_KERNEL_ATTENTION_CASE", "P/P"),
        ffn=os.getenv("RL_KERNEL_FFN_CASE", "P/P"),
        logp=os.getenv("RL_KERNEL_LOGP_CASE", "P/P"),
    )
    bindings = json.loads(os.environ.get("RL_KERNEL_OPERATOR_BINDINGS", "[]"))
    if bindings:
        plan = plan.with_bound_selections(
            *(BoundOperatorSelection.from_dict(item) for item in bindings)
        )
    return plan


def configure_integration_environment(
    plan: IntegrationPlan,
    *,
    readback_dir: str | None = None,
) -> None:
    """Export one plan for Megatron actors and vLLM subprocesses."""

    for module, variable in (
        ("attention", "RL_KERNEL_ATTENTION_CASE"),
        ("ffn", "RL_KERNEL_FFN_CASE"),
        ("logp", "RL_KERNEL_LOGP_CASE"),
    ):
        os.environ[variable] = plan.cases[module].case_id
    if plan.bound_selections:
        os.environ["RL_KERNEL_OPERATOR_BINDINGS"] = json.dumps(
            [binding.to_dict() for binding in plan.bound_selections]
        )
    else:
        os.environ.pop("RL_KERNEL_OPERATOR_BINDINGS", None)
    if readback_dir:
        os.environ["RL_KERNEL_READBACK_DIR"] = readback_dir


__all__ = [
    "BoundOperatorSelection",
    "Implementation",
    "IntegrationPlan",
    "OperatorAblationCase",
    "configure_integration_environment",
    "integration_plan_from_environment",
    "operator_ablation_case",
    "operator_ablation_cases",
]
