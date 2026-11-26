# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Operator selection within numerical and memory constraints (#563 Task 33)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, NamedTuple, Sequence

from rl_engine.runtime.performance.schema import (
    ExecutionPhase,
    OperatorPerformanceRecord,
    OperatorWorkloadKey,
)
from rl_engine.runtime.performance.store import PerformanceStore
from rl_engine.runtime.semantic_registry import (
    OperatorBackendDescriptor,
    OperatorRequirements,
    OperatorResolution,
    OperatorResolutionError,
    OperatorResolutionPolicy,
    SemanticOperatorCatalog,
    _freeze,
    _JsonRecord,
)

SELECTION_DECISION_SCHEMA_VERSION = "rlkernel.runtime.selector.decision.v1"


class SelectionObjective(str, Enum):
    MIN_LATENCY = "min_latency"
    MIN_WORKSPACE = "min_workspace"


class SelectionStatus(str, Enum):
    TUNED = "tuned"
    UNTUNED = "untuned"
    PINNED = "pinned"


@dataclass(frozen=True)
class CandidateRejection(_JsonRecord):
    """Audit record explaining why a candidate was excluded."""

    backend_id: str
    stage: str
    reason: str


@dataclass(frozen=True)
class SelectionPolicy(_JsonRecord):
    """Constraints and objective governing operator selection for one request."""

    phase: ExecutionPhase
    objective: SelectionObjective = SelectionObjective.MIN_LATENCY
    max_workspace_bytes: int | None = None
    pinned_backend_id: str | None = None
    enable_tuning: bool = True
    default_backend_id: str | None = None
    resolution_policy: OperatorResolutionPolicy = field(default_factory=OperatorResolutionPolicy)
    # Measurements are eligible only when their implementation identity matches.
    implementation_fingerprints: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_workspace_bytes is not None and self.max_workspace_bytes < 0:
            raise ValueError("max_workspace_bytes must be >= 0 or None")
        object.__setattr__(self, "phase", ExecutionPhase(self.phase))
        object.__setattr__(self, "objective", SelectionObjective(self.objective))
        if self.pinned_backend_id is not None:
            object.__setattr__(self, "pinned_backend_id", self.pinned_backend_id.strip())
        if self.default_backend_id is not None:
            object.__setattr__(self, "default_backend_id", self.default_backend_id.strip())
        object.__setattr__(
            self, "implementation_fingerprints", _freeze(dict(self.implementation_fingerprints))
        )

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["resolution_policy"] = asdict(self.resolution_policy)
        return result


@dataclass(frozen=True)
class SelectionDecision(_JsonRecord):
    """Auditable selection result tying the chosen backend to its evidence."""

    workload: OperatorWorkloadKey
    selected_backend_id: str
    resolution: OperatorResolution
    status: SelectionStatus
    reason: str
    objective: SelectionObjective
    max_workspace_bytes: int | None
    measurement: OperatorPerformanceRecord | None = None
    eligible_backends: tuple[str, ...] = ()
    rejected_candidates: tuple[CandidateRejection, ...] = ()
    schema_version: str = SELECTION_DECISION_SCHEMA_VERSION
    implementation_fingerprint: str | None = None

    @property
    def is_untuned(self) -> bool:
        return self.status is SelectionStatus.UNTUNED

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["is_untuned"] = self.is_untuned
        return result


class OperatorSelectionError(RuntimeError):
    """Raised when no numerically validated candidate satisfies constraints."""

    def __init__(
        self,
        message: str,
        *,
        workload: OperatorWorkloadKey,
        rejected_candidates: Sequence[CandidateRejection] = (),
    ) -> None:
        super().__init__(message)
        self.workload = workload
        self.rejected_candidates = tuple(rejected_candidates)


class _CandidateProfile(NamedTuple):
    backend_id: str
    descriptor: OperatorBackendDescriptor | None
    measurement: OperatorPerformanceRecord | None


def _workload_rejection(
    workload: OperatorWorkloadKey, descriptor: OperatorBackendDescriptor
) -> str | None:
    properties = descriptor.determinism_or_alignment_properties
    contract = properties.get("numerical_contract_id")
    if contract != workload.numerical_contract_id:
        return f"contract {contract!r} != {workload.numerical_contract_id!r}"
    # The registry declares the validated domain for each physical platform.
    domain = properties.get("validated_workloads", {}).get(workload.device_type, {})
    for dimension, value in (
        ("phases", workload.phase.value),
        ("layouts", workload.layout),
        ("shapes", workload.shape),
    ):
        allowed = domain.get(dimension, ())
        if allowed != "*" and value not in allowed:
            return f"{dimension}: {value!r} is outside the validated workload domain"
    return None


def _workspace_bytes(
    descriptor: OperatorBackendDescriptor, measurement: OperatorPerformanceRecord | None
) -> int | None:
    if measurement is not None and measurement.peak_workspace_bytes is not None:
        return measurement.peak_workspace_bytes
    properties = descriptor.determinism_or_alignment_properties
    declared = properties.get("workspace_bytes", properties.get("max_workspace_bytes"))
    return (
        declared
        if isinstance(declared, int) and not isinstance(declared, bool) and declared >= 0
        else None
    )


def _check_budget(
    budget: int | None,
    descriptor: OperatorBackendDescriptor,
    measurement: OperatorPerformanceRecord | None,
) -> str | None:
    if budget is None:
        return None
    workspace = _workspace_bytes(descriptor, measurement)
    if workspace is None:
        return f"candidate has no known workspace bound under budget ({budget})"
    if workspace > budget:
        return f"workspace ({workspace}) > budget ({budget})"
    return None


class OperatorSelector:
    """Select an operator implementation within numerical and memory constraints."""

    def __init__(
        self, catalog: SemanticOperatorCatalog, store: PerformanceStore | None = None
    ) -> None:
        self.catalog = catalog
        self.store = store or PerformanceStore()

    def _profiles(
        self,
        workload: OperatorWorkloadKey,
        policy: SelectionPolicy,
        requested: Sequence[str] | None,
    ) -> tuple[_CandidateProfile, ...]:
        if requested is not None:
            backend_ids = dict.fromkeys(backend.strip() for backend in requested if backend.strip())
            candidates = (
                (backend_id, self.catalog.backend_descriptor(workload.semantic_op, backend_id))
                for backend_id in backend_ids
            )
        else:
            candidates = (
                (desc.backend_id, desc)
                for desc in self.catalog.backend_descriptors(workload.semantic_op)
            )
        profiles = []
        for backend_id, descriptor in candidates:
            fingerprint = policy.implementation_fingerprints.get(backend_id)
            measurement = None
            if fingerprint and descriptor is not None:
                measurement = self.store.lookup_operator_record(
                    workload,
                    backend_id=backend_id,
                    descriptor_fingerprint=descriptor.descriptor_fingerprint,
                    implementation_fingerprint=fingerprint,
                )
            profiles.append(_CandidateProfile(backend_id, descriptor, measurement))
        return tuple(profiles)

    def select(
        self,
        *,
        workload: OperatorWorkloadKey,
        requirements: OperatorRequirements,
        policy: SelectionPolicy,
        candidate_backends: Sequence[str] | None = None,
    ) -> SelectionDecision:
        return self._select(
            workload, requirements, policy, self._profiles(workload, policy, candidate_backends)
        )

    def _select(
        self,
        workload: OperatorWorkloadKey,
        requirements: OperatorRequirements,
        policy: SelectionPolicy,
        profiles: Sequence[_CandidateProfile],
    ) -> SelectionDecision:
        if workload.phase is not policy.phase:
            raise ValueError(f"workload phase {workload.phase} != policy phase {policy.phase}")
        if (workload.device_type, workload.dtype, workload.topology) != (
            requirements.device,
            requirements.dtype,
            requirements.topology,
        ):
            raise ValueError("workload and requirements must agree on device, dtype, and topology")

        target = "training" if policy.phase is ExecutionPhase.TRAIN else "rollout"
        session = self.catalog.session(policy.resolution_policy)
        eligible: dict[str, OperatorResolution] = {}
        measurements: dict[str, OperatorPerformanceRecord] = {}
        rejected: list[CandidateRejection] = []

        for backend_id, descriptor, measurement in profiles:
            try:
                resolution = session.resolve(
                    semantic_op=workload.semantic_op,
                    requested_backend=backend_id,
                    target=target,
                    requirements=requirements,
                )
            except OperatorResolutionError as exc:
                rejected.append(
                    CandidateRejection(backend_id, "capability_or_numerical_contract", str(exc))
                )
                continue

            if resolution.descriptor is not descriptor:
                raise OperatorSelectionError(
                    f"backend {backend_id!r} changed during selection", workload=workload
                )
            reason = _workload_rejection(workload, descriptor)
            if reason is not None:
                rejected.append(CandidateRejection(backend_id, "workload", reason))
                continue
            reason = _check_budget(policy.max_workspace_bytes, descriptor, measurement)
            if reason is not None:
                rejected.append(CandidateRejection(backend_id, "memory_budget", reason))
                continue
            eligible[backend_id] = resolution
            if measurement is not None and (
                policy.objective is SelectionObjective.MIN_LATENCY
                or _workspace_bytes(descriptor, measurement) is not None
            ):
                measurements[backend_id] = measurement

        if not eligible:
            raise OperatorSelectionError(
                f"no eligible backend for {workload.semantic_op!r}",
                workload=workload,
                rejected_candidates=rejected,
            )

        if policy.pinned_backend_id is not None:
            chosen = policy.pinned_backend_id
            status = SelectionStatus.PINNED
            reason = "pinned candidate passed numerical and memory checks"
        elif policy.enable_tuning and measurements:
            by_workspace = policy.objective is SelectionObjective.MIN_WORKSPACE

            def score(record: OperatorPerformanceRecord) -> tuple[float, float, str]:
                workspace = _workspace_bytes(eligible[record.backend_id].descriptor, record)
                workspace = float("inf") if workspace is None else workspace
                latency = record.total_latency_ms
                if by_workspace:
                    return workspace, latency, record.backend_id
                return latency, workspace, record.backend_id

            chosen = min(measurements.values(), key=score).backend_id
            status = SelectionStatus.TUNED
            reason = f"selected by {policy.objective.value}"
        else:
            chosen = policy.default_backend_id or next(iter(eligible))
            status = SelectionStatus.UNTUNED
            reason = "no matching measurements" if policy.enable_tuning else "tuning disabled"

        if chosen not in eligible:
            raise OperatorSelectionError(
                f"{status.value} backend {chosen!r} is ineligible or over budget",
                workload=workload,
                rejected_candidates=rejected,
            )

        return SelectionDecision(
            workload=workload,
            selected_backend_id=chosen,
            resolution=eligible[chosen],
            status=status,
            reason=reason,
            objective=policy.objective,
            max_workspace_bytes=policy.max_workspace_bytes,
            measurement=measurements.get(chosen) if status is not SelectionStatus.UNTUNED else None,
            eligible_backends=tuple(eligible),
            rejected_candidates=tuple(rejected),
            implementation_fingerprint=policy.implementation_fingerprints.get(chosen),
        )


__all__ = [
    "CandidateRejection",
    "OperatorSelectionError",
    "OperatorSelector",
    "SELECTION_DECISION_SCHEMA_VERSION",
    "SelectionDecision",
    "SelectionObjective",
    "SelectionPolicy",
    "SelectionStatus",
]
