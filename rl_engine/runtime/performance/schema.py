# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Schemas for warmed-up operator and end-to-end performance records (#563 Task 32)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from rl_engine.runtime.semantic_registry import (
    _fingerprint,
    _freeze,
    _JsonRecord,
    _normalize_device,
    _normalize_dtype,
)


class ExecutionPhase(str, Enum):
    TRAIN = "train"
    PREFILL = "prefill"
    DECODE = "decode"
    VERIFY = "verify"


class MeasurementScope(str, Enum):
    OPERATOR = "operator"
    END_TO_END = "end_to_end"


_LEGACY_OPERATOR_RECORD_VERSION = "rlkernel.runtime.performance.operator_record.v1"
OPERATOR_RECORD_SCHEMA_VERSION = "rlkernel.runtime.performance.operator_record.v2"
E2E_RECORD_SCHEMA_VERSION = "rlkernel.runtime.performance.e2e_record.v1"


@dataclass(frozen=True)
class DeviceSoftwareIdentity(_JsonRecord):
    """Reproducible hardware and software version identity for a measurement."""

    device_type: str
    device_model: str
    software_version: str
    driver_version: str = "unknown"
    torch_version: str = "unknown"

    def __post_init__(self) -> None:
        if not self.device_model.strip() or not self.software_version.strip():
            raise ValueError("device_model and software_version must not be empty")
        object.__setattr__(self, "device_type", _normalize_device(self.device_type))
        object.__setattr__(self, "device_model", self.device_model.strip())
        object.__setattr__(self, "software_version", self.software_version.strip())
        object.__setattr__(self, "driver_version", self.driver_version.strip() or "unknown")
        object.__setattr__(self, "torch_version", self.torch_version.strip() or "unknown")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeviceSoftwareIdentity:
        return cls(**payload)


@dataclass(frozen=True)
class OperatorWorkloadKey(_JsonRecord):
    """Lookup key binding an operator measurement to its exact configuration."""

    semantic_op: str
    phase: ExecutionPhase
    device_type: str
    device_model: str
    software_version: str
    dtype: str
    layout: str
    shape: tuple[int, ...]
    numerical_contract_id: str
    topology: Mapping[str, Any] = field(default_factory=dict)
    expert_token_counts: tuple[int, ...] | None = None
    speculative_draft_len: int | None = None

    def __post_init__(self) -> None:
        dims = tuple(int(d) for d in self.shape)
        if not dims or any(d < 0 for d in dims):
            raise ValueError("shape must be a non-empty tuple of non-negative ints")
        if not self.semantic_op.strip() or not self.numerical_contract_id.strip():
            raise ValueError("semantic_op and numerical_contract_id must not be empty")
        object.__setattr__(self, "semantic_op", self.semantic_op.strip())
        object.__setattr__(self, "phase", ExecutionPhase(self.phase))
        object.__setattr__(self, "device_type", _normalize_device(self.device_type))
        object.__setattr__(self, "device_model", self.device_model.strip())
        object.__setattr__(self, "software_version", self.software_version.strip())
        object.__setattr__(self, "dtype", _normalize_dtype(self.dtype))
        object.__setattr__(self, "layout", self.layout.strip())
        object.__setattr__(self, "shape", dims)
        object.__setattr__(self, "numerical_contract_id", self.numerical_contract_id.strip())
        object.__setattr__(self, "topology", _freeze(self.topology))
        if self.expert_token_counts is not None:
            counts = tuple(self.expert_token_counts)
            if any(count < 0 for count in counts):
                raise ValueError("expert token counts must be nonnegative")
            object.__setattr__(self, "expert_token_counts", counts)
        if self.speculative_draft_len is not None and self.speculative_draft_len < 0:
            raise ValueError("speculative draft length must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        for key in ("expert_token_counts", "speculative_draft_len"):
            if data[key] is None:
                del data[key]
        return data

    @property
    def key_fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OperatorWorkloadKey:
        data = dict(payload)
        data["shape"] = tuple(data["shape"])
        return cls(**data)


@dataclass(frozen=True)
class OperatorPerformanceRecord(_JsonRecord):
    """Warmed-up operator latency and peak workspace measurement record."""

    workload: OperatorWorkloadKey
    backend_id: str
    descriptor_fingerprint: str
    implementation_fingerprint: str
    warmup_iterations: int
    measured_iterations: int
    synchronized: bool
    kernel_latency_ms: float
    launch_overhead_ms: float = 0.0
    layout_conversion_ms: float = 0.0
    synchronization_ms: float = 0.0
    peak_workspace_bytes: int | None = None
    peak_allocated_bytes: int | None = None
    scope: MeasurementScope = MeasurementScope.OPERATOR
    schema_version: str = OPERATOR_RECORD_SCHEMA_VERSION
    elapsed_latency_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.synchronized:
            raise ValueError("operator measurements must be synchronized (synchronized=True)")
        if self.warmup_iterations < 1 or self.measured_iterations < 1:
            raise ValueError("warmup_iterations and measured_iterations must be >= 1")
        times = (
            self.kernel_latency_ms,
            self.launch_overhead_ms,
            self.layout_conversion_ms,
            self.synchronization_ms,
        )
        if self.kernel_latency_ms <= 0 or any(not math.isfinite(t) or t < 0 for t in times):
            raise ValueError("latencies must be finite and >= 0, with kernel_latency_ms > 0")
        if self.elapsed_latency_ms is not None and (
            not math.isfinite(self.elapsed_latency_ms) or self.elapsed_latency_ms <= 0
        ):
            raise ValueError("elapsed_latency_ms must be finite and positive")
        workspace, allocated = self.peak_workspace_bytes, self.peak_allocated_bytes
        if any(size is not None and size < 0 for size in (workspace, allocated)) or (
            workspace is not None and allocated is not None and workspace > allocated
        ):
            raise ValueError("observed memory sizes must satisfy 0 <= workspace <= allocated")
        object.__setattr__(self, "scope", MeasurementScope.OPERATOR)
        object.__setattr__(self, "backend_id", self.backend_id.strip())

    @property
    def total_latency_ms(self) -> float:
        if self.elapsed_latency_ms is not None:
            return self.elapsed_latency_ms
        return (
            self.kernel_latency_ms
            + self.launch_overhead_ms
            + self.layout_conversion_ms
            + self.synchronization_ms
        )

    @property
    def record_fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_record_fingerprint=False))

    def to_dict(self, *, include_record_fingerprint: bool = True) -> dict[str, Any]:
        result = super().to_dict()
        result["total_latency_ms"] = self.total_latency_ms
        if include_record_fingerprint:
            result["record_fingerprint"] = self.record_fingerprint
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OperatorPerformanceRecord:
        data = dict(payload)
        recorded_fp = data.pop("record_fingerprint", None)
        version = data.get("schema_version", _LEGACY_OPERATOR_RECORD_VERSION)
        if version == _LEGACY_OPERATOR_RECORD_VERSION:
            if recorded_fp is not None and recorded_fp != _fingerprint(data):
                raise ValueError("operator performance record fingerprint mismatch")
            # v1 wrote zero even when its author had not measured the field.
            for key in ("peak_workspace_bytes", "peak_allocated_bytes"):
                if data.get(key) == 0:
                    data[key] = None
            data["schema_version"] = OPERATOR_RECORD_SCHEMA_VERSION
            recorded_fp = None
        elif version != OPERATOR_RECORD_SCHEMA_VERSION:
            raise ValueError(f"unsupported operator performance record version {version!r}")
        data.pop("total_latency_ms", None)
        data["workload"] = OperatorWorkloadKey.from_dict(data["workload"])
        record = cls(**data)
        if recorded_fp is not None and recorded_fp != record.record_fingerprint:
            raise ValueError("operator performance record fingerprint mismatch")
        return record


@dataclass(frozen=True)
class EndToEndPerformanceRecord(_JsonRecord):
    """End-to-end step/rollout performance record kept separate from operator data."""

    workload_id: str
    model_id: str
    phase: ExecutionPhase
    device: DeviceSoftwareIdentity
    plan_fingerprint: str
    warmup_steps: int
    measured_steps: int
    step_latency_ms: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    topology: Mapping[str, Any] = field(default_factory=dict)
    scope: MeasurementScope = MeasurementScope.END_TO_END
    schema_version: str = E2E_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.warmup_steps < 1 or self.measured_steps < 1 or self.step_latency_ms <= 0:
            raise ValueError("warmup_steps >= 1, measured_steps >= 1, step_latency_ms > 0")
        if self.peak_allocated_bytes < 0 or self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("0 <= peak_allocated_bytes <= peak_reserved_bytes is required")
        object.__setattr__(self, "phase", ExecutionPhase(self.phase))
        object.__setattr__(self, "scope", MeasurementScope.END_TO_END)
        object.__setattr__(self, "topology", _freeze(self.topology))

    @property
    def record_fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_record_fingerprint=False))

    def to_dict(self, *, include_record_fingerprint: bool = True) -> dict[str, Any]:
        result = super().to_dict()
        if include_record_fingerprint:
            result["record_fingerprint"] = self.record_fingerprint
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EndToEndPerformanceRecord:
        data = dict(payload)
        recorded_fp = data.pop("record_fingerprint", None)
        data["device"] = DeviceSoftwareIdentity.from_dict(data["device"])
        record = cls(**data)
        if recorded_fp is not None and recorded_fp != record.record_fingerprint:
            raise ValueError("end-to-end performance record fingerprint mismatch")
        return record


__all__ = [
    "DeviceSoftwareIdentity",
    "E2E_RECORD_SCHEMA_VERSION",
    "EndToEndPerformanceRecord",
    "ExecutionPhase",
    "MeasurementScope",
    "OPERATOR_RECORD_SCHEMA_VERSION",
    "OperatorPerformanceRecord",
    "OperatorWorkloadKey",
]
