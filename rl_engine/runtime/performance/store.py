# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Storage for separated operator and end-to-end performance records (#563 Task 32)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from rl_engine.runtime.performance.schema import (
    EndToEndPerformanceRecord,
    OperatorPerformanceRecord,
    OperatorWorkloadKey,
)

from rl_engine.runtime.semantic_registry import _fingerprint

PERFORMANCE_STORE_SCHEMA_VERSION = "rlkernel.runtime.performance.store.v1"


class PerformanceStoreError(ValueError):
    """Raised when performance records conflict or fail integrity checks."""


class PerformanceStore:
    """In-memory and JSON-serializable store for operator and end-to-end records."""

    def __init__(
        self,
        *,
        operator_records: Iterable[OperatorPerformanceRecord] = (),
        e2e_records: Iterable[EndToEndPerformanceRecord] = (),
    ) -> None:
        self._operator_records: dict[tuple[str, str, str, str], OperatorPerformanceRecord] = {}
        self._e2e_records: dict[tuple[str, str, str, str, str, str], EndToEndPerformanceRecord] = {}
        for record in operator_records:
            self.add_operator_record(record)
        for record in e2e_records:
            self.add_e2e_record(record)

    @staticmethod
    def _operator_key(record: OperatorPerformanceRecord) -> tuple[str, str, str, str]:
        return (
            record.workload.key_fingerprint,
            record.backend_id,
            record.descriptor_fingerprint,
            record.implementation_fingerprint,
        )

    @staticmethod
    def _e2e_key(record: EndToEndPerformanceRecord) -> tuple[str, str, str, str, str, str]:
        return (
            record.workload_id,
            record.phase.value,
            record.model_id,
            _fingerprint(record.device.to_dict()),
            record.plan_fingerprint,
            _fingerprint(record.topology),
        )

    def add_operator_record(
        self, record: OperatorPerformanceRecord, *, replace: bool = False
    ) -> None:
        key = self._operator_key(record)
        existing = self._operator_records.get(key)
        if existing is not None and not replace and existing != record:
            raise PerformanceStoreError(
                f"operator performance record already exists for {record.backend_id!r}"
            )
        self._operator_records[key] = record

    def add_e2e_record(self, record: EndToEndPerformanceRecord, *, replace: bool = False) -> None:
        key = self._e2e_key(record)
        existing = self._e2e_records.get(key)
        if existing is not None and not replace and existing != record:
            raise PerformanceStoreError(
                f"end-to-end performance record already exists for {record.workload_id!r}"
            )
        self._e2e_records[key] = record

    def lookup_operator_record(
        self,
        workload: OperatorWorkloadKey,
        *,
        backend_id: str,
        descriptor_fingerprint: str | None = None,
        implementation_fingerprint: str | None = None,
    ) -> OperatorPerformanceRecord | None:
        prefix = (workload.key_fingerprint, backend_id.strip())
        if descriptor_fingerprint is not None and implementation_fingerprint is not None:
            return self._operator_records.get(
                (*prefix, descriptor_fingerprint, implementation_fingerprint)
            )

        matches = (
            record
            for key, record in self._operator_records.items()
            if key[:2] == prefix
            and (descriptor_fingerprint is None or key[2] == descriptor_fingerprint)
            and (implementation_fingerprint is None or key[3] == implementation_fingerprint)
        )
        return min(
            matches,
            key=lambda record: (
                record.total_latency_ms,
                float("inf")
                if record.peak_workspace_bytes is None
                else record.peak_workspace_bytes,
                record.record_fingerprint,
            ),
            default=None,
        )

    def operator_records(
        self, workload: OperatorWorkloadKey | None = None
    ) -> tuple[OperatorPerformanceRecord, ...]:
        values = self._operator_records.values()
        if workload is not None:
            target_wf = workload.key_fingerprint
            values = [rec for key, rec in self._operator_records.items() if key[0] == target_wf]
        return tuple(sorted(values, key=self._operator_key))

    def e2e_records(self, workload_id: str | None = None) -> tuple[EndToEndPerformanceRecord, ...]:
        values = self._e2e_records.values()
        if workload_id is not None:
            values = [rec for rec in values if rec.workload_id == workload_id.strip()]
        return tuple(sorted(values, key=self._e2e_key))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PERFORMANCE_STORE_SCHEMA_VERSION,
            "operator_records": [rec.to_dict() for rec in self.operator_records()],
            "e2e_records": [rec.to_dict() for rec in self.e2e_records()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PerformanceStore:
        if payload.get("schema_version") != PERFORMANCE_STORE_SCHEMA_VERSION:
            raise PerformanceStoreError("unsupported performance store schema_version")
        return cls(
            operator_records=[
                OperatorPerformanceRecord.from_dict(r) for r in payload.get("operator_records", ())
            ],
            e2e_records=[
                EndToEndPerformanceRecord.from_dict(r) for r in payload.get("e2e_records", ())
            ],
        )

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> PerformanceStore:
        return cls.from_dict(json.loads(Path(path).read_text()))


__all__ = [
    "PERFORMANCE_STORE_SCHEMA_VERSION",
    "PerformanceStore",
    "PerformanceStoreError",
]
