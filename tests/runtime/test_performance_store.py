# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for runtime performance records and storage (#563 Task 32)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rl_engine.runtime.semantic_registry import _fingerprint

from rl_engine.runtime.performance import (
    DeviceSoftwareIdentity,
    EndToEndPerformanceRecord,
    ExecutionPhase,
    MeasurementScope,
    OperatorPerformanceRecord,
    OperatorWorkloadKey,
    PerformanceStore,
    PerformanceStoreError,
)


def _workload() -> OperatorWorkloadKey:
    return OperatorWorkloadKey(
        semantic_op="det_gemm",
        phase=ExecutionPhase.PREFILL,
        device_type="npu",
        device_model="Ascend910B",
        software_version="CANN-8.0",
        dtype="bfloat16",
        layout="row_major",
        shape=(128, 4096),
        numerical_contract_id="contract.gemm.strict.v1",
        topology={"tp": 2, "cp": 1},
    )


def _op_record(
    workload: OperatorWorkloadKey, *, kernel_ms: float = 1.5, workspace: int = 1024
) -> OperatorPerformanceRecord:
    return OperatorPerformanceRecord(
        workload=workload,
        backend_id="ascend_cube_v1",
        descriptor_fingerprint="desc_fp_1",
        implementation_fingerprint="impl_fp_1",
        warmup_iterations=5,
        measured_iterations=20,
        synchronized=True,
        kernel_latency_ms=kernel_ms,
        launch_overhead_ms=0.1,
        layout_conversion_ms=0.2,
        synchronization_ms=0.05,
        peak_workspace_bytes=workspace,
        peak_allocated_bytes=workspace * 2,
    )


def test_operator_record_totals_and_validation():
    wl = _workload()
    rec = _op_record(wl, kernel_ms=2.0, workspace=4096)
    assert rec.scope is MeasurementScope.OPERATOR
    assert pytest.approx(rec.total_latency_ms) == 2.35

    with pytest.raises(ValueError, match="synchronized"):
        OperatorPerformanceRecord(
            workload=wl,
            backend_id="ascend_cube_v1",
            descriptor_fingerprint="d1",
            implementation_fingerprint="i1",
            warmup_iterations=5,
            measured_iterations=10,
            synchronized=False,
            kernel_latency_ms=1.0,
        )


def test_store_separates_operator_and_e2e_records_and_roundtrips_json(tmp_path: Path):
    wl = _workload()
    op_rec = _op_record(wl)
    e2e_rec = EndToEndPerformanceRecord(
        workload_id="qwen3_6_moe_tp2",
        model_id="Qwen3.6-35B-A3B",
        phase=ExecutionPhase.TRAIN,
        device=DeviceSoftwareIdentity("npu", "Ascend910B", "CANN-8.0"),
        plan_fingerprint="plan_fp_1",
        warmup_steps=2,
        measured_steps=10,
        step_latency_ms=420.5,
        peak_allocated_bytes=32 * 1024**3,
        peak_reserved_bytes=36 * 1024**3,
        topology={"tp": 2, "ep": 4},
    )

    updated = replace(e2e_rec, device=replace(e2e_rec.device, software_version="CANN-8.1"))
    store = PerformanceStore(operator_records=[op_rec], e2e_records=[e2e_rec, updated])
    path = store.save_json(tmp_path / "perf.json")
    reloaded = PerformanceStore.load_json(path)

    assert reloaded.operator_records() == (op_rec,)
    assert {record.device.software_version for record in reloaded.e2e_records()} == {
        "CANN-8.0",
        "CANN-8.1",
    }
    assert reloaded.lookup_operator_record(wl, backend_id="ascend_cube_v1") == op_rec
    assert (
        reloaded.lookup_operator_record(
            wl, backend_id="ascend_cube_v1", descriptor_fingerprint="stale"
        )
        is None
    )

    with pytest.raises(PerformanceStoreError, match="already exists"):
        store.add_operator_record(_op_record(wl, kernel_ms=3.0))


def test_old_zero_workspace_is_unknown_but_explicit_zero_roundtrips():
    current = replace(_op_record(_workload()), peak_workspace_bytes=0, peak_allocated_bytes=0)
    old = current.to_dict(include_record_fingerprint=False)
    old.pop("elapsed_latency_ms")
    old["schema_version"] = "rlkernel.runtime.performance.operator_record.v1"
    old["record_fingerprint"] = _fingerprint(old)

    assert OperatorPerformanceRecord.from_dict(old).peak_workspace_bytes is None
    assert OperatorPerformanceRecord.from_dict(current.to_dict()).peak_workspace_bytes == 0
