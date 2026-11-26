# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for constraint-aware operator selector (#563 Task 33)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rl_engine.runtime.performance import (
    ExecutionPhase,
    OperatorPerformanceRecord,
    OperatorWorkloadKey,
    PerformanceStore,
)
from rl_engine.runtime.selector import (
    OperatorSelectionError,
    OperatorSelector,
    SelectionObjective,
    SelectionPolicy,
    SelectionStatus,
)
from rl_engine.runtime.semantic_registry import (
    OperatorBackendDescriptor,
    OperatorFallbackPolicy,
    OperatorLifecycle,
    OperatorRequirements,
    SemanticOperatorCatalog,
)


def _workload(phase: ExecutionPhase = ExecutionPhase.PREFILL) -> OperatorWorkloadKey:
    return OperatorWorkloadKey(
        semantic_op="det_gemm",
        phase=phase,
        device_type="npu",
        device_model="Ascend910B",
        software_version="CANN-8.0",
        dtype="bfloat16",
        layout="row_major",
        shape=(128, 4096),
        numerical_contract_id="contract.gemm.strict.v1",
        topology={"tp": 2, "cp": 1},
    )


def _descriptor(
    backend_id: str,
    *,
    contract_id: str = "contract.gemm.strict.v1",
    workspace_bytes: int = 1024,
) -> OperatorBackendDescriptor:
    return OperatorBackendDescriptor(
        semantic_op="det_gemm",
        backend_id=backend_id,
        supported_targets=frozenset({"training", "rollout"}),
        supported_devices=frozenset({"npu"}),
        supported_dtypes=frozenset({"bfloat16"}),
        supported_topologies={
            "training": {"tp": (1, 2, 4), "cp": (1, 2)},
            "rollout": {"tp": (1, 2, 4), "cp": (1,)},
        },
        determinism_or_alignment_properties={
            "deterministic": True,
            "numerical_contract_id": contract_id,
            "workspace_bytes": workspace_bytes,
            "validated_workloads": {
                "npu": {
                    "phases": ("train", "prefill"),
                    "layouts": ("row_major",),
                    "shapes": ((64, 4096), (128, 4096)),
                }
            },
        },
        lifecycle=OperatorLifecycle.REQUEST,
        implementation_class_or_factory=lambda: object(),
        fallback_policy=OperatorFallbackPolicy.ERROR,
        version_or_build_fingerprint="v1",
    )


def _reqs() -> OperatorRequirements:
    return OperatorRequirements(
        device="npu",
        dtype="bfloat16",
        topology={"tp": 2, "cp": 1},
        alignment_properties={
            "deterministic": True,
            "numerical_contract_id": "contract.gemm.strict.v1",
        },
    )


def _rec(
    wl: OperatorWorkloadKey,
    desc: OperatorBackendDescriptor,
    *,
    kernel_ms: float,
    workspace: int,
) -> OperatorPerformanceRecord:
    return OperatorPerformanceRecord(
        workload=wl,
        backend_id=desc.backend_id,
        descriptor_fingerprint=desc.descriptor_fingerprint,
        implementation_fingerprint="impl_v1",
        warmup_iterations=5,
        measured_iterations=10,
        synchronized=True,
        kernel_latency_ms=kernel_ms,
        peak_workspace_bytes=workspace,
        peak_allocated_bytes=workspace,
    )


def test_selects_by_objective_within_memory_budget():
    wl = _workload(ExecutionPhase.TRAIN)
    d_base = _descriptor("ascend_base", workspace_bytes=1024)
    d_fast = _descriptor("ascend_fast", workspace_bytes=4096)
    d_oom = _descriptor("ascend_oom", workspace_bytes=65536)

    catalog = SemanticOperatorCatalog([d_base, d_fast, d_oom])
    store = PerformanceStore(
        operator_records=[
            _rec(wl, d_base, kernel_ms=3.0, workspace=1024),
            _rec(wl, d_fast, kernel_ms=1.2, workspace=4096),
            _rec(wl, d_oom, kernel_ms=0.5, workspace=65536),
        ]
    )
    selector = OperatorSelector(catalog, store)

    fingerprints = {desc.backend_id: "impl_v1" for desc in (d_base, d_fast, d_oom)}
    by_lat = selector.select(
        workload=wl,
        requirements=_reqs(),
        policy=SelectionPolicy(
            phase=ExecutionPhase.TRAIN,
            objective=SelectionObjective.MIN_LATENCY,
            max_workspace_bytes=8192,
            implementation_fingerprints=fingerprints,
        ),
    )
    assert by_lat.selected_backend_id == "ascend_fast"
    assert by_lat.status is SelectionStatus.TUNED

    by_mem = selector.select(
        workload=wl,
        requirements=_reqs(),
        policy=SelectionPolicy(
            phase=ExecutionPhase.TRAIN,
            objective=SelectionObjective.MIN_WORKSPACE,
            max_workspace_bytes=8192,
            implementation_fingerprints=fingerprints,
        ),
    )
    assert by_mem.selected_backend_id == "ascend_base"

    # A measurement from a different build cannot make a candidate fit the budget.
    store.add_operator_record(_rec(wl, d_fast, kernel_ms=0.5, workspace=1024), replace=True)
    stale = selector.select(
        workload=wl,
        requirements=_reqs(),
        policy=SelectionPolicy(
            phase=ExecutionPhase.TRAIN,
            max_workspace_bytes=2048,
            implementation_fingerprints={"ascend_fast": "impl_v2"},
        ),
    )
    assert stale.is_untuned
    assert stale.selected_backend_id == "ascend_base"
    assert stale.eligible_backends == ("ascend_base",)

    with pytest.raises(ValueError, match="workload and requirements"):
        selector.select(
            workload=replace(wl, dtype="float32"),
            requirements=_reqs(),
            policy=SelectionPolicy(phase=ExecutionPhase.TRAIN),
        )


def test_untuned_default_pinned_and_fail_closed_rejection():
    wl = _workload()
    d_ok = _descriptor("ascend_ok", workspace_bytes=1024)
    d_bad = _descriptor("ascend_approx", contract_id="approx.v1", workspace_bytes=512)
    catalog = SemanticOperatorCatalog([d_ok, d_bad])
    selector = OperatorSelector(catalog)

    untuned = selector.select(
        workload=wl,
        requirements=_reqs(),
        policy=SelectionPolicy(phase=ExecutionPhase.PREFILL, max_workspace_bytes=2048),
    )
    assert untuned.selected_backend_id == "ascend_ok"
    assert untuned.is_untuned is True

    pinned = selector.select(
        workload=wl,
        requirements=_reqs(),
        policy=SelectionPolicy(
            phase=ExecutionPhase.PREFILL,
            pinned_backend_id="ascend_ok",
            max_workspace_bytes=2048,
        ),
    )
    assert pinned.status is SelectionStatus.PINNED

    with pytest.raises(OperatorSelectionError):
        selector.select(
            workload=wl,
            requirements=_reqs(),
            policy=SelectionPolicy(
                phase=ExecutionPhase.PREFILL,
                pinned_backend_id="ascend_approx",
            ),
        )


def test_workload_admission_and_unknown_workspace():
    wl = _workload()
    desc = _descriptor("workspace", workspace_bytes=4096)
    missing = replace(_rec(wl, desc, kernel_ms=1.0, workspace=4096), peak_workspace_bytes=None)
    selector = OperatorSelector(
        SemanticOperatorCatalog([desc]), PerformanceStore(operator_records=[missing])
    )
    policy = SelectionPolicy(
        phase=wl.phase,
        max_workspace_bytes=0,
        implementation_fingerprints={desc.backend_id: missing.implementation_fingerprint},
    )
    with pytest.raises(OperatorSelectionError) as invalid:
        selector.select(workload=wl, requirements=_reqs(), policy=policy)
    assert invalid.value.rejected_candidates[0].stage == "memory_budget"

    for unsupported in (
        replace(wl, shape=(96, 4096)),
        replace(wl, layout="column_major"),
        replace(wl, phase=ExecutionPhase.VERIFY),
    ):
        with pytest.raises(OperatorSelectionError) as invalid:
            selector.select(
                workload=unsupported,
                requirements=_reqs(),
                policy=SelectionPolicy(phase=unsupported.phase),
            )
        assert invalid.value.rejected_candidates[0].stage == "workload"
