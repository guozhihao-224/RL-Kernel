# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Selection cache with version, shape-bucket, and MoE token-count keys (#563 Task 34)."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from rl_engine.runtime.performance.schema import OperatorWorkloadKey
from rl_engine.runtime.selector.policy import (
    OperatorSelector,
    SelectionDecision,
    SelectionObjective,
    SelectionPolicy,
    _check_budget,
    _workload_rejection,
)
from rl_engine.runtime.semantic_registry import (
    OperatorRequirements,
    _fingerprint,
    _JsonRecord,
)

SELECTION_CACHE_KEY_SCHEMA_VERSION = "rlkernel.runtime.selector.cache_key.v1"
CACHED_SELECTION_SCHEMA_VERSION = "rlkernel.runtime.selector.cached_selection.v1"


@dataclass(frozen=True)
class ShapeBucketSpec(_JsonRecord):
    """Validated bucket for a range of tensor shapes sharing one representative key."""

    bucket_id: str
    representative_shape: tuple[int, ...]
    validated_shapes: frozenset[tuple[int, ...]]

    def __post_init__(self) -> None:
        shapes = frozenset(tuple(shape) for shape in self.validated_shapes)
        if self.representative_shape not in shapes or any(
            len(shape) != len(self.representative_shape) for shape in shapes
        ):
            raise ValueError("bucket shapes must include the representative and have equal rank")
        object.__setattr__(self, "validated_shapes", shapes)


@dataclass(frozen=True)
class SelectionCacheKey(_JsonRecord):
    """Complete cache key composing OperatorWorkloadKey with candidate and MoE state."""

    workload: OperatorWorkloadKey
    # Order also controls the default when the candidates are untuned.
    candidate_fingerprints: tuple[tuple[str, str | None, str | None], ...]
    policy_fingerprint: str
    bucket_id: str | None = None
    schema_version: str = SELECTION_CACHE_KEY_SCHEMA_VERSION

    @property
    def cache_key_fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class CachedSelection(_JsonRecord):
    """Selection decision paired with its cache key and hit/miss provenance."""

    cache_key: SelectionCacheKey
    decision: SelectionDecision
    cache_hit: bool
    cache_reason: str
    requested_shape: tuple[int, ...]
    schema_version: str = CACHED_SELECTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["cache_key_fingerprint"] = self.cache_key.cache_key_fingerprint
        return result


class SelectionCache:
    """Cache operator selections with automatic fingerprint and constraint invalidation."""

    def __init__(
        self,
        selector: OperatorSelector,
        *,
        buckets: Iterable[tuple[str, ShapeBucketSpec]] = (),
        max_entries: int = 128,
    ) -> None:
        if max_entries < 1:
            raise ValueError("selection cache capacity must be positive")
        self.selector = selector
        self._max_entries = max_entries
        self._buckets: dict[str, list[ShapeBucketSpec]] = {}
        self._entries: OrderedDict[str, tuple[SelectionCacheKey, SelectionDecision]] = OrderedDict()
        for semantic_op, bucket in buckets:
            self.register_bucket(semantic_op, bucket)

    def register_bucket(self, semantic_op: str, bucket: ShapeBucketSpec) -> None:
        op_name = semantic_op.strip()
        self._buckets.setdefault(op_name, []).append(bucket)
        self.invalidate(semantic_op=op_name)

    def _resolve_bucket(self, semantic_op: str, shape: tuple[int, ...]) -> ShapeBucketSpec | None:
        return next(
            (b for b in self._buckets.get(semantic_op, ()) if shape in b.validated_shapes),
            None,
        )

    def select(
        self,
        *,
        workload: OperatorWorkloadKey,
        requirements: OperatorRequirements,
        policy: SelectionPolicy,
        candidate_backends: Sequence[str] | None = None,
        expert_token_counts: Sequence[int] | None = None,
        speculative_draft_len: int | None = None,
    ) -> CachedSelection:
        if expert_token_counts is not None:
            workload = replace(workload, expert_token_counts=tuple(expert_token_counts))
        if speculative_draft_len is not None:
            workload = replace(workload, speculative_draft_len=speculative_draft_len)

        profiles = self.selector._profiles(workload, policy, candidate_backends)
        effective_workload = workload
        bucket = self._resolve_bucket(workload.semantic_op, workload.shape)
        if bucket is not None and workload.shape != bucket.representative_shape:
            representative = replace(workload, shape=bucket.representative_shape)
            if policy.objective is SelectionObjective.MIN_WORKSPACE or any(
                record is not None
                or (
                    desc is not None
                    and (
                        (_workload_rejection(workload, desc) is None)
                        != (_workload_rejection(representative, desc) is None)
                    )
                )
                for _, desc, record in profiles
            ):
                bucket = None
            else:
                backend_ids = tuple(profile.backend_id for profile in profiles)
                samples = self.selector._profiles(representative, policy, backend_ids)
                budget = policy.max_workspace_bytes
                if any(
                    desc is not sample_desc
                    or (
                        desc is not None
                        and (_check_budget(budget, desc, record) is None)
                        != (_check_budget(budget, desc, sample) is None)
                    )
                    for (_, desc, record), (_, sample_desc, sample) in zip(
                        profiles, samples, strict=True
                    )
                ):
                    bucket = None
                else:
                    effective_workload, profiles = representative, samples

        key = SelectionCacheKey(
            workload=effective_workload,
            candidate_fingerprints=tuple(
                (
                    backend_id,
                    None if desc is None else desc.descriptor_fingerprint,
                    None if record is None else record.record_fingerprint,
                )
                for backend_id, desc, record in profiles
            ),
            policy_fingerprint=_fingerprint(
                {"policy": policy.to_dict(), "requirements": requirements.to_dict()}
            ),
            bucket_id=None if bucket is None else bucket.bucket_id,
        )
        fingerprint = key.cache_key_fingerprint
        entry = self._entries.pop(fingerprint, None)
        if entry is None:
            decision = self.selector._select(effective_workload, requirements, policy, profiles)
        else:
            _, decision = entry
        self._entries[fingerprint] = (key, decision)
        if len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

        reason = "cache_miss" if entry is None else "cache_hit"
        label = "exact" if bucket is None else f"bucket={bucket.bucket_id}"
        return CachedSelection(
            key, decision, entry is not None, f"{reason}:{label}", workload.shape
        )

    def invalidate(
        self,
        *,
        semantic_op: str | None = None,
        device_model: str | None = None,
        software_version: str | None = None,
    ) -> int:
        to_remove = [
            fp
            for fp, (k, _) in self._entries.items()
            if (semantic_op is None or k.workload.semantic_op == semantic_op.strip())
            and (device_model is None or k.workload.device_model == device_model.strip())
            and (
                software_version is None or k.workload.software_version == software_version.strip()
            )
        ]
        for fp in to_remove:
            self._entries.pop(fp, None)
        return len(to_remove)

    def __len__(self) -> int:
        return len(self._entries)


__all__ = [
    "CACHED_SELECTION_SCHEMA_VERSION",
    "CachedSelection",
    "SELECTION_CACHE_KEY_SCHEMA_VERSION",
    "SelectionCache",
    "SelectionCacheKey",
    "ShapeBucketSpec",
]
