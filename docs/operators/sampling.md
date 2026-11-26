# Sampling

The sampling backend provides a unified interface for rollout token sampling. It routes
to optimized vendor libraries when available and falls back to PyTorch sampling logic.

## Entry Point

```python
from rl_engine.ops.sampling.api import SamplerBackend

sampler = SamplerBackend()
tokens = sampler.sample(logits, top_k=50, top_p=0.95, temperature=1.0)
```

## Backends

| Platform | Backend | Status |
| --- | --- | --- |
| NVIDIA CUDA | FlashInfer | Active |
| AMD ROCm / Moore Threads MUSA | PyTorch | Portable fallback |
| Ascend NPU | PyTorch on CANN | Registered candidate |
| CPU / fallback | PyTorch | Active fallback |

## Supported Modes

- Unfiltered sampling from logits.
- Top-k sampling.
- Top-p sampling.
- Combined top-k and top-p sampling with a shared probability transform.
- Deterministic sampling when supported by the backend.

Pass the same `SamplingPolicyConfig` to rollout sampling and
`transformed_sampling_logprobs` for differentiable training recomputation.
The policy applies temperature, top-k and top-p within `real_vocab_size`.
Greedy rows assign all probability to the highest-scoring real token.
A policy call draws once per row, including rows with a point-mass distribution.
An explicit generator takes precedence over a seed. A seed initializes a new
generator on the logits device for each call. Build per-row policy tensors
before sampling and reuse the validated configuration across calls.

CUDA calls using the original keyword arguments and scalar temperature use
FlashInfer filtering and report their vendor probability definition. Pass an
explicit policy to share the probability definition with training.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `logits` | `[B, V]` or compatible backend shape | Floating point | Converted to contiguous layout. |
| Output | `[B]` | Integer token IDs | One sampled token per row. |

## Tests and Benchmarks

```bash
python benchmarks/operators/sampling/benchmark_sampling.py
```

Add focused tests when changing backend routing, supported sampling modes, or numerical
behavior.

## Implementation Files

- `rl_engine/ops/sampling/api.py`
- `rl_engine/platforms/constants.py`
