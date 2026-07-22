# src/utils/profiling.py
"""
Compute cost and latency profiling for a trained model.

Provides ``profile_model``, called once by ``evaluate_model`` after the
real test-set pass, to attach exact per-sample GFLOPs plus latency/
throughput to a run's results. All four model families (MLP, BERT, both
GNNs, the multi-domain geometric model) expose a uniform
``forward_batch(batch, device) -> (logits, targets)``, so this works
generically without any per-family code.

FLOPs are counted via ``torch.utils.flop_counter.FlopCounterMode``, which
intercepts every ATen op dispatched during a single forward call. Its
default table only has formulas for matmul/conv/attention ops: this
module extends it so elementwise geometric math (acos/sin/cos/cosh/sinh/
tanh/exp/clamp, used by the spherical Riemannian encoders and GAT-style
attention) and GNN aggregation (scatter_add_/scatter_reduce_) are counted
too, rather than silently contributing zero. Pure data-movement ops
(indexing, concatenation, reshaping) and comparisons are intentionally
left uncounted: they involve no floating-point arithmetic, so including
them would misrepresent what "FLOPs" means.

Since input shapes are fixed per model+config, the FLOP count for a
forward pass is deterministic: one pass gives an exact count, not a
statistical estimate. Latency/throughput, unlike FLOPs, do have real
run-to-run timing noise, so those are measured over several iterations.

"""

import math
import time
from typing import Any, Dict, cast

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode, flop_registry, register_flop_formula

aten = torch.ops.aten


def _register(targets, formula) -> None:
    """Register `formula` for any of `targets` not already in the global registry.

    Guards against `register_flop_formula`'s "duplicate registration" error,
    which would otherwise fire if this module is imported more than once
    within a single interpreter (e.g. interactive/notebook re-execution).
    """
    targets = targets if isinstance(targets, (list, tuple)) else [targets]
    missing = [t for t in targets if t not in flop_registry]
    if missing:
        register_flop_formula(missing)(formula)


def _elementwise_unary_flop(*args: Any, out_shape: Any = None, **kwargs: Any) -> int:
    """1 flop per output element (standard elementwise-op convention)."""
    return math.prod(out_shape)


def _elementwise_binary_flop(*args: Any, out_shape: Any = None, **kwargs: Any) -> int:
    """1 flop per output element, broadcasting already resolved in out_shape."""
    return math.prod(out_shape)


def _dot_flop(a_shape: Any, b_shape: Any, *args: Any, out_shape: Any = None, **kwargs: Any) -> int:
    """Dot product of two length-n vectors: n multiplies + (n-1) adds ~ 2n."""
    return 2 * math.prod(a_shape)


def _reduction_flop(*args: Any, out_shape: Any = None, **kwargs: Any) -> int:
    """norm/sum/mean: ~1 flop per input element visited by the reduction."""
    return math.prod(args[0])


def _softmax_flop(*args: Any, out_shape: Any = None, **kwargs: Any) -> int:
    """Fused softmax kernel: exp + sum-reduce + div ~ 3 flops per input element."""
    return 3 * math.prod(args[0])


def _scatter_add_flop(
    self_shape: Any, dim: Any, index_shape: Any, src_shape: Any, *args: Any, out_shape: Any = None, **kwargs: Any
) -> int:
    """scatter_add_/scatter_reduce_: one add per scattered (GNN aggregation) element."""
    return math.prod(src_shape)


_register(
    [aten.acos, aten.cos, aten.cosh, aten.sin, aten.sinh, aten.tanh,
     aten.exp, aten.relu, aten.leaky_relu, aten.clamp],
    _elementwise_unary_flop,
)
_register([aten.add, aten.sub, aten.mul, aten.div], _elementwise_binary_flop)
_register(aten.dot, _dot_flop)
_register([aten.linalg_vector_norm, aten.sum, aten.mean], _reduction_flop)
_register(aten._softmax, _softmax_flop)
_register([aten.scatter_add_, aten.scatter_reduce_], _scatter_add_flop)


def profile_model(
    model: nn.Module,
    batch: Any,
    device: str,
    n_warmup: int = 5,
    n_iters: int = 20,
) -> Dict[str, Any]:
    """
    Measure exact per-sample GFLOPs and latency/throughput for one model.

    Parameters
    ----------
    model : nn.Module
        Trained model, already moved to `device` and in eval mode by the
        caller (evaluate_model).
    batch : Any
        A single batch as produced by the model's test DataLoader, in
        whatever container type its forward_batch expects (tuple/dict/
        PyG Data/HeteroData).
    device : str
        Target device string (e.g. 'cuda', 'cpu').
    n_warmup : int
        Untimed forward_batch calls before timing starts, to absorb GPU/
        cuDNN warmup effects.
    n_iters : int
        Timed forward_batch calls averaged for the latency/throughput
        estimate. FLOPs, unlike latency, don't need multiple iterations
        (fixed input shapes -> deterministic op count).

    Returns
    -------
    dict with keys:
        gflops_per_sample, gflops_per_batch, profiled_batch_size,
        latency_ms_per_batch, throughput_samples_per_sec
    """
    model.eval()
    is_cuda = device.startswith("cuda")
    dynamic_model = cast(Any, model)

    with torch.no_grad():
        _, targets = dynamic_model.forward_batch(batch, device)
        for _ in range(max(n_warmup - 1, 0)):
            dynamic_model.forward_batch(batch, device)
        batch_size = int(targets.shape[0])

        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            dynamic_model.forward_batch(batch, device)
        total_flops = flop_counter.get_total_flops()

        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            dynamic_model.forward_batch(batch, device)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed_sec = time.perf_counter() - start

    latency_ms = (elapsed_sec / n_iters) * 1000.0
    throughput = (batch_size * n_iters) / elapsed_sec

    return {
        "gflops_per_sample": total_flops / batch_size / 1e9,
        "gflops_per_batch": total_flops / 1e9,
        "profiled_batch_size": batch_size,
        "latency_ms_per_batch": latency_ms,
        "throughput_samples_per_sec": throughput,
    }