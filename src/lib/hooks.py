"""PyTorch forward hooks for residual-stream capture.

Strategy: attach a single forward hook on each transformer block. The
hook captures the BLOCK OUTPUT (which is the residual-stream value
after that block writes into it), CPU-offloads it, and stores it in a
caller-supplied dict keyed by layer index.

Why we capture block outputs (not block inputs): the residual stream is
the running sum that each block reads from and writes back to. After
block N's residual addition, the stream represents "the model's state
after layer N has had its say". That's the right granularity for
layer-by-layer probes.

Why CPU-offload immediately: storing every layer's activations on GPU
across hundreds of inputs blows past 8 GB on the 3060 Ti. Each tensor
takes only the GPU-time of one forward pass, so the per-input cost is
mostly bandwidth, not compute.

We deliberately don't use TransformerLens — its Qwen 2.5 support has
been spotty and we want a thin, dependable layer over HuggingFace
transformers. ~50 LOC of hooks is more reliable than a third-party
abstraction in flux.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


def _block_list(model: nn.Module) -> list[nn.Module]:
    """Return the ordered list of transformer blocks for common architectures.

    Qwen 2.5 (and most Llama-family models) expose blocks as
    `model.model.layers[i]`. Fall back to a few alternative paths so
    this lib doesn't tightly couple to one HF class name.
    """
    # Order matters: try AutoModel-flat first (Qwen2Model.layers), then
    # AutoModelForCausalLM-nested (model.model.layers), then GPT-style.
    candidates = [
        ("layers",),                      # Qwen2Model, LlamaModel from AutoModel
        ("model", "layers"),              # AutoModelForCausalLM wrapping the above
        ("model", "model", "layers"),     # legacy path some HF versions used
        ("transformer", "h"),             # GPT-2/Neo style
    ]
    for path in candidates:
        cur = model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok and isinstance(cur, (nn.ModuleList, list)):
            return list(cur)
    raise RuntimeError(
        "Could not locate transformer block list. "
        "Tried: model.model.layers, model.transformer.h, model.layers."
    )


@contextmanager
def capture_residuals(
    model: nn.Module,
    out: dict[int, torch.Tensor],
) -> Iterator[None]:
    """Context manager: install hooks, yield, remove hooks.

    Usage:
        out: dict[int, torch.Tensor] = {}
        with capture_residuals(model, out):
            model(**inputs)
        # `out` now has one CPU tensor per layer, shape [batch, seq, d_model]

    Hooks are removed on exit even if the wrapped block raises. The
    output dict is populated in-place; callers can mutate it across
    multiple forward passes by clearing between calls.
    """
    blocks = _block_list(model)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_idx, block in enumerate(blocks):
        # Closure-capture layer_idx; default-arg trick avoids the
        # late-binding gotcha in for-loops.
        def _make_hook(li: int = layer_idx):
            def hook(_module, _args, output):
                # HuggingFace block outputs are typically (hidden, ...)
                # tuples; the first element is the residual stream.
                tensor = output[0] if isinstance(output, tuple) else output
                # Detach so we don't pin the autograd graph; CPU-offload
                # so 28 layers x batch x seq x d_model doesn't OOM the GPU.
                out[li] = tensor.detach().to("cpu", non_blocking=True)

            return hook

        handles.append(block.register_forward_hook(_make_hook()))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


def num_layers(model: nn.Module) -> int:
    """Number of transformer blocks, useful for sizing pre-allocated output."""
    return len(_block_list(model))
