"""Activation patching at the LAST-TOKEN POSITION per layer.

Method (per-pair, per-layer):

  1. Run clean(A) on text_a → capture each block's output residual and
     the final-token logits.
  2. Run clean(B) on text_b → same.
  3. For each layer L, run text_a again. A hook on block L replaces
     only the LAST-TOKEN position in the residual stream with the
     corresponding row from clean(B); other positions are left alone.
     Read off the final-token logits.
  4. Compute the "transfer fraction" at layer L:

        T(L) = 1 - KL( p_a_patched(L) || p_b ) / KL( p_a || p_b )

     T(L) = 0 means patching at layer L did nothing; A's predictions
     are unchanged. T(L) = 1 means patching at layer L fully shifted
     A's predictions onto B's. T < 0 means patching pushed A FURTHER
     from B than the clean run was.

Why last-token only and not full-residual: replacing the ENTIRE
layer-L residual is equivalent to "from here on, run as if input was
B", so every layer trivially gives T=1.0. The interesting question
is not "can we replace everything" but "where in the stack does the
prediction anchor (last token) carry the dialect signal?". Last-token
patching answers that.

The flat-across-layers linear probe and the negative head-ablation
result both predicted distributed representation; activation patching
at the last-token position is the gold-standard mech-interp test of
that prediction.

Restrictions:
  - We only operate on pairs where the BM and NN tokenizations have the
    same length (so layer outputs have the same shape and direct slot-
    by-slot replacement is sound). On D1 (Apertium nob->nno) this is
    typically the majority.

Output:
  paper/figures/08_activation_patching.png   per-layer transfer curve
  runs/probes/activation_patching.csv        raw per-pair-layer numbers

Usage:

    python src/08_activation_patching.py [--limit N] [--contrast d1|d2|d3]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.config import (
    DTYPE_NAME,
    MAX_TOKENS,
    MODEL_ID,
    REPO_ROOT,
    contrast_jsonl_path,
)
from lib.eval_set import load_pairs
from lib.hooks import _block_list


def _resolve_dtype(name: str) -> torch.dtype:
    table = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    return table.get(name, torch.float16)


@torch.no_grad()
def run_clean_capture(
    model,
    tokenizer,
    text: str,
    device: str,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Forward pass on `text`. Returns (final-token logits, layer outputs).

    `layer_outputs[i]` is the residual stream AFTER block i, on GPU,
    shape [1, seq_len, d_model]. Captured via forward hooks on each
    block — NOT via output_hidden_states, because HF's hidden_states[-1]
    is the post-final-norm output rather than the last block's raw
    output, which would double-normalize when we patch later.
    """
    blocks = _block_list(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(li: int):
        def hook(_module, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[li] = h.detach()  # GPU
        return hook

    for li, block in enumerate(blocks):
        handles.append(block.register_forward_hook(make_hook(li)))

    try:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            padding=False,
        )
        input_ids = enc["input_ids"].to(device)
        out = model(input_ids=input_ids, return_dict=True)
        logits = out.logits[0, -1, :].to(torch.float32)
    finally:
        for h in handles:
            h.remove()

    return logits, captured


@torch.no_grad()
def run_patched_lasttoken(
    model,
    tokenizer,
    text: str,
    layer_idx: int,
    replacement_last_token: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Run `text` through model, but at block `layer_idx`, replace
    ONLY the last-token position of the residual stream with
    `replacement_last_token` (shape [1, d_model]).

    The hook receives the block's output [1, S, d_model], copies it,
    overwrites position [-1] with the replacement vector, and returns
    the modified tensor. Earlier positions retain A's processing.
    """
    block = _block_list(model)[layer_idx]

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, -1, :] = replacement_last_token  # broadcast [1, d] -> [1, d]
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    handle = block.register_forward_hook(hook)
    try:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            padding=False,
        )
        input_ids = enc["input_ids"].to(device)
        out = model(input_ids=input_ids, return_dict=True)
        return out.logits[0, -1, :].to(torch.float32)
    finally:
        handle.remove()


def kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """KL(p || q) in nats, on softmax distributions."""
    return float((p * (p.add(eps).log() - q.add(eps).log())).sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--contrast", default="d1")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(DTYPE_NAME)

    pairs_all = load_pairs(contrast_jsonl_path(args.contrast))
    print(f"[08] loading {MODEL_ID} (CausalLM)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model.to(device)
    model.eval()

    n_layers = len(_block_list(model))
    print(f"[08] n_layers={n_layers}")

    # Last-token patching only needs B's last-token residual; the per-
    # token shapes of A and B don't have to match. We keep all pairs.
    pairs = pairs_all[: args.limit]
    print(f"[08] using {len(pairs)} pairs for last-token patching")

    # Per-layer transfer accumulator
    transfer_per_layer = np.zeros((len(pairs), n_layers), dtype=np.float64)
    kl_baseline = np.zeros(len(pairs), dtype=np.float64)

    rows: list[dict] = []

    for pi, pair in enumerate(tqdm(pairs, desc="patch")):
        # Clean runs
        logits_a, layers_a = run_clean_capture(model, tokenizer, pair["text_a"], device)
        logits_b, layers_b = run_clean_capture(model, tokenizer, pair["text_b"], device)
        p_a = torch.softmax(logits_a, dim=-1)
        p_b = torch.softmax(logits_b, dim=-1)
        kl_ab = kl_div(p_a, p_b)
        kl_baseline[pi] = kl_ab
        if kl_ab < 1e-8:
            # Predictions identical even on different inputs; transfer
            # is undefined. Mark row for filtering.
            transfer_per_layer[pi, :] = np.nan
            continue

        # Patch each layer L's LAST-TOKEN POSITION from B into A.
        # Only the last-token slice of B is needed; B's full sequence
        # length doesn't have to match A's.
        for li in range(n_layers):
            replacement_last = layers_b[li][:, -1, :]  # [1, d]
            logits_patched = run_patched_lasttoken(
                model, tokenizer, pair["text_a"], li, replacement_last, device,
            )
            p_patched = torch.softmax(logits_patched, dim=-1)
            kl_patched_b = kl_div(p_patched, p_b)
            transfer = 1.0 - (kl_patched_b / kl_ab)
            transfer_per_layer[pi, li] = transfer
            rows.append({
                "contrast": args.contrast,
                "pair_id": pair["id"],
                "layer": li,
                "kl_baseline": kl_ab,
                "kl_patched_b": kl_patched_b,
                "transfer": transfer,
            })

        # Free GPU memory between pairs
        del layers_a, layers_b
        if device == "cuda":
            torch.cuda.empty_cache()

    # Aggregate: mean transfer per layer (ignoring NaNs)
    layer_means = np.nanmean(transfer_per_layer, axis=0)
    layer_stds = np.nanstd(transfer_per_layer, axis=0)

    out_dir = REPO_ROOT / "runs" / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = REPO_ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"activation_patching_{args.contrast}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "contrast", "pair_id", "layer",
                "kl_baseline", "kl_patched_b", "transfer",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[08] csv -> {csv_path}")

    json_path = out_dir / f"activation_patching_{args.contrast}.json"
    json_path.write_text(
        json.dumps(
            {
                "contrast": args.contrast,
                "n_pairs": len(pairs),
                "n_layers": int(n_layers),
                "kl_baseline_mean": float(np.nanmean(kl_baseline)),
                "transfer_per_layer_mean": layer_means.tolist(),
                "transfer_per_layer_std": layer_stds.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Print summary
    print(f"[08] mean baseline KL(A||B): {np.nanmean(kl_baseline):.3f} nats")
    print("[08] transfer per layer (mean across pairs):")
    for li in range(n_layers):
        marker = ""
        if layer_means[li] > 0.5:
            marker = " <- high transfer"
        elif layer_means[li] < 0:
            marker = " <- negative (push away)"
        print(f"     L{li:>2}: {layer_means[li]:+.3f} +- {layer_stds[li]:.3f}{marker}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    layers = np.arange(n_layers)
    ax.plot(layers, layer_means, "-o", markersize=4, color="#d62728", label="mean transfer")
    ax.fill_between(
        layers,
        layer_means - layer_stds,
        layer_means + layer_stds,
        alpha=0.2,
        color="#d62728",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1, color="black", linewidth=0.5, linestyle=":")
    ax.text(0.5, 1.02, "full transfer", fontsize=8, color="black", alpha=0.6)
    ax.text(0.5, 0.02, "no transfer", fontsize=8, color="black", alpha=0.6)
    ax.set_xlabel("Layer at which we patched A's residual with B's")
    ax.set_ylabel(r"transfer fraction $1 - \mathrm{KL}(p_{a\,patched} \| p_b) / \mathrm{KL}(p_a \| p_b)$")
    ax.set_title(
        f"DIA-LOC: activation patching, contrast={args.contrast.upper()}\n"
        f"n_pairs={len(pairs)} ({MODEL_ID})"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = fig_dir / f"08_activation_patching_{args.contrast}.png"
    fig.savefig(png_path, dpi=150)
    print(f"[08] figure -> {png_path}")


if __name__ == "__main__":
    main()
