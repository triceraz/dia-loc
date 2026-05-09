"""Attention-head ablation: which heads carry the dialectal signal?

For each (layer L, head H) in Qwen 2.5 1.5B (28 x 12 = 336 heads):
  1. Install a forward-pre-hook on layer L's o_proj that zeros the
     head_dim slice belonging to head H before output projection.
  2. Run a subset of D1 paired inputs through the model.
  3. Mean-pool the FINAL-layer residual.
  4. Train a 5-fold linear probe on those activations.
  5. Record probe accuracy.

Heads where ablation drops the BM/NN probe accuracy noticeably are the
heads that carry the dialect signal. Heads where it doesn't are
load-bearing for other reasons.

This is the "where is dialect represented?" probe at head granularity.
The linear probe (05) showed ~0.80 accuracy across all layers; this
script asks which specific heads contribute to that accuracy.

Compute: 336 ablations x N pairs x 2 sides forwards. With N=30 (default)
that's ~20k forwards, ~25 min on a 3060 Ti.

Output:
  paper/figures/06_head_ablation.png   (28xH heatmap of probe accuracy)
  runs/probes/head_ablation.csv

Run:

    python src/06_head_ablation.py [--limit N] [--contrast d1|d2|d3]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lib.config import (
    DTYPE_NAME,
    MAX_TOKENS,
    MODEL_ID,
    REPO_ROOT,
    activations_dir,
    contrast_jsonl_path,
)
from lib.eval_set import load_pairs
from lib.hooks import _block_list


def _resolve_dtype(name: str) -> torch.dtype:
    table = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    return table.get(name, torch.float16)


def _attn_o_proj(model, layer_idx: int) -> torch.nn.Module:
    """Locate the o_proj sub-module for a given layer. Qwen 2.5 path:
    model.layers[i].self_attn.o_proj."""
    block = _block_list(model)[layer_idx]
    return block.self_attn.o_proj


@contextmanager
def ablate_head(model, layer_idx: int, head_idx: int, head_dim: int):
    """Zero head `head_idx`'s contribution to o_proj's input on layer `layer_idx`.

    o_proj receives the concatenation of all heads' outputs along the
    last dim, shape [B, S, H * head_dim]. We zero the slice belonging
    to `head_idx` before the projection runs.
    """
    o_proj = _attn_o_proj(model, layer_idx)
    start = head_idx * head_dim
    end = start + head_dim

    def pre_hook(module, args):
        (x,) = args
        x = x.clone()
        x[..., start:end] = 0
        return (x,)

    handle = o_proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def capture_final_residual(
    model,
    tokenizer,
    texts: list[str],
    device: str,
) -> torch.Tensor:
    """Forward each text, capture mean-pooled residual at the FINAL layer.

    Returns shape [n_texts, d_model] in fp16 on CPU.
    """
    n_layers = len(_block_list(model))
    final_layer_idx = n_layers - 1

    out_holder: dict[int, torch.Tensor] = {}

    def final_hook(module, _args, output):
        # Block output is (hidden, ...) tuple; first element is residual.
        out_holder["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = _block_list(model)[final_layer_idx].register_forward_hook(final_hook)

    rows: list[torch.Tensor] = []
    try:
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKENS,
                padding=False,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            out_holder.clear()
            model(input_ids=input_ids, attention_mask=attention_mask)

            h = out_holder["h"].to(torch.float32)  # [1, S, D]
            mask = attention_mask.to(torch.float32).unsqueeze(-1)  # [1, S, 1]
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [1, D]
            rows.append(pooled.to("cpu").to(torch.float16))
    finally:
        handle.remove()

    return torch.cat(rows, dim=0)


def probe_accuracy(a: np.ndarray, b: np.ndarray, n_folds: int = 5, seed: int = 0) -> float:
    X = np.concatenate([a, b], axis=0)
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))]).astype(np.int64)
    accs: list[float] = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
        clf.fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return float(np.mean(accs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--contrast", default="d1")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(DTYPE_NAME)

    # Load contrast
    pairs = load_pairs(contrast_jsonl_path(args.contrast))[: args.limit]
    if not pairs:
        sys.exit(f"no pairs for contrast {args.contrast}")
    texts_a = [p["text_a"] for p in pairs]
    texts_b = [p["text_b"] for p in pairs]
    print(
        f"[06] contrast={args.contrast}  n_pairs={len(pairs)}  device={device}"
    )

    print(f"[06] loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(MODEL_ID, dtype=dtype)
    model.to(device)
    model.eval()

    n_layers = len(_block_list(model))
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    print(f"[06] n_layers={n_layers} n_heads={n_heads} head_dim={head_dim}")

    # Baseline (no ablation) probe accuracy
    print("[06] baseline forward pass ...")
    a_base = capture_final_residual(model, tokenizer, texts_a, device).numpy().astype(np.float32)
    b_base = capture_final_residual(model, tokenizer, texts_b, device).numpy().astype(np.float32)
    baseline_acc = probe_accuracy(a_base, b_base)
    print(f"[06] baseline final-layer probe accuracy: {baseline_acc:.3f}")

    # Per (layer, head) ablation
    out_dir = REPO_ROOT / "runs" / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = REPO_ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.zeros((n_layers, n_heads), dtype=np.float32)
    rows: list[dict] = []

    total = n_layers * n_heads
    pbar = tqdm(total=total, desc="ablate")
    for li in range(n_layers):
        for hi in range(n_heads):
            with ablate_head(model, li, hi, head_dim):
                a = capture_final_residual(model, tokenizer, texts_a, device).numpy().astype(np.float32)
                b = capture_final_residual(model, tokenizer, texts_b, device).numpy().astype(np.float32)
            acc = probe_accuracy(a, b)
            matrix[li, hi] = acc
            rows.append({
                "contrast": args.contrast,
                "layer": li,
                "head": hi,
                "n_pairs": len(pairs),
                "ablated_probe_acc": acc,
                "baseline_probe_acc": baseline_acc,
                "delta": acc - baseline_acc,
            })
            pbar.update(1)
    pbar.close()

    csv_path = out_dir / f"head_ablation_{args.contrast}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "contrast", "layer", "head", "n_pairs",
                "ablated_probe_acc", "baseline_probe_acc", "delta",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[06] csv -> {csv_path}")

    json_path = out_dir / f"head_ablation_{args.contrast}.json"
    json_path.write_text(
        json.dumps(
            {
                "contrast": args.contrast,
                "n_pairs": len(pairs),
                "n_layers": n_layers,
                "n_heads": n_heads,
                "baseline_probe_acc": baseline_acc,
                "matrix": matrix.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Plot heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    delta = matrix - baseline_acc
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(
        delta,
        aspect="auto",
        origin="lower",
        cmap="RdBu",
        vmin=-abs(delta).max(),
        vmax=abs(delta).max(),
    )
    cbar = plt.colorbar(im, ax=ax, label="probe acc delta vs baseline")
    cbar.ax.tick_params(labelsize=8)
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(0, n_layers, 4))
    ax.set_title(
        f"Head ablation effect on {args.contrast.upper()} linear probe\n"
        f"baseline acc = {baseline_acc:.3f}  ({MODEL_ID})\n"
        "Blue = ablating this head DROPS the probe (head carries dialect signal)"
    )
    fig.tight_layout()
    png_path = fig_dir / f"06_head_ablation_{args.contrast}.png"
    fig.savefig(png_path, dpi=150)
    print(f"[06] figure -> {png_path}")

    # Top-5 dialect-carrying heads
    flat = [(li, hi, delta[li, hi]) for li in range(n_layers) for hi in range(n_heads)]
    flat.sort(key=lambda x: x[2])
    print(f"[06] top-5 heads where ablation hurts {args.contrast} most:")
    for li, hi, d in flat[:5]:
        print(f"     L{li:>2} H{hi:>2}: delta={d:+.3f}")


if __name__ == "__main__":
    main()
