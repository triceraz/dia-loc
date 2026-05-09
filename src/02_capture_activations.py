"""Capture residual-stream activations on every contrast pair.

For each pair (text_a, text_b) in D1 / D2 / D3:
  1. Tokenize both sides separately, padded to the same length so we
     can stack later for batched probes
  2. Forward each through the model with hooks installed (lib/hooks.py)
  3. Pool the residual stream per layer with mean-over-tokens (excluding
     padding) so each input collapses to a single d_model vector per
     layer. Mean-pooling is the standard lightweight summary for
     sentence-level analysis; the alternative of last-token only would
     entangle our signal with end-of-sentence-specific structure.
  4. Save as a single torch tensor per (contrast, side) of shape
     [n_pairs, n_layers, d_model] in fp16

Per-input wall-clock on a 3060 Ti for Qwen 2.5 1.5B at fp16:
roughly 100-200 ms total including disk write. 500 inputs total
(D1 + D2 + (eventual) D3) lands in the low minutes.

Run:

    python src/02_capture_activations.py [--limit N]

`--limit N` truncates each contrast to the first N pairs; useful for a
fast smoke test before committing to the full run.

Output:

    runs/activations/<checkpoint_slug>/d1_a.pt  shape [n,L,d]
    runs/activations/<checkpoint_slug>/d1_b.pt
    runs/activations/<checkpoint_slug>/d2_a.pt
    runs/activations/<checkpoint_slug>/d2_b.pt
    runs/activations/<checkpoint_slug>/d3_a.pt  (if D3 present)
    runs/activations/<checkpoint_slug>/d3_b.pt
    runs/activations/<checkpoint_slug>/manifest.json

The manifest records contrast-set order, model id, dtype, n_layers,
d_model, and the original pair ids in the same order as rows in the
tensors. Downstream scripts read manifest.json before opening tensors
so they can validate alignment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure we can `from lib.foo import ...` no matter what cwd this is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lib.config import (
    CONTRAST_SETS,
    DTYPE_NAME,
    MAX_TOKENS,
    MODEL_ID,
    activations_dir,
    contrast_jsonl_path,
)
from lib.eval_set import load_pairs
from lib.hooks import capture_residuals, num_layers


def _resolve_dtype(name: str) -> torch.dtype:
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"unknown dtype: {name}")
    return table[name]


def _pool_layer_outputs(
    layer_outputs: dict[int, torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool each layer's [batch, seq, d] tensor over non-pad tokens.

    Returns shape [batch, n_layers, d] in fp16. Mask comes from the
    tokenizer (1 = real token, 0 = pad) so the pooled vector reflects
    only real content.
    """
    n_layers = len(layer_outputs)
    # CPU tensors, since the hooks moved them off-GPU.
    mask = attention_mask.to(torch.float32).unsqueeze(-1)  # [B, S, 1]
    mask_sum = mask.sum(dim=1).clamp(min=1.0)  # [B, 1] avoids div-by-zero

    pooled_per_layer: list[torch.Tensor] = []
    for li in range(n_layers):
        h = layer_outputs[li].to(torch.float32)  # [B, S, d]
        pooled = (h * mask).sum(dim=1) / mask_sum  # [B, d]
        pooled_per_layer.append(pooled)
    # Stack along a new layer axis -> [B, L, d]
    return torch.stack(pooled_per_layer, dim=1).to(torch.float16)


@torch.no_grad()
def capture_one_side(
    model,
    tokenizer,
    texts: list[str],
    device: str,
) -> torch.Tensor:
    """Run all `texts` through the model, capture mean-pooled residuals.

    Returns a single tensor of shape [n_texts, n_layers, d_model] in fp16
    on CPU.
    """
    n_layers = num_layers(model)
    pooled_rows: list[torch.Tensor] = []

    for text in tqdm(texts, desc="forward", leave=False):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            padding=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out: dict[int, torch.Tensor] = {}
        with capture_residuals(model, out):
            model(input_ids=input_ids, attention_mask=attention_mask)

        if len(out) != n_layers:
            raise RuntimeError(
                f"hook captured {len(out)} layers, expected {n_layers}"
            )

        # Pool on CPU using the cpu-offloaded activations from the hook.
        pooled = _pool_layer_outputs(out, attention_mask.to("cpu"))  # [1, L, d]
        pooled_rows.append(pooled)

    # Concatenate along the batch axis. Each row is [1, L, d]; stacking
    # gives [n_texts, L, d].
    return torch.cat(pooled_rows, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate each contrast set to the first N pairs (smoke test).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda / cpu / auto (default: auto-detect cuda)",
    )
    parser.add_argument(
        "--contrasts",
        default=None,
        help=(
            "Comma-separated contrast slugs to capture (e.g. 'd3' or 'd1,d3'). "
            "Default: all known contrasts. Useful for adding a new contrast "
            "to existing activations without re-doing the others."
        ),
    )
    args = parser.parse_args()

    if args.contrasts:
        wanted = {s.strip() for s in args.contrasts.split(",") if s.strip()}
    else:
        wanted = None

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(DTYPE_NAME)

    out_dir = activations_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[02] model={MODEL_ID}  device={device}  dtype={DTYPE_NAME}")
    print(f"[02] writing to {out_dir}")

    print("[02] loading tokenizer + model ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        # Qwen is fine, but some models need this. Reuse EOS as pad to
        # avoid emitting a brand-new token id that the embedding hasn't
        # seen.
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    )
    model.to(device)
    model.eval()
    n_layers = num_layers(model)
    d_model = getattr(getattr(model, "config", None), "hidden_size", None)
    print(
        f"[02] model loaded in {time.time() - t0:.1f}s, "
        f"n_layers={n_layers}, d_model={d_model}"
    )

    # Read existing manifest if present, so a partial re-run (e.g.
    # `--contrasts d3` after D1+D2 are already captured) merges into the
    # existing record rather than dropping prior contrasts on the floor.
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    else:
        manifest = {}

    manifest.update({
        "model_id": MODEL_ID,
        "dtype": DTYPE_NAME,
        "n_layers": n_layers,
        "d_model": d_model,
        "device": device,
        "max_tokens": MAX_TOKENS,
    })
    manifest.setdefault("contrasts", [])

    # Index existing manifest entries by slug so we can replace one
    # contrast's entry without disturbing the others.
    by_slug: dict[str, dict] = {
        c["slug"]: c for c in manifest.get("contrasts", []) if isinstance(c, dict)
    }

    for slug, lang_a, lang_b, fname, label in CONTRAST_SETS:
        if wanted is not None and slug not in wanted:
            continue
        in_path = contrast_jsonl_path(slug)
        if not in_path.exists():
            print(f"[02] {slug}: input missing ({in_path}), skipping")
            continue
        pairs = load_pairs(in_path)
        if args.limit:
            pairs = pairs[: args.limit]
        if not pairs:
            print(f"[02] {slug}: 0 pairs, skipping")
            continue

        ids = [p["id"] for p in pairs]
        texts_a = [p["text_a"] for p in pairs]
        texts_b = [p["text_b"] for p in pairs]

        print(f"[02] {slug} ({label}): {len(pairs)} pairs")
        t1 = time.time()
        acts_a = capture_one_side(model, tokenizer, texts_a, device)
        torch.save(acts_a, out_dir / f"{slug}_a.pt")
        del acts_a

        acts_b = capture_one_side(model, tokenizer, texts_b, device)
        torch.save(acts_b, out_dir / f"{slug}_b.pt")
        del acts_b

        print(f"[02]   done in {time.time() - t1:.1f}s")

        by_slug[slug] = {
            "slug": slug,
            "lang_a": lang_a,
            "lang_b": lang_b,
            "label": label,
            "n_pairs": len(pairs),
            "ids": ids,
            "files": [f"{slug}_a.pt", f"{slug}_b.pt"],
        }

    # Reorder the manifest's contrasts list to match CONTRAST_SETS so it's
    # stable across runs.
    manifest["contrasts"] = [
        by_slug[c[0]] for c in CONTRAST_SETS if c[0] in by_slug
    ]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[02] manifest -> {manifest_path}")
    print("[02] done.")


if __name__ == "__main__":
    main()
