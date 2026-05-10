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


POOL_MODES = ("mean", "last")


def _pool_layer_outputs(
    layer_outputs: dict[int, torch.Tensor],
    attention_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Reduce each layer's [batch, seq, d] tensor to [batch, d] per `mode`.

    Returns shape [batch, n_layers, d] in fp16.

    Modes:
      mean: mean over real (non-pad) token positions. Sentence-level
            summary; what 03 (cosine + CKA) wants.
      last: residual at the last real token position. The autoregressive
            "next-token prediction" anchor; what 04 (logit lens) needs
            to make sense.
    """
    if mode not in POOL_MODES:
        raise ValueError(f"unknown pool mode: {mode}; expected {POOL_MODES}")

    n_layers = len(layer_outputs)
    pooled_per_layer: list[torch.Tensor] = []

    if mode == "mean":
        mask = attention_mask.to(torch.float32).unsqueeze(-1)  # [B, S, 1]
        mask_sum = mask.sum(dim=1).clamp(min=1.0)
        for li in range(n_layers):
            h = layer_outputs[li].to(torch.float32)  # [B, S, d]
            pooled_per_layer.append((h * mask).sum(dim=1) / mask_sum)  # [B, d]
    else:  # "last"
        # Index of the last real token per row. attention_mask has 1's
        # for real tokens and 0's for pad; argmax-from-the-right gives
        # us the last 1.
        last_idx = attention_mask.long().sum(dim=1) - 1  # [B]
        for li in range(n_layers):
            h = layer_outputs[li].to(torch.float32)  # [B, S, d]
            # Gather along the seq axis using last_idx per batch row.
            batch_idx = torch.arange(h.shape[0], device=h.device)
            pooled_per_layer.append(h[batch_idx, last_idx])  # [B, d]

    return torch.stack(pooled_per_layer, dim=1).to(torch.float16)


@torch.no_grad()
def capture_per_token(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    layers: list[int],
) -> tuple[dict[int, torch.Tensor], list[dict]]:
    """Capture per-token residuals at SPECIFIC layers, flattened.

    For each text, runs forward, captures residual at each requested
    layer for every real token, and returns:

      acts_by_layer: {layer_idx: [total_tokens, d_model] fp16 tensor}
      meta:          list of per-row dicts with input_idx, position,
                     and length-of-source-input

    SAE training reads the flat tensor; differential analysis groups
    rows by input_idx via the meta list.
    """
    n_layers = num_layers(model)
    for li in layers:
        if not (0 <= li < n_layers):
            raise ValueError(f"layer {li} out of range [0, {n_layers})")

    rows_by_layer: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
    # Meta is shared across layers — same (input_idx, position) ordering
    # per layer's flat tensor since all layers see the same inputs.
    meta: list[dict] = []

    for input_idx, text in enumerate(tqdm(texts, desc="forward (per-token)", leave=False)):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            padding=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        seq_len = int(attention_mask.sum().item())

        out: dict[int, torch.Tensor] = {}
        with capture_residuals(model, out):
            model(input_ids=input_ids, attention_mask=attention_mask)

        for li in layers:
            h = out[li].to(torch.float32)  # [1, S, D]
            # padding=False, so slicing to seq_len is exact.
            real = h[0, :seq_len, :].to(torch.float16).cpu()  # [seq_len, D]
            rows_by_layer[li].append(real)

        for pos in range(seq_len):
            meta.append({
                "input_idx": input_idx,
                "position": pos,
                "input_seq_len": seq_len,
            })

    # Concatenate per-layer rows into a single flat tensor.
    acts_by_layer = {
        li: torch.cat(rows_by_layer[li], dim=0) if rows_by_layer[li] else torch.empty(0, num_layers(model))
        for li in layers
    }
    return acts_by_layer, meta


@torch.no_grad()
def capture_one_side(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    pool_modes: list[str],
) -> dict[str, torch.Tensor]:
    """Run all `texts` through the model, capture pooled residuals.

    Returns a dict mapping each pool mode to a tensor of shape
    [n_texts, n_layers, d_model] in fp16 on CPU.
    """
    n_layers = num_layers(model)
    rows_per_mode: dict[str, list[torch.Tensor]] = {m: [] for m in pool_modes}

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

        # Hooks already CPU-offloaded the activations; mask must be on CPU
        # too so the index/multiply happens against the same device.
        cpu_mask = attention_mask.to("cpu")
        for mode in pool_modes:
            pooled = _pool_layer_outputs(out, cpu_mask, mode)  # [1, L, d]
            rows_per_mode[mode].append(pooled)

    return {m: torch.cat(rows, dim=0) for m, rows in rows_per_mode.items()}


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
    parser.add_argument(
        "--pool",
        default="mean,last",
        help=(
            "Comma-separated pooling modes. 'mean' (sentence summary; for "
            "cosine+CKA), 'last' (last real token; for logit lens). Default "
            "captures both since the marginal cost is tiny."
        ),
    )
    parser.add_argument(
        "--per-token-layers",
        default="",
        help=(
            "Comma-separated layer indices to ALSO save per-token "
            "activations for (e.g. '20' or '0,14,27'). Each chosen layer "
            "produces a flat [total_tokens, d_model] tensor + meta JSON "
            "for SAE training and per-token analysis. Skipped when empty."
        ),
    )
    args = parser.parse_args()

    if args.contrasts:
        wanted = {s.strip() for s in args.contrasts.split(",") if s.strip()}
    else:
        wanted = None

    pool_modes = [m.strip() for m in args.pool.split(",") if m.strip()]
    for m in pool_modes:
        if m not in POOL_MODES:
            sys.exit(f"unknown pool mode: {m}; expected {POOL_MODES}")

    per_token_layers: list[int] = []
    if args.per_token_layers:
        per_token_layers = [
            int(x.strip()) for x in args.per_token_layers.split(",") if x.strip()
        ]

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

        print(
            f"[02] {slug} ({label}): {len(pairs)} pairs, "
            f"pool={','.join(pool_modes)}"
        )
        t1 = time.time()
        # Mean-pool stays on the canonical filenames {slug}_a.pt / {slug}_b.pt
        # for backward compat with 03_similarity.py. Other pool modes get a
        # suffix so multiple modes can coexist.
        acts_a = capture_one_side(model, tokenizer, texts_a, device, pool_modes)
        for mode, t in acts_a.items():
            suffix = "" if mode == "mean" else f"_{mode}"
            torch.save(t, out_dir / f"{slug}_a{suffix}.pt")
        del acts_a

        acts_b = capture_one_side(model, tokenizer, texts_b, device, pool_modes)
        for mode, t in acts_b.items():
            suffix = "" if mode == "mean" else f"_{mode}"
            torch.save(t, out_dir / f"{slug}_b{suffix}.pt")
        del acts_b

        print(f"[02]   done in {time.time() - t1:.1f}s")

        files = []
        for mode in pool_modes:
            suffix = "" if mode == "mean" else f"_{mode}"
            files.extend([f"{slug}_a{suffix}.pt", f"{slug}_b{suffix}.pt"])

        # Optional per-token capture for SAE training + non-final-position
        # logit lens. One flat [total_tokens, d_model] tensor + meta JSON
        # per (side, layer).
        if per_token_layers:
            for side, texts_side in (("a", texts_a), ("b", texts_b)):
                acts_by_layer, meta = capture_per_token(
                    model, tokenizer, texts_side, device, per_token_layers,
                )
                for li, t in acts_by_layer.items():
                    pt_path = out_dir / f"{slug}_{side}_l{li:02d}_pertoken.pt"
                    torch.save(t, pt_path)
                    files.append(pt_path.name)
                meta_path = out_dir / f"{slug}_{side}_pertoken_meta.json"
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False),
                    encoding="utf-8",
                )
                files.append(meta_path.name)
            print(
                f"[02]   per-token: layers={per_token_layers} "
                f"({sum(t.shape[0] for t in acts_by_layer.values())} tokens/side)"
            )

        by_slug[slug] = {
            "slug": slug,
            "lang_a": lang_a,
            "lang_b": lang_b,
            "label": label,
            "n_pairs": len(pairs),
            "ids": ids,
            "pool_modes": pool_modes,
            "per_token_layers": per_token_layers,
            "files": files,
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
