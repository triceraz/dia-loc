"""Central config for DIA-LOC: model id, paths, contrast registry.

Everything the activation-capture, probes, and SAE scripts share goes
here so the wiring stays consistent across stages.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Primary model for v1 of the paper. Qwen 2.5 1.5B Instruct is small enough
# to fit on a 3060 Ti at FP16 with plenty of headroom for activations, and
# big enough to have non-trivial residual-stream structure to localize.
# When we want to extend to Gemma 3 4B (the production Hugin model) for v2,
# overriding MODEL_ID via env var is enough — every script reads from here.
import os

MODEL_ID = os.environ.get("DIA_LOC_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# Tokenizer + dtype defaults. fp16 keeps memory low; the activation tensors
# we save are also fp16 to halve disk usage (~3 GB instead of ~6 GB).
DTYPE_NAME = os.environ.get("DIA_LOC_DTYPE", "float16")

# Maximum input length. FLORES sentences are typically 30-200 chars,
# Wikipedia leads similar. 256 tokens is comfortable headroom for those
# at Qwen's tokenization rate, and keeps activation memory bounded.
MAX_TOKENS = int(os.environ.get("DIA_LOC_MAX_TOKENS", "256"))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root is the parent of `src/`. Resolved at import time so callers
# get the same directory regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"

# All activations land under runs/{checkpoint_slug}/{contrast_id}/{layer}.pt
# Checkpoint slug = sanitized MODEL_ID (slashes to underscores).
def checkpoint_slug() -> str:
    return MODEL_ID.replace("/", "_").replace("-", "_")


def activations_dir() -> Path:
    return RUNS_DIR / "activations" / checkpoint_slug()


# ---------------------------------------------------------------------------
# Contrast set registry
# ---------------------------------------------------------------------------

# (slug, lang_a, lang_b, jsonl filename, label)
CONTRAST_SETS: list[tuple[str, str, str, str, str]] = [
    ("d1", "bm", "nn", "d1_bm_nn_pairs.jsonl", "Bokmål vs Nynorsk (dialectal)"),
    ("d2", "nb", "en", "d2_nb_en_pairs.jsonl", "NB vs EN (foreign-language)"),
    ("d3", "bm", "bm", "d3_bm_bm_pairs.jsonl", "Bokmål vs Bokmål (control)"),
]


def contrast_jsonl_path(slug: str) -> Path:
    for s, _, _, fname, _ in CONTRAST_SETS:
        if s == slug:
            return DATA_DIR / fname
    raise KeyError(f"unknown contrast slug: {slug}")
