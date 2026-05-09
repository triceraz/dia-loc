"""Build the three contrast sets for DIA-LOC.

D1: paired BM <-> NN — sourced via Apertium nob->nno translation of BM seeds
                       drawn from random Norwegian Bokmål Wikipedia articles.
D2: paired NB <-> EN — sourced from FLORES-200 (nob_Latn / eng_Latn).
D3: paired BM <-> BM — LLM-paraphrased BM seeds (surface-variation control).

The point of D3 is methodological hygiene. Any probe that distinguishes
paraphrases of the same dialect at deep layers is detecting surface noise,
not language structure. D3 tells us where to set the noise floor for D1/D2
claims.

Run:

    python src/01_build_contrast_sets.py

Outputs (data/):
    d1_bm_nn_pairs.jsonl  (200 pairs)
    d2_nb_en_pairs.jsonl  (200 pairs)
    d3_bm_bm_pairs.jsonl  (100 pairs)

Schema (one JSON object per line):
    {"id": "d1_001",
     "lang_a": "bm", "lang_b": "nn",
     "text_a": "...", "text_b": "...",
     "source": "wikipedia_seed+apertium_nob_nno"}

Environment for D3 (LLM paraphrase via Tenki MLX tunnel):
    LLM_BASE_URL  e.g. https://mlx.tenki.no/v1
    LLM_API_KEY   bearer token
    LLM_MODEL     e.g. mlx-community/gemma-3-4b-it-4bit
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load .env from the repo root if present, so the operator can stash
# LLM_BASE_URL / LLM_API_KEY / LLM_MODEL once and forget. The .env file
# is gitignored. Falls through silently if python-dotenv isn't installed
# or the file doesn't exist.
_REPO_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

# Wikipedia API endpoints. The Norwegian Bokmål wiki lives at no.wikipedia.org
# (the project code is "nowiki"); the Nynorsk wiki is at nn.wikipedia.org.
WIKI_BM_API = "https://no.wikipedia.org/w/api.php"

# Apertium has the gold-standard rule-based bm<->nn translator. The free
# public web service is at apertium.org/apy. Pair codes follow ISO-639-3:
# nob = Norwegian Bokmål, nno = Norwegian Nynorsk.
APERTIUM_API = "https://apertium.org/apy/translate"
APERTIUM_PAIR = "nob|nno"

# FLORES-200 release tarball, hosted by Facebook AI Research. ~13 MB,
# contains plain text dev / devtest splits per language. We download
# this once and cache it under data/cache/. This avoids the HuggingFace
# `datasets` route, which broke when datasets v4 dropped support for
# script-based datasets and the parquet mirror (`openlanguagedata/
# flores_plus`) became gated.
FLORES_TARBALL_URL = (
    "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
)

D1_TARGET = 200
D2_TARGET = 200
D3_TARGET = 100

# Quality bounds for seed sentences. Too short and there's no contextual
# signal; too long and we hit attention-cost issues during activation capture.
MIN_LEN = 40
MAX_LEN = 220

# Be polite to the public APIs. Wikipedia tolerates ~200 req/min for read
# operations; Apertium's public APy is friendlier when batched but we err
# conservative with a 0.5s sleep between calls.
WIKI_SLEEP = 0.3
APERTIUM_SLEEP = 0.5


# ---------------------------------------------------------------------------
# Wikipedia BM seed mining
# ---------------------------------------------------------------------------

def _wikipedia_random_pageids(n: int) -> list[int]:
    """Return n random article page IDs from Norwegian Bokmål Wikipedia."""
    r = requests.get(
        WIKI_BM_API,
        params={
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": min(n, 50),  # API max
        },
        headers={"User-Agent": "DIA-LOC research bot (Tenki Labs)"},
        timeout=30,
    )
    r.raise_for_status()
    return [p["id"] for p in r.json()["query"]["random"]]


def _wikipedia_extracts(page_ids: list[int]) -> list[str]:
    """Fetch plain-text intro extracts for a batch of page IDs."""
    r = requests.get(
        WIKI_BM_API,
        params={
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": "true",
            "explaintext": "true",
            "pageids": "|".join(str(i) for i in page_ids),
        },
        headers={"User-Agent": "DIA-LOC research bot (Tenki Labs)"},
        timeout=30,
    )
    r.raise_for_status()
    return [
        page.get("extract", "")
        for page in r.json()["query"]["pages"].values()
    ]


def _first_clean_sentence(extract: str) -> Optional[str]:
    """Return the first sentence of a Wikipedia extract that meets quality bounds."""
    if not extract:
        return None
    # Wikipedia plaintext extracts use ". " as a sentence-ish boundary. This
    # is good enough for our purposes; we don't need linguistic-grade
    # tokenization for sourcing seed text.
    for raw in extract.split(". "):
        s = raw.strip().rstrip(".")
        if not s:
            continue
        if "==" in s or s.startswith("{") or s.startswith("|"):
            # Skip wikitext fragments that occasionally leak through
            continue
        if MIN_LEN <= len(s) <= MAX_LEN:
            return s + "."
    return None


def fetch_bm_seed_sentences(n: int) -> list[str]:
    """Pull n short BM sentences from random Norwegian Bokmål Wikipedia articles.

    The seeds end up in BOTH D1 (translated to NN via Apertium) and D3
    (paraphrased in BM via an LLM), so we want clean, definition-style
    lead sentences. "Oslo er hovedstaden i Norge" is the archetype.
    """
    sentences: list[str] = []
    seen: set[str] = set()
    pbar = tqdm(total=n, desc="wiki bm seeds")
    while len(sentences) < n:
        try:
            ids = _wikipedia_random_pageids(40)
            extracts = _wikipedia_extracts(ids)
        except Exception as e:
            tqdm.write(f"  wiki batch failed, retrying: {e}")
            time.sleep(2)
            continue

        for ext in extracts:
            s = _first_clean_sentence(ext)
            if s and s not in seen:
                sentences.append(s)
                seen.add(s)
                pbar.update(1)
                if len(sentences) >= n:
                    break
        time.sleep(WIKI_SLEEP)
    pbar.close()
    return sentences[:n]


# ---------------------------------------------------------------------------
# D1: BM -> NN via Apertium
# ---------------------------------------------------------------------------

def apertium_translate(text: str, pair: str = APERTIUM_PAIR) -> str:
    """Translate a sentence via the Apertium public web API.

    Apertium is rule-based, deterministic, and explicitly designed for
    closely-related language pairs like Bokmål<->Nynorsk. It's the right
    tool here precisely because we want a TRUE paraphrase (same meaning,
    different surface form) rather than a noisy LLM translation that
    might drift semantically.
    """
    r = requests.get(
        APERTIUM_API,
        params={"langpair": pair, "q": text, "markUnknown": "no"},
        timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("responseStatus") != 200:
        raise RuntimeError(f"apertium error: {body.get('responseDetails', body)}")
    return body["responseData"]["translatedText"]


def build_d1(target_size: int = D1_TARGET) -> None:
    """Build D1: BM -> NN paired sentences via Apertium translation."""
    out_path = DATA_DIR / "d1_bm_nn_pairs.jsonl"
    if out_path.exists():
        existing = sum(1 for _ in out_path.open(encoding="utf-8"))
        if existing >= target_size:
            print(f"d1 already has {existing} pairs at {out_path}, skipping")
            return

    # Oversample seeds — some translations will be no-ops (same word forms
    # in BM and NN) or fail. Empirically ~70-80% survive the filter.
    seeds = fetch_bm_seed_sentences(int(target_size * 1.4))

    out: list[dict] = []
    for i, bm in enumerate(tqdm(seeds, desc="d1: apertium nob->nno")):
        if len(out) >= target_size:
            break
        try:
            nn = apertium_translate(bm)
        except Exception as e:
            tqdm.write(f"  seed {i}: apertium failed ({e}), skipping")
            time.sleep(1)
            continue
        nn = (nn or "").strip()
        if not nn or nn == bm.strip():
            # Apertium returned identical text — sentence has no BM/NN-divergent
            # forms, so it's not useful for the contrast.
            continue
        out.append({
            "id": f"d1_{len(out):03d}",
            "lang_a": "bm",
            "lang_b": "nn",
            "text_a": bm,
            "text_b": nn,
            "source": "wikipedia_seed+apertium_nob_nno",
        })
        time.sleep(APERTIUM_SLEEP)

    _write_jsonl(out_path, out)
    print(f"d1: wrote {len(out)} pairs -> {out_path}")


# ---------------------------------------------------------------------------
# D2: FLORES-200
# ---------------------------------------------------------------------------

def _ensure_flores_tarball() -> Path:
    """Download FLORES-200 tarball into data/cache/ if not already there."""
    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / "flores200_dataset.tar.gz"
    if tar_path.exists() and tar_path.stat().st_size > 1_000_000:
        return tar_path
    print(f"d2: downloading FLORES-200 tarball ({FLORES_TARBALL_URL}) ...")
    with requests.get(FLORES_TARBALL_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        pbar = tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc="d2: download",
        )
        with tar_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        pbar.close()
    return tar_path


def _read_flores_lines(tar_path: Path, lang_code: str, split: str = "dev") -> list[str]:
    """Extract one language's sentences from the FLORES tarball.

    The tarball layout (since the v1 release) is:
        flores200_dataset/<split>/<lang_code>.<split>
    where each file is one sentence per line, plain UTF-8.
    """
    import tarfile

    # The tarball entries are stored with a leading "./" prefix on the
    # canonical FLORES-200 release. Try both forms so we work even if a
    # future repack drops it.
    candidates = [
        f"./flores200_dataset/{split}/{lang_code}.{split}",
        f"flores200_dataset/{split}/{lang_code}.{split}",
    ]
    with tarfile.open(tar_path, "r:gz") as tar:
        f = None
        for name in candidates:
            try:
                f = tar.extractfile(name)
                if f is not None:
                    break
            except KeyError:
                continue
        if f is None:
            raise RuntimeError(
                f"FLORES member for {lang_code}.{split} not found in tarball; "
                f"tried: {candidates}"
            )
        if f is None:
            raise RuntimeError(f"FLORES member {member_name} unreadable")
        return [line.decode("utf-8").strip() for line in f.readlines()]


def build_d2(target_size: int = D2_TARGET) -> None:
    """Build D2: NB <-> EN paired sentences from FLORES-200.

    FLORES-200 is the canonical multilingual benchmark for sentence-level
    translation parity. The dev split has ~1k high-quality human-translated
    sentences per language. We download the official tarball directly from
    Facebook AI's CDN — the HuggingFace datasets route is no longer
    workable on `datasets` >= 4.0 (script-based loaders dropped) and the
    maintained parquet mirror is gated.
    """
    out_path = DATA_DIR / "d2_nb_en_pairs.jsonl"
    if out_path.exists():
        existing = sum(1 for _ in out_path.open(encoding="utf-8"))
        if existing >= target_size:
            print(f"d2 already has {existing} pairs at {out_path}, skipping")
            return

    tar_path = _ensure_flores_tarball()
    print(f"d2: extracting nob_Latn / eng_Latn from {tar_path.name}")
    nob_lines = _read_flores_lines(tar_path, "nob_Latn", "dev")
    eng_lines = _read_flores_lines(tar_path, "eng_Latn", "dev")
    if len(nob_lines) != len(eng_lines):
        sys.exit(
            f"FLORES line count mismatch: nob={len(nob_lines)} eng={len(eng_lines)} "
            "— refusing to build D2 from misaligned files."
        )
    print(f"d2: {len(nob_lines)} parallel lines available")

    out: list[dict] = []
    for i, (nb, en) in enumerate(zip(nob_lines, eng_lines)):
        if len(out) >= target_size:
            break
        nb, en = nb.strip(), en.strip()
        if not (MIN_LEN <= len(nb) <= MAX_LEN and MIN_LEN <= len(en) <= MAX_LEN):
            continue
        out.append({
            "id": f"d2_{len(out):03d}",
            "lang_a": "nb",
            "lang_b": "en",
            "text_a": nb,
            "text_b": en,
            "source": "flores-200",
        })

    _write_jsonl(out_path, out)
    print(f"d2: wrote {len(out)} pairs -> {out_path}")


# ---------------------------------------------------------------------------
# D3: BM -> BM paraphrase via LLM (Tenki MLX tunnel by default)
# ---------------------------------------------------------------------------

def llm_paraphrase_bm(text: str) -> str:
    """Get a near-paraphrase of a BM sentence via an OpenAI-compatible endpoint.

    Defaults to the Tenki MLX tunnel (LLM_BASE_URL=https://mlx.tenki.no/v1)
    so we don't burn a third-party API budget on the control set. Any
    OpenAI-compatible endpoint works; just point LLM_BASE_URL at it.
    """
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")

    if not (api_key and base_url and model):
        raise RuntimeError(
            "LLM_BASE_URL, LLM_API_KEY, LLM_MODEL must all be set for D3. "
            "Suggested: point LLM_BASE_URL at https://mlx.tenki.no/v1 and "
            "use the same LLM_API_KEY as the production tenki-web container."
        )

    r = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "temperature": 0.6,
            "max_tokens": 200,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Du er en parafrasegenerator for norsk bokmål. "
                        "Brukeren gir deg en setning på bokmål. Skriv én ny "
                        "setning som betyr nøyaktig det samme, men med andre "
                        "ord og en annen setningsstruktur. Hold den på bokmål, "
                        "ikke nynorsk. Returner kun den nye setningen, ingen "
                        "anførselstegn, ingen forklaring."
                    ),
                },
                {"role": "user", "content": text},
            ],
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=90,
    )
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"].strip()
    # Models occasionally wrap the output in quotes despite the instruction
    return out.strip("\"'`")


def build_d3(target_size: int = D3_TARGET) -> None:
    """Build D3: BM <-> BM paraphrases (control for surface variation)."""
    out_path = DATA_DIR / "d3_bm_bm_pairs.jsonl"
    if out_path.exists():
        existing = sum(1 for _ in out_path.open(encoding="utf-8"))
        if existing >= target_size:
            print(f"d3 already has {existing} pairs at {out_path}, skipping")
            return

    # Reuse fresh seeds (we don't share with D1's seeds — we want the control
    # to be statistically independent so D1 vs D3 isn't confounded).
    seeds = fetch_bm_seed_sentences(int(target_size * 1.5))

    out: list[dict] = []
    for i, bm in enumerate(tqdm(seeds, desc="d3: llm bm paraphrase")):
        if len(out) >= target_size:
            break
        try:
            paraphrase = llm_paraphrase_bm(bm)
        except Exception as e:
            tqdm.write(f"  seed {i}: paraphrase failed ({e}), skipping")
            time.sleep(2)
            continue
        if not paraphrase or paraphrase == bm.strip():
            continue
        out.append({
            "id": f"d3_{len(out):03d}",
            "lang_a": "bm",
            "lang_b": "bm",
            "text_a": bm,
            "text_b": paraphrase,
            "source": "wikipedia_seed+llm_paraphrase",
        })
        # Small backoff so the local MLX server isn't slammed
        time.sleep(0.2)

    _write_jsonl(out_path, out)
    print(f"d3: wrote {len(out)} pairs -> {out_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("DIA-LOC week 1: building contrast sets D1, D2, D3")
    print(f"Output dir: {DATA_DIR}")
    print()

    # D2 first — the easiest, fully automatic, no rate-limited API hits.
    # If FLORES isn't reachable, we want to know before spending half an
    # hour on Wikipedia seeds.
    build_d2()
    print()

    build_d1()
    print()

    build_d3()
    print()

    print("Done. Quick stats:")
    for name in ("d1_bm_nn_pairs", "d2_nb_en_pairs", "d3_bm_bm_pairs"):
        p = DATA_DIR / f"{name}.jsonl"
        n = sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0
        print(f"  {name}: {n} pairs")


if __name__ == "__main__":
    main()
