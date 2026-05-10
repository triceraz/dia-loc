"""Evaluate trained DPO adapters on the held-out 38-pair eval set.

Two metrics per adapter (and a baseline of the un-adapted base model):

  1. log-prob margin: mean over eval rows of
        log P(chosen | prompt) - log P(rejected | prompt).
     A positive, large margin means the adapter prefers the NN
     paraphrase over the BM passthrough as the response. The DPO
     training objective is exactly to push this margin up, so this
     is the in-sample metric.

  2. NN validity rate at generation: greedy-decode 32 tokens from
     each prompt, count the fraction of word tokens accepted by
     Apertium-sme-nob's NN analyzer (uralicNLP). This is the
     out-of-distribution metric that tells us whether the model
     learned to GENERATE nynorsk, not just to score it.

We compare three checkpoints:
  - base       (no adapter)
  - full       (LoRA on all 28 layers)
  - targeted   (LoRA only on layers 21-26 per DIA-LOC §4.6)

The v0.4 hypothesis: targeted matches or beats full.

Run:

    python src/15_dpo_evaluate.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.config import DATA_DIR, MODEL_ID, REPO_ROOT


WORD_RE = re.compile(r"[A-Za-zÆØÅæøåÄäÖö]+(?:-[A-Za-zÆØÅæøåÄäÖö]+)?")


def load_eval_rows() -> list[dict]:
    rows: list[dict] = []
    with (DATA_DIR / "dpo_eval.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def sequence_logprob(model, tokenizer, prompt_text: str, response: str, device: str) -> float:
    """Sum of token log-probs for `response` given `prompt_text`.

    Both are encoded together; we mask out the prompt part so only
    the response tokens contribute to the average.
    """
    full = prompt_text + response
    full_ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    out = model(full_ids)
    logits = out.logits  # [1, S, V]
    # Score response tokens at positions [prompt_len-1, S-1)
    # (logits at position t predict token at t+1).
    target_ids = full_ids[:, prompt_len:]
    target_logits = logits[:, prompt_len - 1 : -1, :]
    log_probs = torch.log_softmax(target_logits, dim=-1)
    gathered = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(gathered.sum().item())


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt_text: str, device: str, max_new: int = 32) -> str:
    ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_ids = out[0, ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def nn_validity(text: str) -> float:
    """Fraction of word-tokens in `text` accepted by Apertium-sme-nob's
    NN analyzer (via uralicNLP).

    Lazy-imports uralicNLP so this script still runs (with the metric
    skipped) on environments where it isn't installed.
    """
    try:
        from uralicNLP import uralicApi
    except ImportError:
        return float("nan")

    words = WORD_RE.findall(text)
    if not words:
        return 0.0
    n_valid = 0
    for w in words:
        try:
            analyses = uralicApi.analyze(w.lower(), "nob")
            if analyses:
                n_valid += 1
        except Exception:
            pass
    return n_valid / len(words)


def evaluate_checkpoint(
    label: str,
    adapter_dir: Path | None,
    rows: list[dict],
    tokenizer,
    base_model,
    device: str,
) -> dict:
    print(f"[15] evaluating {label} ...")
    if adapter_dir is None:
        model = base_model
    else:
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    margins: list[float] = []
    chosen_lps: list[float] = []
    rejected_lps: list[float] = []
    gens: list[str] = []
    val_rates: list[float] = []

    for r in rows:
        prompt_text = encode_prompt(tokenizer, r["prompt"])
        lp_chosen = sequence_logprob(model, tokenizer, prompt_text, r["chosen"], device)
        lp_rejected = sequence_logprob(model, tokenizer, prompt_text, r["rejected"], device)
        chosen_lps.append(lp_chosen)
        rejected_lps.append(lp_rejected)
        margins.append(lp_chosen - lp_rejected)

        gen = greedy_generate(model, tokenizer, prompt_text, device, max_new=48)
        gens.append(gen)
        val_rates.append(nn_validity(gen))

    # If we patched a peft adapter onto the base, unload so the next
    # iteration sees a clean base.
    if adapter_dir is not None:
        base_again = model.unload()
        # base_model already references the same underlying weights,
        # nothing else to do.
        del base_again

    return {
        "label": label,
        "n": len(rows),
        "mean_margin": sum(margins) / len(margins),
        "mean_chosen_lp": sum(chosen_lps) / len(chosen_lps),
        "mean_rejected_lp": sum(rejected_lps) / len(rejected_lps),
        "mean_nn_validity": (
            sum(v for v in val_rates if v == v) / max(1, sum(1 for v in val_rates if v == v))
        ),
        "first_three_generations": gens[:3],
    }


def main() -> None:
    rows = load_eval_rows()
    print(f"[15] {len(rows)} eval rows")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[15] loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    ).to(device)
    base_model.eval()

    runs_dir = REPO_ROOT / "runs" / "dpo"
    results: list[dict] = []

    results.append(
        evaluate_checkpoint("base", None, rows, tokenizer, base_model, device)
    )

    # Auto-discover every trained adapter under runs/dpo/.
    discovered: list[Path] = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "adapter_model.safetensors").exists()
    )
    for adapter in discovered:
        results.append(
            evaluate_checkpoint(
                adapter.name, adapter, rows, tokenizer, base_model, device
            )
        )

    out = REPO_ROOT / "runs" / "dpo" / "evaluation.json"
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[15] -> {out}")
    print()
    print(
        f"  {'variant':<10} {'margin':>8}  {'chosen_lp':>10}  "
        f"{'rejected_lp':>12}  {'NN valid':>9}"
    )
    for r in results:
        print(
            f"  {r['label']:<10} {r['mean_margin']:>+8.3f}  "
            f"{r['mean_chosen_lp']:>10.3f}  {r['mean_rejected_lp']:>12.3f}  "
            f"{r['mean_nn_validity']:>9.3f}"
        )


if __name__ == "__main__":
    main()
