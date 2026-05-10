"""Build a DPO dataset from DIA-LOC's D1 BMvsNN paraphrase pairs.

DPO requires (prompt, chosen, rejected) triples. We frame the
nynorsk-translation task as:

  prompt   = "Skriv på nynorsk: <BM sentence>"
  chosen   = <NN sentence>          (Apertium-translated, target form)
  rejected = <BM sentence>          (the same source — wrong target form)

The "rejected = passthrough" framing is deliberate. The base model's
default failure mode under "skriv på nynorsk" is to keep producing
bokmål; the rejected sample teaches the model to *prefer NN over a
BM passthrough*, which is exactly the failure to correct.

Splits: 80% / 20% deterministic by D1 pair id (md5 % 5 == 0 → eval).
   train: 160 pairs (DPO training set)
   eval:  40 pairs (held out for §15 evaluation; never seen in DPO)

Output:
   data/dpo_train.jsonl   — DPO triples
   data/dpo_eval.jsonl    — held-out for evaluation

Run:
   python src/13_build_dpo_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.config import DATA_DIR, contrast_jsonl_path
from lib.eval_set import load_pairs


PROMPT_TEMPLATE = "Skriv på nynorsk: {bm}"


def _is_eval(pair_id: str) -> bool:
    """Deterministic 80/20 split by md5 of the pair id."""
    h = int(hashlib.md5(pair_id.encode("utf-8")).hexdigest(), 16)
    return h % 5 == 0


def main() -> None:
    pairs = load_pairs(contrast_jsonl_path("d1"))
    print(f"[13] loaded {len(pairs)} D1 pairs (BMvsNN paraphrases)")

    train_rows: list[dict] = []
    eval_rows: list[dict] = []

    for p in pairs:
        bm = p["text_a"].strip()
        nn = p["text_b"].strip()
        if bm == nn:
            continue  # Apertium produced no change for this pair

        record = {
            "id": p["id"],
            "prompt": PROMPT_TEMPLATE.format(bm=bm),
            "chosen": nn,
            "rejected": bm,
        }
        if _is_eval(p["id"]):
            eval_rows.append(record)
        else:
            train_rows.append(record)

    out_train = DATA_DIR / "dpo_train.jsonl"
    out_eval = DATA_DIR / "dpo_eval.jsonl"
    out_train.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n",
        encoding="utf-8",
    )
    out_eval.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_rows) + "\n",
        encoding="utf-8",
    )

    print(f"[13] train: {len(train_rows)} -> {out_train}")
    print(f"[13] eval:  {len(eval_rows)} -> {out_eval}")
    if train_rows:
        sample = train_rows[0]
        print()
        print("[13] sample triple:")
        print(f"  prompt:   {sample['prompt']}")
        print(f"  chosen:   {sample['chosen']}")
        print(f"  rejected: {sample['rejected']}")


if __name__ == "__main__":
    main()
