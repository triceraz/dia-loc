"""DPO LoRA training for BM->NN dialect adherence.

Trains two LoRA adapters on Qwen 2.5 1.5B Instruct using the DPO triples
from src/13:

  full      : standard LoRA, target_modules=[q,k,v,o], r=16, all 28 layers
  targeted  : same shape, but layers_to_transform=[21,22,23,24,25,26]
              (the consolidation band identified in DIA-LOC v0.3 §4.6/4.7)

Comparing the two answers a v0.4 follow-up question to DIA-LOC: the
paper showed that the dialect signal *consolidates* in layers 21-26 of
the last-token residual. If that finding is mechanistically right,
LoRA capacity targeted only at those layers should match (or exceed)
the gain from full-stack LoRA, with about 21% of the trainable
parameters.

Compute: ~10-20 minutes per adapter on a 3060 Ti.

Run:

    python src/14_dpo_train.py --variant full
    python src/14_dpo_train.py --variant targeted

Outputs (per variant):

    runs/dpo/<variant>/   adapter_model.safetensors + tokenizer + meta
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from lib.config import DATA_DIR, MODEL_ID, REPO_ROOT


# Layers with strongest BM/NN consolidation in Qwen 2.5 1.5B per
# DIA-LOC §4.6 (50% transfer L21, 90% transfer L26).
TARGETED_LAYERS = [21, 22, 23, 24, 25, 26]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("full", "targeted"), required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    out_dir = REPO_ROOT / "runs" / "dpo" / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[14] variant={args.variant}  out={out_dir}")

    # ---- load base model + tokenizer ----
    print(f"[14] loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    # ---- LoRA config ----
    lora_kwargs = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "task_type": "CAUSAL_LM",
        "bias": "none",
    }
    if args.variant == "targeted":
        lora_kwargs["layers_to_transform"] = TARGETED_LAYERS
        # peft requires us to also pass the layer pattern so it knows
        # which Modules constitute a "layer" for indexing. For Qwen 2.5
        # the per-layer container is at model.model.layers.{i}.
        lora_kwargs["layers_pattern"] = "layers"

    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- dataset ----
    print(f"[14] loading dataset ...")
    raw = load_dataset(
        "json",
        data_files={"train": str(DATA_DIR / "dpo_train.jsonl")},
        split="train",
    )

    # Apply Qwen's chat template to the prompt so the model sees the same
    # framing it'd see at inference time.
    def format_row(ex):
        msgs = [{"role": "user", "content": ex["prompt"]}]
        templated = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt": templated,
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
        }

    raw = raw.map(format_row)
    print(f"[14] DPO triples: {len(raw)}")

    # ---- DPO config ----
    cfg = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        beta=args.beta,
        max_length=args.max_length,
        report_to="none",
        seed=42,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # peft auto-handles ref_model from adapter-disabled base
        args=cfg,
        train_dataset=raw,
        processing_class=tokenizer,
    )
    trainer.train()

    # ---- save adapter ----
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {
        "variant": args.variant,
        "model_id": MODEL_ID,
        "epochs": args.epochs,
        "lr": args.lr,
        "beta": args.beta,
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "layers_to_transform": (
            TARGETED_LAYERS if args.variant == "targeted" else None
        ),
        "n_train": len(raw),
    }
    (out_dir / "training_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[14] adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
