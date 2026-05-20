#!/usr/bin/env python3
"""Entry point: fine-tune an LLM with LoRA/QLoRA on custom instruction data."""
import argparse
import json
import os
import sys
from pathlib import Path

from src.lora_config import LoRAConfig, TrainingConfig
from src.trainer import train
from src.evaluation import run_evaluation, compute_perplexity
from src.dataset import load_alpaca_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/phi-2", help="HF model id or local path")
    p.add_argument("--data", default="data/sample_instructions.json")
    p.add_argument("--output", default="outputs/lora_adapter")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--qlora", action="store_true", help="Enable 4-bit QLoRA")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--wandb-project", default="llm-finetuning")
    p.add_argument("--run-name", default=None)
    p.add_argument("--eval-only", action="store_true", help="Skip training, run eval on existing adapter")
    p.add_argument("--eval-samples", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()

    lora_cfg = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_in_4bit=args.qlora,
    )
    cfg = TrainingConfig(
        model_name=args.model,
        output_dir=args.output,
        data_path=args.data,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        fp16=args.fp16,
        bf16=args.bf16,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        lora=lora_cfg,
    )

    if not args.eval_only:
        trainer = train(cfg)

    # evaluation
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print("\n[Eval] Loading fine-tuned model for evaluation...")
    tok = AutoTokenizer.from_pretrained(args.output)
    model = AutoModelForCausalLM.from_pretrained(
        args.output,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    examples = load_alpaca_json(args.data)
    eval_set = examples[: args.eval_samples]
    metrics = run_evaluation(model, tok, eval_set)

    print("\n[Results]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    out_path = Path(args.output) / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Eval] Metrics saved → {out_path}")


if __name__ == "__main__":
    main()
