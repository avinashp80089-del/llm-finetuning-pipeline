import os
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.lora_config import LoRAConfig, TrainingConfig, build_bnb_config, build_peft_config
from src.dataset import load_and_tokenize


def load_base_model(cfg: TrainingConfig):
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if cfg.lora.load_in_4bit:
        load_kwargs["quantization_config"] = build_bnb_config(cfg.lora)
    else:
        load_kwargs["torch_dtype"] = torch.float16 if cfg.fp16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)
    model.config.use_cache = False  # required for gradient checkpointing
    model.enable_input_require_grads()
    return model, tok


def apply_lora(model, cfg: LoRAConfig):
    from peft import get_peft_model, prepare_model_for_kbit_training

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_cfg = build_peft_config(cfg)
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


def build_training_args(cfg: TrainingConfig) -> TrainingArguments:
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        report_to=cfg.report_to,
        run_name=cfg.run_name or f"lora-{cfg.model_name.split('/')[-1]}",
        dataloader_num_workers=0,
        group_by_length=True,  # speeds up training by reducing padding
    )


def train(cfg: TrainingConfig):
    model, tok = load_base_model(cfg)
    model = apply_lora(model, cfg.lora)

    train_ds, val_ds = load_and_tokenize(cfg.data_path, tok, cfg.max_length)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    args = build_training_args(cfg)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print(f"[Trainer] Starting fine-tuning: {len(train_ds)} train / {len(val_ds)} val samples")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    print(f"[Trainer] Adapter saved → {cfg.output_dir}")
    return trainer
