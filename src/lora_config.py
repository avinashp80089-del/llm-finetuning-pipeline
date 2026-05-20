from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # QLoRA-specific
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    model_name: str = "microsoft/phi-2"
    output_dir: str = "outputs/lora_adapter"
    data_path: str = "data/sample_instructions.json"

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_length: int = 512

    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 3

    fp16: bool = False
    bf16: bool = False

    # W&B
    report_to: str = "wandb"
    run_name: Optional[str] = None
    wandb_project: str = "llm-finetuning"

    lora: LoRAConfig = field(default_factory=LoRAConfig)


def build_peft_config(cfg: LoRAConfig):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )


def build_bnb_config(cfg: LoRAConfig):
    import torch
    from transformers import BitsAndBytesConfig

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=dtype_map.get(cfg.bnb_4bit_compute_dtype, torch.float16),
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )
