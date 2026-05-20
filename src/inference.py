from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.dataset import ALPACA_PROMPT, ALPACA_PROMPT_NO_INPUT


def load_merged_model(adapter_path: str, base_model: Optional[str] = None):
    """Load adapter and merge weights into base model for fast inference (no PEFT overhead)."""
    from peft import PeftModel

    # read base model name from adapter config if not supplied
    if base_model is None:
        import json
        cfg_path = Path(adapter_path) / "adapter_config.json"
        with open(cfg_path) as f:
            base_model = json.load(f)["base_model_name_or_path"]

    tok = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()  # bake LoRA weights into base — halves inference latency
    model.eval()
    return model, tok


def load_adapter_model(adapter_path: str, base_model: Optional[str] = None):
    """Load with PEFT adapter attached — useful when you need to switch adapters at runtime."""
    from peft import PeftModel

    if base_model is None:
        import json
        cfg_path = Path(adapter_path) / "adapter_config.json"
        with open(cfg_path) as f:
            base_model = json.load(f)["base_model_name_or_path"]

    tok = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tok


def build_prompt(instruction: str, input_text: str = "") -> str:
    if input_text.strip():
        return ALPACA_PROMPT.format(instruction=instruction, input=input_text, output="")
    return ALPACA_PROMPT_NO_INPUT.format(instruction=instruction, output="")


def generate(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    prompt = build_prompt(instruction, input_text)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class FineTunedPipeline:
    def __init__(self, adapter_path: str, base_model: Optional[str] = None, merge: bool = True):
        if merge:
            self.model, self.tokenizer = load_merged_model(adapter_path, base_model)
        else:
            self.model, self.tokenizer = load_adapter_model(adapter_path, base_model)

    def __call__(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        return generate(
            self.model,
            self.tokenizer,
            instruction,
            input_text,
            max_new_tokens,
            temperature,
        )
