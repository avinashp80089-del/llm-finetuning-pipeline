import json
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizer


ALPACA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def load_alpaca_json(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    return data


def format_prompt(example: Dict) -> str:
    if example.get("input", "").strip():
        return ALPACA_PROMPT.format(**example)
    return ALPACA_PROMPT_NO_INPUT.format(**example)


def tokenize_dataset(
    examples: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    add_eos: bool = True,
) -> Dataset:
    texts = [format_prompt(e) for e in examples]
    if add_eos:
        texts = [t + tokenizer.eos_token for t in texts]

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    # labels = input_ids for causal LM; masking handled by DataCollatorForLanguageModeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    return Dataset.from_dict(tokenized)


def split_dataset(dataset: Dataset, val_ratio: float = 0.1, seed: int = 42):
    split = dataset.train_test_split(test_size=val_ratio, seed=seed)
    return split["train"], split["test"]


def load_and_tokenize(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    val_ratio: float = 0.1,
):
    examples = load_alpaca_json(data_path)
    dataset = tokenize_dataset(examples, tokenizer, max_length)
    return split_dataset(dataset, val_ratio)
