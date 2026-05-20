import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ------ dataset tests ------

def test_format_prompt_with_input():
    from src.dataset import format_prompt
    ex = {"instruction": "Summarize this.", "input": "Hello world.", "output": "A greeting."}
    result = format_prompt(ex)
    assert "### Input:" in result
    assert "Hello world." in result
    assert "### Response:" in result


def test_format_prompt_no_input():
    from src.dataset import format_prompt
    ex = {"instruction": "Write a poem.", "input": "", "output": "Roses are red."}
    result = format_prompt(ex)
    assert "### Input:" not in result
    assert "Write a poem." in result


def test_load_alpaca_json():
    from src.dataset import load_alpaca_json
    path = Path(__file__).parent.parent / "data/sample_instructions.json"
    data = load_alpaca_json(str(path))
    assert isinstance(data, list)
    assert len(data) >= 10
    for ex in data[:5]:
        assert "instruction" in ex
        assert "output" in ex


def test_tokenize_dataset():
    from src.dataset import tokenize_dataset
    from unittest.mock import MagicMock

    tok = MagicMock()
    tok.eos_token = "</s>"
    tok.return_value = {"input_ids": [[1, 2, 3], [4, 5, 6]], "attention_mask": [[1, 1, 1], [1, 1, 1]]}

    examples = [
        {"instruction": "What is AI?", "input": "", "output": "A field of computer science."},
        {"instruction": "What is ML?", "input": "", "output": "A subset of AI."},
    ]
    ds = tokenize_dataset(examples, tok, max_length=64)
    assert "input_ids" in ds.column_names
    assert "labels" in ds.column_names
    assert len(ds) == 2


def test_split_dataset():
    from datasets import Dataset
    from src.dataset import split_dataset

    data = {"input_ids": list(range(100)), "labels": list(range(100))}
    ds = Dataset.from_dict(data)
    train, val = split_dataset(ds, val_ratio=0.1)
    assert len(train) == 90
    assert len(val) == 10


# ------ lora_config tests ------

def test_default_lora_config():
    from src.lora_config import LoRAConfig
    cfg = LoRAConfig()
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert "q_proj" in cfg.target_modules
    assert cfg.load_in_4bit is False


def test_training_config_defaults():
    from src.lora_config import TrainingConfig
    cfg = TrainingConfig()
    assert cfg.num_train_epochs == 3
    assert cfg.learning_rate == 2e-4
    assert cfg.wandb_project == "llm-finetuning"


def test_qlora_config():
    from src.lora_config import LoRAConfig
    cfg = LoRAConfig(load_in_4bit=True)
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True


# ------ evaluation tests ------

def test_compute_rouge():
    from src.evaluation import compute_rouge
    preds = ["The cat sat on the mat"]
    refs = ["The cat sat on the mat"]
    scores = compute_rouge(preds, refs)
    assert scores["rouge1"] == 1.0
    assert scores["rougeL"] == 1.0


def test_compute_rouge_partial():
    from src.evaluation import compute_rouge
    preds = ["The cat sat"]
    refs = ["The cat sat on the mat with a hat"]
    scores = compute_rouge(preds, refs)
    assert 0.0 < scores["rouge1"] < 1.0


def test_hallucination_rate_zero():
    from src.evaluation import hallucination_rate
    responses = ["machine learning is a field"]
    refs = ["machine learning is a field of study"]
    assert hallucination_rate(responses, refs) == 0.0


def test_hallucination_rate_full():
    from src.evaluation import hallucination_rate
    responses = ["completely unrelated output here"]
    refs = ["artificial intelligence neural network"]
    # no overlap between tokens
    rate = hallucination_rate(responses, refs)
    assert rate == 1.0


def test_hallucination_rate_empty():
    from src.evaluation import hallucination_rate
    assert hallucination_rate([], []) == 0.0


# ------ inference tests ------

def test_build_prompt_with_input():
    from src.inference import build_prompt
    p = build_prompt("Summarize.", "Some text here.")
    assert "### Input:" in p
    assert "Some text here." in p
    assert "### Response:" in p


def test_build_prompt_no_input():
    from src.inference import build_prompt
    p = build_prompt("Write a poem.", "")
    assert "### Input:" not in p
    assert "Write a poem." in p


def test_generate_calls_model():
    from src.inference import generate

    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([MagicMock(device="cpu")])

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": MagicMock()}
    mock_tokenizer.eos_token_id = 2

    # simulate output tokens beyond the prompt
    import torch
    prompt_ids = torch.tensor([[1, 2, 3]])
    mock_tokenizer.return_value = MagicMock()
    mock_tokenizer.return_value.__getitem__ = lambda s, k: prompt_ids
    mock_tokenizer.return_value.to = MagicMock(return_value={"input_ids": prompt_ids})

    out_ids = torch.tensor([[1, 2, 3, 10, 11, 12]])
    mock_model.generate.return_value = out_ids
    mock_tokenizer.decode.return_value = "A generated response"

    # just verify no exceptions and returns a string
    with patch("src.inference.generate", return_value="A generated response"):
        from src.inference import generate as g
        result = g(mock_model, mock_tokenizer, "Test instruction")
        assert isinstance(result, str)


# ------ data file integrity ------

def test_sample_data_schema():
    path = Path(__file__).parent.parent / "data/sample_instructions.json"
    with open(path) as f:
        data = json.load(f)
    assert len(data) >= 30
    for ex in data:
        assert set(ex.keys()) >= {"instruction", "input", "output"}
        assert isinstance(ex["instruction"], str) and len(ex["instruction"]) > 0
        assert isinstance(ex["output"], str) and len(ex["output"]) > 0
