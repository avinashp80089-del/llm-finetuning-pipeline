# LLM Fine-Tuning Pipeline

End-to-end pipeline for instruction fine-tuning of large language models using LoRA and QLoRA. Trains on Alpaca-format datasets, tracks experiments with Weights & Biases, and evaluates with ROUGE and perplexity.

## Architecture

```
data/sample_instructions.json
         │
         ▼
   src/dataset.py          ← Alpaca prompt formatting + tokenization
         │
         ▼
   src/lora_config.py      ← LoRA / QLoRA / BnB config dataclasses
         │
         ▼
   src/trainer.py          ← HuggingFace Trainer + W&B logging
         │
         ▼
   outputs/lora_adapter/   ← saved adapter weights
         │
         ▼
   src/evaluation.py       ← ROUGE, perplexity, hallucination rate
         │
         ▼
   src/inference.py        ← merged-weight inference pipeline
```

## Quick Start

```bash
pip install -r requirements.txt

# fine-tune Phi-2 with LoRA
python train.py \
  --model microsoft/phi-2 \
  --data data/sample_instructions.json \
  --output outputs/phi2_lora \
  --epochs 3 \
  --lora-r 16

# QLoRA (4-bit) on a larger model
python train.py \
  --model meta-llama/Llama-3.2-1B \
  --qlora \
  --bf16 \
  --lora-r 64 \
  --lora-alpha 128 \
  --output outputs/llama_qlora
```

## Inference

```python
from src.inference import FineTunedPipeline

pipe = FineTunedPipeline("outputs/phi2_lora")
response = pipe("Explain gradient descent in simple terms.")
print(response)
```

## Dataset Format

Alpaca-style JSON array:

```json
[
  {
    "instruction": "Explain what a transformer model is.",
    "input": "",
    "output": "A transformer is a neural network..."
  }
]
```

`input` is optional — leave empty for pure instruction-following examples.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| ROUGE-1/2/L | N-gram overlap between generated and reference responses |
| Perplexity | Cross-entropy of model on held-out text (lower = better) |
| Hallucination Rate | Fraction of responses with zero token overlap with reference |

Results saved to `outputs/<run>/eval_metrics.json`.

## W&B Integration

Set `WANDB_API_KEY` in your environment or run `wandb login`. Training metrics, loss curves, and evaluation results stream to your project automatically.

```bash
export WANDB_PROJECT=llm-finetuning
export WANDB_API_KEY=your_key_here
```

## Config Files

Pre-built configs in `configs/`:
- `lora_phi2.yaml` — Phi-2 LoRA (fast, no quantization)
- `qlora_llama.yaml` — Llama 3.2 QLoRA (memory-efficient, 4-bit)

## Project Structure

```
llm-finetuning-pipeline/
├── src/
│   ├── dataset.py       # data loading, formatting, tokenization
│   ├── lora_config.py   # LoRA + QLoRA configuration dataclasses
│   ├── trainer.py       # model loading, PEFT setup, training loop
│   ├── evaluation.py    # ROUGE, perplexity, hallucination metrics
│   └── inference.py     # merged-weight inference pipeline
├── tests/
│   └── test_pipeline.py # unit tests (pytest)
├── configs/
│   ├── lora_phi2.yaml
│   └── qlora_llama.yaml
├── data/
│   └── sample_instructions.json   # 50 Alpaca-format examples
├── train.py             # CLI entry point
└── requirements.txt
```

## Requirements

- Python 3.10+
- CUDA GPU recommended (CPU works for small models)
- ~6GB VRAM for Phi-2 LoRA; ~8GB for QLoRA with 4-bit
