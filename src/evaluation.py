import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 4,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    total_nll, total_tokens = 0.0, 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])

        # out.loss is mean NLL over non-padding tokens
        n_tokens = enc["attention_mask"].sum().item()
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens

    return math.exp(total_nll / total_tokens)


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("pip install rouge-score")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in agg:
            agg[k].append(scores[k].fmeasure)

    return {k: round(float(np.mean(v)), 4) for k, v in agg.items()}


def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 200,
    temperature: float = 0.1,
) -> List[str]:
    model.eval()
    device = next(model.parameters()).device
    responses = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        # strip the prompt tokens
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return responses


def hallucination_rate(responses: List[str], ground_truths: List[str]) -> float:
    """Proxy: fraction of responses with no token overlap with reference (rough upper bound)."""
    count = 0
    for resp, ref in zip(responses, ground_truths):
        resp_toks = set(resp.lower().split())
        ref_toks = set(ref.lower().split())
        if len(resp_toks & ref_toks) == 0:
            count += 1
    return round(count / len(responses), 4) if responses else 0.0


def run_evaluation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_examples: List[Dict[str, str]],
    max_new_tokens: int = 200,
) -> Dict[str, Any]:
    from src.dataset import format_prompt, ALPACA_PROMPT_NO_INPUT

    # build prompt-only strings (no output appended) for generation
    prompts = []
    for ex in eval_examples:
        text = format_prompt(ex)
        # strip the expected output so model generates it
        cut = text.find("### Response:\n") + len("### Response:\n")
        prompts.append(text[:cut])

    references = [e["output"] for e in eval_examples]

    print(f"[Eval] generating {len(prompts)} responses...")
    preds = generate_responses(model, tokenizer, prompts, max_new_tokens)

    rouge = compute_rouge(preds, references)
    halluc = hallucination_rate(preds, references)

    return {
        **rouge,
        "hallucination_rate": halluc,
        "n_samples": len(eval_examples),
    }
