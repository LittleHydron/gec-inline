"""Load a base model (optionally with a LoRA adapter) and generate corrections.

Designed to be importable on:
- a Colab GPU runtime (CUDA, bf16),
- a local CPU/4-bit machine,
- HuggingFace Spaces (ZeroGPU).

For Spaces, wrap :func:`generate_batch` in a function decorated with
``@spaces.GPU`` — the ``model``/``tokenizer`` globals stay on CPU
between calls and are moved by the decorator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .parse import parse_inline
from .prompts import build_chat_messages

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


@dataclass
class GenerationResult:
    source: str
    raw: str             # exactly what the model produced
    corrected: str       # parsed-back plain sentence
    parse_ok: bool


def load_model(
    base_model: str = DEFAULT_BASE_MODEL,
    adapter_id: str | None = None,
    dtype: str = "auto",
    device: str | None = None,
):
    """Load tokenizer + model (+ optional LoRA adapter).

    Memory profile:
      Qwen2.5-3B in bf16: ~6 GB VRAM
      Qwen2.5-3B in 4-bit (bnb): ~2 GB VRAM (set dtype='4bit')
    """
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    kwargs: dict = {}
    if dtype == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = "auto" if dtype == "auto" else getattr(torch, dtype)
        if device:
            kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if adapter_id:
        model = PeftModel.from_pretrained(model, adapter_id)
    model.eval()
    return tok, model


@torch.inference_mode()
def generate_batch(
    sentences: list[str],
    tokenizer,
    model,
    *,
    include_few_shot: bool = False,
    max_new_tokens: int = 192,
    temperature: float = 0.0,
    batch_size: int = 8,
) -> list[GenerationResult]:
    """Run generation on a list of sentences. Greedy decoding by default."""
    device = next(model.parameters()).device
    out: list[GenerationResult] = []

    for batch_start in range(0, len(sentences), batch_size):
        batch = sentences[batch_start : batch_start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_chat_messages(s, include_few_shot=include_few_shot),
                tokenize=False,
                add_generation_prompt=True,
            )
            for s in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        outputs = model.generate(**enc, **gen_kwargs)
        decoded = tokenizer.batch_decode(
            outputs[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
        )
        for src, raw in zip(batch, decoded):
            raw = raw.strip()
            corrected, _, parse_ok = parse_inline(raw)
            if not corrected:
                corrected = src
            out.append(GenerationResult(source=src, raw=raw, corrected=corrected, parse_ok=parse_ok))
    return out


def generate_one(sentence: str, tokenizer, model, **kwargs) -> GenerationResult:
    return generate_batch([sentence], tokenizer, model, **kwargs)[0]


def env_default(name: str, fallback: str) -> str:
    return os.environ.get(name) or fallback
