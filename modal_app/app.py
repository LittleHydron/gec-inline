"""Modal app: SFT + DPO training and batch generation for gec-inline.

Replaces the Colab notebooks (``notebooks/01``, ``notebooks/02``) — same
recipe and hyperparameters, but runs detached on Modal GPUs and persists
every artifact (adapters, predictions, DPO pairs) to a ``modal.Volume``,
so nothing is lost to an ephemeral filesystem.

One-time setup::

    pip install modal
    modal setup                                        # browser auth
    modal secret create huggingface HF_TOKEN=hf_xxx    # write-scoped token

Full pipeline (run from the repo root)::

    modal run -m modal_app.app::smoke      # ~3 min end-to-end sanity check
    modal run -m modal_app.app::run_all    # SFT -> DPO pairs -> DPO -> all predictions

Or step by step::

    modal run -m modal_app.app::sft
    modal run -m modal_app.app::dpo        # rebuilds DPO pairs from SFT outputs first
    modal run -m modal_app.app::gen_all    # 4 models x 3 eval sets
    modal volume get gec-inline-results predictions/ results/predictions/

Cost on L4 (~$0.80/h): SFT ~1.5 h, DPO ~0.75 h, generation ~3 GPU-h in
parallel containers — ~$5 total, well inside Modal's $30/month credits.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent

HF_USER = "Lopato4ka"
QWEN_BASE = "Qwen/Qwen2.5-3B-Instruct"
UNSLOTH_BASE_4BIT = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 1024
SEED = 3407

GPU = "L4"  # 24 GB, Ada -> bf16; ~$0.80/h

app = modal.App("gec-inline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "unsloth",          # pins compatible torch / transformers / trl / peft
        "datasets",
        "huggingface_hub",
        "hf_transfer",
        "tqdm",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("gec", "scripts")
    .add_local_dir(REPO_ROOT / "data" / "processed", remote_path="/data")
)

# Persistent storage:
#   /results  — predictions, rebuilt dpo.jsonl, local adapter copies
#   HF cache  — base-model weights shared across containers
results_vol = modal.Volume.from_name("gec-inline-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("gec-inline-hf-cache", create_if_missing=True)

VOLUMES = {
    "/results": results_vol,
    "/root/.cache/huggingface": hf_cache_vol,
}
SECRETS = [modal.Secret.from_name("huggingface")]  # provides HF_TOKEN


def adapter_repo(hf_user: str) -> str:
    return f"{hf_user}/qwen2.5-3b-gec-sft"


def merged_repo(hf_user: str) -> str:
    return f"{hf_user}/qwen2.5-3b-gec-sft-merged"


def dpo_repo(hf_user: str) -> str:
    return f"{hf_user}/qwen2.5-3b-gec-dpo"


def _compat_config(cls, **kwargs):
    """Build a TRL config object tolerating cross-version kwarg renames.

    TRL renamed SFTConfig.max_seq_length -> max_length (~0.20); anything the
    installed version doesn't know is dropped with a warning instead of
    crashing a multi-hour training run.
    """
    import dataclasses

    names = {f.name for f in dataclasses.fields(cls)}
    renames = {"max_seq_length": "max_length"}
    out = {}
    for k, v in kwargs.items():
        if k not in names and k in renames and renames[k] in names:
            k = renames[k]
        if k in names:
            out[k] = v
        else:
            print(f"WARNING: {cls.__name__} has no field '{k}' — dropping")
    return cls(**out)


def _compat_trainer(cls, tokenizer, **kwargs):
    """TRL renamed the trainer's tokenizer= kwarg to processing_class=.

    Prefer tokenizer= when the (Unsloth-patched) signature still takes it —
    Unsloth's patch hooks that path to substitute its '<EOS_TOKEN>' config
    sentinel with the real token; processing_class= bypasses the hook.
    """
    import inspect

    params = inspect.signature(cls.__init__).parameters
    key = "tokenizer" if "tokenizer" in params else "processing_class"
    return cls(**{key: tokenizer}, **kwargs)


def _fresh_lora(model):
    """The shared LoRA head recipe (matches notebooks 01/02 and report.md)."""
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )


@app.function(image=image, gpu=GPU, timeout=6 * 3600, volumes=VOLUMES, secrets=SECRETS)
def train_sft(hf_user: str = HF_USER, max_steps: int = 0, push: bool = True) -> dict:
    """SFT stage — mirrors notebooks/01_sft_qwen25_3b.ipynb.

    Pushes BOTH the LoRA adapter ({user}/qwen2.5-3b-gec-sft) and a merged
    16-bit model ({user}/qwen2.5-3b-gec-sft-merged). The merged repo is what
    the DPO stage trains from: reloading it in 4-bit and attaching a fresh
    LoRA makes "adapter disabled" == the SFT policy, which is exactly the
    frozen reference DPO needs.
    """
    # Unsloth MUST be imported before trl/transformers/peft — it patches them
    # at import time; the reverse order leaves the patches half-applied (and
    # its '<EOS_TOKEN>' SFTConfig sentinel never gets substituted).
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    import os

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=UNSLOTH_BASE_4BIT,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    model = _fresh_lora(model)

    raw = load_dataset("json", data_files="/data/train.jsonl", split="train")
    print("train rows:", len(raw))

    def format_example(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}

    ds = raw.map(format_example, remove_columns=raw.column_names)

    use_bf16 = is_bfloat16_supported()
    config = _compat_config(
        SFTConfig,
        output_dir="/results/checkpoints/sft",
        # Unsloth patches SFTConfig with an '<EOS_TOKEN>' sentinel default;
        # give it the real token so TRL's vocabulary check passes.
        eos_token=tokenizer.eos_token,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        num_train_epochs=2,
        max_steps=max_steps or -1,  # >0 only for cheap dry runs
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_8bit",
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        seed=SEED,
        bf16=use_bf16,
        fp16=not use_bf16,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )
    trainer = _compat_trainer(SFTTrainer, tokenizer,
                              model=model, train_dataset=ds, args=config)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    stats = trainer.train()
    print("train metrics:", stats.metrics)

    # Sanity check: the bracketed format should appear after fine-tuning.
    FastLanguageModel.for_inference(model)
    from gec.prompts import build_chat_messages

    for s in ["I goes to school every day .",
              "She have did her homework already ."]:
        prompt = tokenizer.apply_chat_template(
            build_chat_messages(s), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        print(repr(tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                    skip_special_tokens=True)))

    if push:
        token = os.environ["HF_TOKEN"]
        model.push_to_hub(adapter_repo(hf_user), token=token, private=False)
        tokenizer.push_to_hub(adapter_repo(hf_user), token=token, private=False)
        model.push_to_hub_merged(merged_repo(hf_user), tokenizer,
                                 save_method="merged_16bit", token=token, private=False)
        print("pushed:", adapter_repo(hf_user), "and", merged_repo(hf_user))
    else:
        print("push skipped (dry run)")

    results_vol.commit()
    return stats.metrics


@app.function(image=image, gpu=GPU, timeout=3 * 3600, volumes=VOLUMES, secrets=SECRETS)
def generate(config: dict) -> str:
    """Run one model config over one eval JSONL; write predictions to /results.

    config keys:
        eval_file  — path inside the container (e.g. /data/eval_bea_dev.jsonl)
        out_name   — predictions filename (e.g. sft_bea_dev.jsonl)
        base       — HF id of the base model
        adapter    — HF id of a LoRA adapter, or None
        few_shot   — prepend the 3-shot examples (base-model baseline)
        limit      — generate only the first N rows (0 = all)
    """
    import json

    from tqdm import tqdm

    from gec.inference import generate_batch, load_model

    eval_file = config["eval_file"]
    out_path = Path("/results/predictions") / config["out_name"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = config.get("batch_size", 16)

    rows = [json.loads(l) for l in open(eval_file, encoding="utf-8") if l.strip()]
    if config.get("limit"):
        rows = rows[: config["limit"]]
    sentences = [r["source"] for r in rows]
    print(f"{config['out_name']}: {len(sentences)} sources from {eval_file}")

    tok, model = load_model(config["base"], adapter_id=config.get("adapter"),
                            dtype="bfloat16", device="cuda")

    results = []
    for start in tqdm(range(0, len(sentences), batch_size), desc=config["out_name"]):
        chunk = sentences[start : start + batch_size]
        for r in generate_batch(chunk, tok, model,
                                include_few_shot=config.get("few_shot", False),
                                batch_size=batch_size):
            results.append({"source": r.source, "raw": r.raw,
                            "corrected": r.corrected, "parse_ok": r.parse_ok})

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    results_vol.commit()
    print(f"wrote {len(results)} predictions to {out_path}")
    return str(out_path)


@app.function(image=image, timeout=1800, volumes=VOLUMES, secrets=SECRETS)
def build_dpo_pairs(max_pairs: int = 3000) -> str:
    """Rebuild dpo.jsonl with REAL SFT mistakes as rejections (CPU only).

    Uses /results/predictions/sft_train_subset.jsonl produced by the
    `generate` step; sentences where the SFT output matches gold fall back
    to corruption / cross-sentence noise (scripts/build_dpo_pairs.py).
    """
    import subprocess

    cmd = [
        "python", "-m", "scripts.build_dpo_pairs",
        "--train", "/data/train.jsonl",
        "--sft-preds", "/results/predictions/sft_train_subset.jsonl",
        "--out", "/results/dpo.jsonl",
        "--max-pairs", str(max_pairs),
        "--seed", str(SEED),
    ]
    subprocess.run(cmd, check=True, cwd="/root")
    results_vol.commit()
    return "/results/dpo.jsonl"


@app.function(image=image, gpu=GPU, timeout=4 * 3600, volumes=VOLUMES, secrets=SECRETS)
def train_dpo(hf_user: str = HF_USER) -> dict:
    """DPO bonus stage — mirrors notebooks/02_dpo_qwen25_3b.ipynb.

    Loads the merged 16-bit SFT model in 4-bit, attaches a FRESH LoRA and
    trains it with DPO (ref_model=None => TRL disables the adapter for the
    reference forward pass, i.e. the reference policy is the SFT model).
    """
    # Unsloth first — see the comment in train_sft.
    from unsloth import FastLanguageModel, is_bfloat16_supported

    import os

    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    from gec.prompts import SYSTEM_PROMPT, build_user_message

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=merged_repo(hf_user),
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    model = _fresh_lora(model)

    # Prefer the Modal-rebuilt pairs (real SFT rejections); fall back to the
    # corruption-only file committed in the repo.
    pairs_file = "/results/dpo.jsonl"
    if not Path(pairs_file).exists():
        pairs_file = "/data/dpo.jsonl"
        print("WARNING: /results/dpo.jsonl not found - using corruption-only repo pairs")
    print("DPO pairs:", pairs_file)

    def render(ex):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(ex["source"])},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "chosen": ex["chosen"], "rejected": ex["rejected"]}

    raw = load_dataset("json", data_files=pairs_file, split="train")
    ds = raw.map(render, remove_columns=raw.column_names)
    print("DPO rows:", len(ds))

    use_bf16 = is_bfloat16_supported()
    config = _compat_config(
        DPOConfig,
        output_dir="/results/checkpoints/dpo",
        eos_token=tokenizer.eos_token,  # harmless if DPOConfig lacks the field
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        num_train_epochs=1,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        optim="adamw_8bit",
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        seed=SEED,
        bf16=use_bf16,
        fp16=not use_bf16,
        beta=0.1,
        max_length=MAX_SEQ_LEN,
        max_prompt_length=512,
        report_to="none",
    )
    trainer = _compat_trainer(
        DPOTrainer, tokenizer,
        model=model,
        ref_model=None,  # adapter-disabled forward pass == SFT reference policy
        args=config,
        train_dataset=ds,
    )
    stats = trainer.train()
    print("train metrics:", stats.metrics)

    token = os.environ["HF_TOKEN"]
    model.push_to_hub(dpo_repo(hf_user), token=token, private=False)
    tokenizer.push_to_hub(dpo_repo(hf_user), token=token, private=False)
    print("pushed:", dpo_repo(hf_user))

    results_vol.commit()
    return stats.metrics


# ---------------------------------------------------------------------------
# Eval-prediction configs: 4 model variants x 3 eval sets.
# Note the DPO adapter sits on top of the MERGED SFT base, not the raw Qwen.
# ---------------------------------------------------------------------------

EVAL_FILES = {
    "bea_dev": "/data/eval_bea_dev.jsonl",
    "jfleg_dev": "/data/eval_jfleg_dev.jsonl",
    "jfleg_test": "/data/eval_jfleg_test.jsonl",
}


def model_variants(hf_user: str) -> dict[str, dict]:
    return {
        "base_zeroshot": {"base": QWEN_BASE, "adapter": None, "few_shot": False},
        "base_fewshot": {"base": QWEN_BASE, "adapter": None, "few_shot": True},
        "sft": {"base": QWEN_BASE, "adapter": adapter_repo(hf_user), "few_shot": False},
        "dpo": {"base": merged_repo(hf_user), "adapter": dpo_repo(hf_user), "few_shot": False},
    }


def eval_configs(hf_user: str, variants: list[str] | None = None,
                 limit: int = 0) -> list[dict]:
    out = []
    for vname, v in model_variants(hf_user).items():
        if variants and vname not in variants:
            continue
        for ename, efile in EVAL_FILES.items():
            out.append({"eval_file": efile, "out_name": f"{vname}_{ename}.jsonl",
                        "limit": limit, **v})
    return out


TRAIN_SUBSET_CONFIG = {
    # SFT predictions on (a subset of) the training sources -> DPO rejections.
    "eval_file": "/data/train.jsonl",
    "out_name": "sft_train_subset.jsonl",
    "base": QWEN_BASE,
    "few_shot": False,
    "limit": 4000,
}


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def smoke(hf_user: str = HF_USER):
    """Cheap end-to-end check: base model + few-shot on 8 BEA-dev sentences."""
    cfg = {"eval_file": EVAL_FILES["bea_dev"], "out_name": "smoke_base_fewshot.jsonl",
           "base": QWEN_BASE, "adapter": None, "few_shot": True,
           "limit": 8, "batch_size": 8}
    print(generate.remote(cfg))
    print("Smoke OK. Pull with: modal volume get gec-inline-results "
          "predictions/smoke_base_fewshot.jsonl results/predictions/")


@app.local_entrypoint()
def sft(hf_user: str = HF_USER, max_steps: int = 0, no_push: bool = False):
    """Full SFT by default; --max-steps 2 --no-push for a cheap dry run."""
    print(train_sft.remote(hf_user, max_steps, not no_push))


@app.local_entrypoint()
def dpo(hf_user: str = HF_USER, reuse_pairs: bool = False,
        skip_subset_gen: bool = False):
    """Generate SFT-on-train predictions, rebuild DPO pairs, then train DPO.

    --skip-subset-gen reuses the existing sft_train_subset.jsonl in the
    volume (e.g. when only the pair-building logic changed);
    --reuse-pairs skips straight to training.
    """
    if not reuse_pairs:
        if not skip_subset_gen:
            generate.remote({**TRAIN_SUBSET_CONFIG, "adapter": adapter_repo(hf_user)})
        build_dpo_pairs.remote()
    print(train_dpo.remote(hf_user))


@app.local_entrypoint()
def gen_all(hf_user: str = HF_USER, variants: str = "", limit: int = 0):
    """All eval predictions in parallel containers.

    --variants 'sft,dpo' restricts to a comma-separated subset of
    {base_zeroshot, base_fewshot, sft, dpo}.
    """
    wanted = [v for v in variants.split(",") if v] or None
    configs = eval_configs(hf_user, wanted, limit)
    for path in generate.map(configs):
        print("done:", path)
    print("Pull everything with: modal volume get gec-inline-results "
          "predictions/ results/predictions/")


@app.local_entrypoint()
def run_all(hf_user: str = HF_USER):
    """The whole pipeline. Base-model predictions run in parallel with SFT."""
    base_calls = [generate.spawn(c)
                  for c in eval_configs(hf_user, ["base_zeroshot", "base_fewshot"])]

    train_sft.remote(hf_user)
    generate.remote({**TRAIN_SUBSET_CONFIG, "adapter": adapter_repo(hf_user)})
    build_dpo_pairs.remote()
    train_dpo.remote(hf_user)

    ft_calls = [generate.spawn(c) for c in eval_configs(hf_user, ["sft", "dpo"])]
    for call in base_calls + ft_calls:
        print("done:", call.get())

    print("All artifacts in the gec-inline-results volume. Pull with:\n"
          "  modal volume get gec-inline-results predictions/ results/predictions/\n"
          "  modal volume get gec-inline-results dpo.jsonl data/processed/dpo.modal.jsonl")
