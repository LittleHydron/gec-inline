"""Generate the SFT and DPO Colab notebooks from in-file cell lists.

Run this once to (re-)materialise the .ipynb files:

    python notebooks/_build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text,
    }


def write_nb(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


# ============================================================
# 01_sft_qwen25_3b.ipynb — SFT with Unsloth on Colab T4
# ============================================================

SFT_CELLS = [
    md(
        "# GEC inline-edit fine-tuning — SFT (Qwen2.5-3B-Instruct + Unsloth)\n"
        "\n"
        "**Inputs:** `data/processed/train.jsonl` produced by `scripts/build_dataset.py`.\n"
        "\n"
        "**Output:** a LoRA adapter pushed to `<your-hf-username>/qwen2.5-3b-gec-sft`.\n"
        "\n"
        "Runs on a free Colab T4 (16 GB). Wall time ≈ 45–60 min for 10 k examples × 2 epochs."
    ),
    md("## 1. Install dependencies"),
    code(
        "%%capture\n"
        "!pip install -q unsloth\n"
        "# Unsloth pins compatible torch / xformers / trl / peft — let it manage them.\n"
        "!pip install -q --no-deps trl peft accelerate bitsandbytes\n"
        "!pip install -q datasets huggingface_hub"
    ),
    md(
        "## 2. Fetch the training data\n"
        "Clone the project repo so we can read `data/processed/train.jsonl`. "
        "Replace the URL with your fork if you renamed it."
    ),
    code(
        "import os\n"
        "REPO_URL = os.environ.get('GEC_REPO_URL', 'https://github.com/LittleHydron/gec-inline')\n"
        "!test -d gec-inline || git clone --depth 1 $REPO_URL\n"
        "%cd gec-inline\n"
        "!ls data/processed/"
    ),
    md("## 3. Load Qwen2.5-3B-Instruct in 4-bit with a LoRA head"),
    code(
        "from unsloth import FastLanguageModel\n"
        "\n"
        "MAX_SEQ_LEN = 1024\n"
        "BASE_MODEL = 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit'\n"
        "\n"
        "model, tokenizer = FastLanguageModel.from_pretrained(\n"
        "    model_name = BASE_MODEL,\n"
        "    max_seq_length = MAX_SEQ_LEN,\n"
        "    load_in_4bit = True,\n"
        ")\n"
        "model = FastLanguageModel.get_peft_model(\n"
        "    model,\n"
        "    r = 16,\n"
        "    target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],\n"
        "    lora_alpha = 32,\n"
        "    lora_dropout = 0.0,\n"
        "    bias = 'none',\n"
        "    use_gradient_checkpointing = 'unsloth',\n"
        "    random_state = 3407,\n"
        ")"
    ),
    md(
        "## 4. Format the data with the Qwen chat template\n"
        "Each row's `messages` field already has the system + user + assistant turns. "
        "We render them through `tokenizer.apply_chat_template` and let Unsloth's "
        "`train_on_responses_only` mask the prompt out of the loss."
    ),
    code(
        "from datasets import load_dataset\n"
        "\n"
        "raw = load_dataset('json', data_files='data/processed/train.jsonl', split='train')\n"
        "print('rows:', len(raw))\n"
        "print('example messages:', raw[0]['messages'])\n"
        "\n"
        "def format_example(ex):\n"
        "    text = tokenizer.apply_chat_template(ex['messages'], tokenize=False)\n"
        "    return {'text': text}\n"
        "\n"
        "ds = raw.map(format_example, remove_columns=raw.column_names)\n"
        "print(ds[0]['text'][:600])"
    ),
    md("## 5. Configure the SFT trainer"),
    code(
        "from trl import SFTTrainer, SFTConfig\n"
        "from unsloth import is_bfloat16_supported\n"
        "from unsloth.chat_templates import train_on_responses_only\n"
        "\n"
        "# T4 is Turing -> fp16 only. A100/L4/H100 are Ampere+ -> bf16.\n"
        "USE_BF16 = is_bfloat16_supported()\n"
        "\n"
        "config = SFTConfig(\n"
        "    output_dir = 'outputs/sft',\n"
        "    per_device_train_batch_size = 2,\n"
        "    gradient_accumulation_steps = 4,\n"
        "    warmup_ratio = 0.03,\n"
        "    num_train_epochs = 2,\n"
        "    learning_rate = 2e-4,\n"
        "    lr_scheduler_type = 'cosine',\n"
        "    weight_decay = 0.01,\n"
        "    optim = 'adamw_8bit',\n"
        "    logging_steps = 20,\n"
        "    save_strategy = 'epoch',\n"
        "    save_total_limit = 1,\n"
        "    seed = 3407,\n"
        "    bf16 = USE_BF16,\n"
        "    fp16 = not USE_BF16,\n"
        "    max_seq_length = MAX_SEQ_LEN,\n"
        "    dataset_text_field = 'text',\n"
        "    packing = False,\n"
        "    report_to = 'none',\n"
        ")\n"
        "\n"
        "trainer = SFTTrainer(\n"
        "    model = model,\n"
        "    tokenizer = tokenizer,\n"
        "    train_dataset = ds,\n"
        "    args = config,\n"
        ")\n"
        "\n"
        "# Mask the prompt out of the loss so only the assistant span is trained.\n"
        "# (Tags below are Qwen2.5's chat-template instruction markers.)\n"
        "trainer = train_on_responses_only(\n"
        "    trainer,\n"
        "    instruction_part = '<|im_start|>user\\n',\n"
        "    response_part = '<|im_start|>assistant\\n',\n"
        ")"
    ),
    md("## 6. Train"),
    code(
        "stats = trainer.train()\n"
        "stats.metrics"
    ),
    md(
        "## 7. Sanity check the model\n"
        "Run a couple of sentences through the trained model and confirm "
        "we see the bracketed format."
    ),
    code(
        "from unsloth import FastLanguageModel\n"
        "FastLanguageModel.for_inference(model)\n"
        "\n"
        "from gec.prompts import build_chat_messages\n"
        "\n"
        "for s in ['I goes to school every day .',\n"
        "          'She have did her homework already .',\n"
        "          'The cats was sleeping on the rug .']:\n"
        "    prompt = tokenizer.apply_chat_template(\n"
        "        build_chat_messages(s),\n"
        "        tokenize=False, add_generation_prompt=True,\n"
        "    )\n"
        "    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')\n"
        "    out = model.generate(**inputs, max_new_tokens=128, do_sample=False)\n"
        "    print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))\n"
        "    print('---')"
    ),
    md(
        "## 8. Push the adapter to the HuggingFace Hub\n"
        "Generate an HF access token at https://huggingface.co/settings/tokens "
        "(write scope) and paste it when prompted."
    ),
    code(
        "from huggingface_hub import login\n"
        "login()\n"
        "\n"
        "HF_USER = 'Lopato4ka'  # <- EDIT if you are not Lopato4ka\n"
        "ADAPTER_REPO = f'{HF_USER}/qwen2.5-3b-gec-sft'\n"
        "MERGED_REPO  = f'{HF_USER}/qwen2.5-3b-gec-sft-merged'\n"
        "\n"
        "model.push_to_hub(ADAPTER_REPO, private=False)\n"
        "tokenizer.push_to_hub(ADAPTER_REPO, private=False)\n"
        "# Merged 16-bit copy: the DPO stage loads THIS as its base so that\n"
        "# 'adapter disabled' == the SFT policy (the frozen DPO reference).\n"
        "model.push_to_hub_merged(MERGED_REPO, tokenizer, save_method='merged_16bit', private=False)\n"
        "print('pushed:', ADAPTER_REPO, 'and', MERGED_REPO)"
    ),
    md(
        "## 9. (Optional) Generate predictions on the eval set\n"
        "Doing this here saves you re-loading the model later for `scripts/eval.py`."
    ),
    code(
        "import json, sys\n"
        "from pathlib import Path\n"
        "from tqdm import tqdm\n"
        "from gec.inference import generate_batch\n"
        "\n"
        "EVAL_PATH = 'data/processed/eval_bea_dev.jsonl'\n"
        "OUT_PATH  = 'results/predictions/sft_bea_dev.jsonl'\n"
        "Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "rows = [json.loads(line) for line in open(EVAL_PATH)]\n"
        "sentences = [r['source'] for r in rows]\n"
        "\n"
        "results = []\n"
        "for start in tqdm(range(0, len(sentences), 8)):\n"
        "    batch = sentences[start:start+8]\n"
        "    out = generate_batch(batch, tokenizer, model, batch_size=8)\n"
        "    for r in out:\n"
        "        results.append({'source': r.source, 'raw': r.raw,\n"
        "                        'corrected': r.corrected, 'parse_ok': r.parse_ok})\n"
        "\n"
        "with open(OUT_PATH, 'w') as f:\n"
        "    for r in results:\n"
        "        f.write(json.dumps(r) + '\\n')\n"
        "print('wrote', OUT_PATH, len(results))"
    ),
]


# ============================================================
# 02_dpo_qwen25_3b.ipynb — DPO on top of the SFT adapter
# ============================================================

DPO_CELLS = [
    md(
        "# GEC inline-edit fine-tuning — DPO bonus stage\n"
        "\n"
        "**Inputs:** the SFT adapter from notebook 01 and "
        "`data/processed/dpo.jsonl` (built by `scripts/build_dpo_pairs.py`).\n"
        "\n"
        "**Output:** a DPO-tuned adapter pushed to `<user>/qwen2.5-3b-gec-dpo`.\n"
        "\n"
        "Wall time ≈ 25–35 min on free Colab T4."
    ),
    md("## 1. Install dependencies"),
    code(
        "%%capture\n"
        "!pip install -q unsloth\n"
        "!pip install -q --no-deps trl peft accelerate bitsandbytes\n"
        "!pip install -q datasets huggingface_hub"
    ),
    md("## 2. Clone the project repo"),
    code(
        "import os\n"
        "REPO_URL = os.environ.get('GEC_REPO_URL', 'https://github.com/LittleHydron/gec-inline')\n"
        "!test -d gec-inline || git clone --depth 1 $REPO_URL\n"
        "%cd gec-inline"
    ),
    md(
        "## 3. Load the merged SFT model + a fresh LoRA\n"
        "Notebook 01 pushed a merged 16-bit copy of base+SFT. We reload it "
        "in 4-bit and attach a **new** trainable LoRA, so DPO updates a fresh "
        "adapter while 'adapter disabled' (TRL's `ref_model=None` trick) is "
        "exactly the SFT policy — the frozen reference DPO needs.\n"
        "\n"
        "(Loading the *adapter* repo here instead would fail: Unsloth returns "
        "a PeftModel and `get_peft_model` refuses to stack a second adapter.)"
    ),
    code(
        "from unsloth import FastLanguageModel\n"
        "\n"
        "MAX_SEQ_LEN = 1024\n"
        "HF_USER = 'Lopato4ka'  # <- EDIT if you are not Lopato4ka\n"
        "SFT_MERGED = f'{HF_USER}/qwen2.5-3b-gec-sft-merged'\n"
        "\n"
        "model, tokenizer = FastLanguageModel.from_pretrained(\n"
        "    model_name = SFT_MERGED,\n"
        "    max_seq_length = MAX_SEQ_LEN,\n"
        "    load_in_4bit = True,\n"
        ")\n"
        "model = FastLanguageModel.get_peft_model(\n"
        "    model,\n"
        "    r = 16,\n"
        "    target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],\n"
        "    lora_alpha = 32,\n"
        "    lora_dropout = 0.0,\n"
        "    bias = 'none',\n"
        "    use_gradient_checkpointing = 'unsloth',\n"
        "    random_state = 3407,\n"
        ")"
    ),
    md(
        "## 4. Prepare DPO pairs\n"
        "Each pair has a `prompt` (rendered text), a `chosen` (gold bracketed) "
        "and a `rejected` (corrupted / SFT-mistaken) completion. We re-wrap the "
        "prompt with the Qwen chat template so the model sees the same input "
        "format as during SFT."
    ),
    code(
        "import json\n"
        "from datasets import load_dataset\n"
        "from gec.prompts import SYSTEM_PROMPT, build_user_message\n"
        "\n"
        "def render(ex):\n"
        "    messages = [\n"
        "        {'role': 'system', 'content': SYSTEM_PROMPT},\n"
        "        {'role': 'user',   'content': build_user_message(ex['source'])},\n"
        "    ]\n"
        "    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n"
        "    return {'prompt': prompt, 'chosen': ex['chosen'], 'rejected': ex['rejected']}\n"
        "\n"
        "raw = load_dataset('json', data_files='data/processed/dpo.jsonl', split='train')\n"
        "ds = raw.map(render, remove_columns=raw.column_names)\n"
        "print('rows:', len(ds))\n"
        "print(ds[0])"
    ),
    md("## 5. Configure DPO"),
    code(
        "from trl import DPOTrainer, DPOConfig\n"
        "from unsloth import is_bfloat16_supported\n"
        "\n"
        "USE_BF16 = is_bfloat16_supported()\n"
        "\n"
        "config = DPOConfig(\n"
        "    output_dir = 'outputs/dpo',\n"
        "    per_device_train_batch_size = 1,\n"
        "    gradient_accumulation_steps = 8,\n"
        "    warmup_ratio = 0.05,\n"
        "    num_train_epochs = 1,\n"
        "    learning_rate = 5e-6,\n"
        "    lr_scheduler_type = 'cosine',\n"
        "    weight_decay = 0.0,\n"
        "    optim = 'adamw_8bit',\n"
        "    logging_steps = 20,\n"
        "    save_strategy = 'epoch',\n"
        "    save_total_limit = 1,\n"
        "    seed = 3407,\n"
        "    bf16 = USE_BF16,\n"
        "    fp16 = not USE_BF16,\n"
        "    beta = 0.1,\n"
        "    max_length = MAX_SEQ_LEN,\n"
        "    max_prompt_length = 512,\n"
        "    report_to = 'none',\n"
        ")\n"
        "\n"
        "trainer = DPOTrainer(\n"
        "    model = model,\n"
        "    ref_model = None,  # TRL builds the ref from the model w/ adapter disabled\n"
        "    args = config,\n"
        "    train_dataset = ds,\n"
        "    tokenizer = tokenizer,\n"
        ")"
    ),
    md("## 6. Train"),
    code("stats = trainer.train()\nstats.metrics"),
    md("## 7. Push the DPO adapter to the Hub"),
    code(
        "from huggingface_hub import login\n"
        "login()\n"
        "\n"
        "DPO_REPO = f'{HF_USER}/qwen2.5-3b-gec-dpo'\n"
        "\n"
        "model.push_to_hub(DPO_REPO, private=False)\n"
        "tokenizer.push_to_hub(DPO_REPO, private=False)\n"
        "print('pushed:', DPO_REPO)"
    ),
    md("## 8. Generate eval predictions"),
    code(
        "import json\n"
        "from pathlib import Path\n"
        "from tqdm import tqdm\n"
        "from unsloth import FastLanguageModel\n"
        "from gec.inference import generate_batch\n"
        "\n"
        "FastLanguageModel.for_inference(model)\n"
        "\n"
        "EVAL_PATH = 'data/processed/eval_bea_dev.jsonl'\n"
        "OUT_PATH  = 'results/predictions/dpo_bea_dev.jsonl'\n"
        "Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "rows = [json.loads(line) for line in open(EVAL_PATH)]\n"
        "sentences = [r['source'] for r in rows]\n"
        "results = []\n"
        "for start in tqdm(range(0, len(sentences), 8)):\n"
        "    batch = sentences[start:start+8]\n"
        "    for r in generate_batch(batch, tokenizer, model, batch_size=8):\n"
        "        results.append({'source': r.source, 'raw': r.raw,\n"
        "                        'corrected': r.corrected, 'parse_ok': r.parse_ok})\n"
        "\n"
        "with open(OUT_PATH, 'w') as f:\n"
        "    for r in results:\n"
        "        f.write(json.dumps(r) + '\\n')\n"
        "print('wrote', OUT_PATH, len(results))"
    ),
]


if __name__ == "__main__":
    root = Path(__file__).parent
    write_nb(root / "01_sft_qwen25_3b.ipynb", SFT_CELLS)
    write_nb(root / "02_dpo_qwen25_3b.ipynb", DPO_CELLS)
