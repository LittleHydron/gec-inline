"""Serve the Gradio demo from Modal on an L4 GPU.

The HF Space (Lopato4ka/gec-inline) runs the same app on the free CPU
tier, where a 3B model takes minutes per request. This deployment serves
the identical UI from Modal instead: requests run on an L4 in seconds,
the container scales to zero after 5 idle minutes (so it costs nothing
while nobody is using it), and the HF cache volume already holds the
base model + adapters from the training runs, keeping cold starts short.

Deploy (from the repo root)::

    modal deploy -m modal_app.serve

The printed URL (https://<workspace>--gec-inline-demo-ui.modal.run) is
the demo link. Cost: ~$0.80/h of L4 time, billed only while warm.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent

SFT_ADAPTER = "Lopato4ka/qwen2.5-3b-gec-sft"
DPO_BASE = "Lopato4ka/qwen2.5-3b-gec-sft-merged"
DPO_ADAPTER = "Lopato4ka/qwen2.5-3b-gec-dpo"

app = modal.App("gec-inline-demo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.45.0",
        "peft>=0.13.0",
        "accelerate>=0.34.0",
        "gradio>=4.44.0",
        "fastapi[standard]",
        "huggingface_hub",
        "hf_transfer",
        "numpy",
        "tqdm",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "GEC_SFT_ADAPTER": SFT_ADAPTER,
        "GEC_DPO_BASE_MODEL": DPO_BASE,
        "GEC_DPO_ADAPTER": DPO_ADAPTER,
    })
    .add_local_python_source("gec")
    # app.py doubles as the HF Space entrypoint; mount it under a
    # non-clashing module path and import it inside the function.
    .add_local_file(REPO_ROOT / "app.py", "/root/gec_demo_app.py")
)

# Re-use the training pipeline's HF cache — base model and adapters are
# already in it, so cold starts skip the multi-GB downloads.
hf_cache_vol = modal.Volume.from_name("gec-inline-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    scaledown_window=300,   # scale to zero after 5 idle minutes
    timeout=900,
    max_containers=1,       # one L4 max — caps worst-case spend at ~$0.80/h
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def ui():
    import importlib.util

    import gradio as gr
    from fastapi import FastAPI

    spec = importlib.util.spec_from_file_location("gec_demo_app", "/root/gec_demo_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return gr.mount_gradio_app(FastAPI(), mod.build_ui().queue(), path="/")
