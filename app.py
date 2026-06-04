"""Gradio UI for the GEC inline-edit fine-tuned model.

Deploys to HuggingFace Spaces on ZeroGPU: the heavy generation function
is wrapped with @spaces.GPU so the model is loaded on a free H100 burst.

Model dropdown lets you switch between:
  - "Base + few-shot"  -> Qwen/Qwen2.5-3B-Instruct with a 3-shot prompt
                          (fair comparison baseline — see report.md)
  - "SFT"              -> base + the SFT LoRA adapter
  - "DPO"              -> merged SFT model + the DPO LoRA adapter
                          (the DPO adapter was trained on top of the merged
                          16-bit SFT model, not the raw base)

Environment variables:
  GEC_BASE_MODEL      default 'Qwen/Qwen2.5-3B-Instruct'
  GEC_SFT_ADAPTER     HF Hub id of the SFT adapter (required for SFT)
  GEC_DPO_ADAPTER     HF Hub id of the DPO adapter (required for DPO)
  GEC_DPO_BASE_MODEL  HF Hub id of the merged SFT model the DPO adapter
                      sits on (required for DPO, e.g.
                      'Lopato4ka/qwen2.5-3b-gec-sft-merged')
"""

from __future__ import annotations

import html
import os
import re
import time
from functools import lru_cache

import gradio as gr

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

from gec.inference import generate_batch, load_model
from gec.parse import parse_inline

INTRO = """\
# ✏️ GEC inline-edit — fine-tuned Qwen2.5-3B

This space corrects English grammar by emitting **inline edits** in the
format `I {goes=>go} to school` — the exact syntax requested in the
LPNLP fine-tuning assignment.

Pick a model variant on the right and paste any English sentence. The
left panel shows the model's raw bracketed output; the right one shows
the cleaned-up corrected sentence after parsing the brackets back out.
"""

EXAMPLES = [
    "I goes to school every day .",
    "She have did her homework already .",
    "The childrens was playing in park yesterday .",
    "He don't likes apples and oranges .",
    "We was supposed to met them at noon ; but they didn't came .",
    "Although she is tired but she continue working .",
]

BASE_MODEL = os.environ.get("GEC_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
SFT_ADAPTER = os.environ.get("GEC_SFT_ADAPTER", "")
DPO_ADAPTER = os.environ.get("GEC_DPO_ADAPTER", "")
DPO_BASE_MODEL = os.environ.get("GEC_DPO_BASE_MODEL", "")

VARIANTS = ["Base + few-shot"]
if SFT_ADAPTER:
    VARIANTS.append("SFT")
if DPO_ADAPTER and DPO_BASE_MODEL:
    VARIANTS.append("DPO")

DEFAULT_VARIANT = "DPO" if "DPO" in VARIANTS else ("SFT" if "SFT" in VARIANTS else VARIANTS[0])


@lru_cache(maxsize=4)
def _load(variant: str):
    """Load (tokenizer, model) lazily. Cached per variant."""
    print(f"[boot] loading variant={variant} …")
    t0 = time.time()
    base, adapter = BASE_MODEL, None
    if variant == "SFT":
        adapter = SFT_ADAPTER
    elif variant == "DPO":
        # The DPO adapter was trained on the merged 16-bit SFT model.
        base, adapter = DPO_BASE_MODEL, DPO_ADAPTER
    tok, model = load_model(base, adapter_id=adapter)
    # On a GPU host (e.g. the Modal demo) move the weights over; on CPU
    # Spaces this is a no-op and on ZeroGPU the @spaces.GPU decorator
    # owns device placement (cuda is not visible at load time there).
    import torch
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"[boot] loaded {variant} in {time.time() - t0:.1f}s")
    return tok, model


_BRACKET_RE = re.compile(r"\{([^{}=]*?)=>([^{}=]*?)\}")


def _highlight_edits(raw: str) -> str:
    """Render the bracketed string as HTML with red strike + green replacement."""
    def repl(m: re.Match) -> str:
        src = html.escape(m.group(1).strip())
        tgt = html.escape(m.group(2).strip())
        if not src:
            return f'<span style="background:#dcfce7;color:#166534;padding:0 2px;border-radius:3px">+{tgt}</span>'
        if not tgt:
            return f'<span style="background:#fee2e2;color:#991b1b;padding:0 2px;border-radius:3px;text-decoration:line-through">{src}</span>'
        return (
            f'<span style="background:#fee2e2;color:#991b1b;padding:0 2px;border-radius:3px;text-decoration:line-through">{src}</span>'
            f'<span style="background:#dcfce7;color:#166534;padding:0 2px;border-radius:3px;margin-left:2px">{tgt}</span>'
        )
    escaped = html.escape(raw)
    # Re-allow the `{x=>y}` markup we control.
    escaped = re.sub(
        r"\{([^{}=]*?)=&gt;([^{}=]*?)\}",
        lambda m: "{" + m.group(1) + "=>" + m.group(2) + "}",
        escaped,
    )
    return _BRACKET_RE.sub(repl, escaped)


def _correct_impl(sentence: str, variant: str) -> tuple[str, str, str]:
    """Run the model and return (highlighted HTML, raw bracketed, cleaned text)."""
    sentence = (sentence or "").strip()
    if not sentence:
        return "_Type a sentence first._", "", ""
    if variant not in VARIANTS:
        return f"_Variant `{variant}` unavailable — check env vars._", "", ""

    tok, model = _load(variant)
    few_shot = variant == "Base + few-shot"
    result = generate_batch([sentence], tok, model, include_few_shot=few_shot, batch_size=1)[0]
    cleaned = result.corrected
    if not result.parse_ok:
        cleaned = f"{cleaned}  ⚠️ (output had unmatched braces)"
    return _highlight_edits(result.raw), result.raw, cleaned


if HAS_SPACES:
    @spaces.GPU(duration=60)
    def correct(sentence: str, variant: str):
        return _correct_impl(sentence, variant)
else:
    def correct(sentence: str, variant: str):
        return _correct_impl(sentence, variant)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="GEC inline-edit") as demo:
        gr.Markdown(INTRO)

        with gr.Row():
            with gr.Column(scale=3):
                sentence = gr.Textbox(
                    label="Sentence to correct",
                    placeholder="I goes to school every day .",
                    lines=2,
                    autofocus=True,
                )
                with gr.Row():
                    correct_btn = gr.Button("Correct", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)

                highlighted = gr.HTML(label="Edits (highlighted)")
                raw_box = gr.Textbox(label="Raw model output (bracketed)", interactive=False)
                clean_box = gr.Textbox(label="Cleaned corrected sentence", interactive=False)

                gr.Examples(EXAMPLES, inputs=sentence, label="Try these")

            with gr.Column(scale=1, min_width=260):
                variant = gr.Dropdown(
                    choices=VARIANTS,
                    value=DEFAULT_VARIANT,
                    label="Model variant",
                )
                gr.Markdown(
                    "**Base model:** "
                    f"`{BASE_MODEL}`\n\n"
                    f"**SFT adapter:** `{SFT_ADAPTER or '(not configured)'}`\n\n"
                    f"**DPO adapter:** `{DPO_ADAPTER or '(not configured)'}`\n\n"
                    "---\n\n"
                    "First request may take 30–60 s while ZeroGPU warms up. "
                    "Subsequent requests within the 60 s window are fast."
                )

        correct_btn.click(correct, inputs=[sentence, variant], outputs=[highlighted, raw_box, clean_box])
        sentence.submit(correct, inputs=[sentence, variant], outputs=[highlighted, raw_box, clean_box])
        clear_btn.click(lambda: ("", "", "", ""), outputs=[sentence, highlighted, raw_box, clean_box])

    return demo


if __name__ == "__main__":
    build_ui().queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        theme=gr.themes.Soft(),
        ssr_mode=False,
    )
