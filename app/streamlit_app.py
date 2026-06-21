"""Streamlit demo for MicroGPT: type a prompt, tune sampling, see generated text."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch

# allow `streamlit run app/streamlit_app.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microgpt.config import TrainConfig  # noqa: E402
from microgpt.device import get_device  # noqa: E402
from microgpt.generate import generate_text, load_model  # noqa: E402

st.set_page_config(page_title="MicroGPT", page_icon="🧠", layout="centered")
st.title("🧠 MicroGPT")
st.caption("A GPT-style transformer trained from scratch.")


@st.cache_resource
def _load():
    device = get_device()
    model, tokenizer = load_model(device)
    return model, tokenizer, device


cfg = TrainConfig()

if not cfg.ckpt_path.exists():
    st.warning(
        "No trained checkpoint found. Train a model first:\n\n"
        "```\npython scripts/download_data.py\npython -m microgpt.train\n```"
    )
    st.stop()

model, tokenizer, device = _load()
st.success(f"Model loaded on **{device.type}** — {model.num_params() / 1e6:.2f}M params")

with st.sidebar:
    st.header("Sampling")
    max_new_tokens = st.slider("Max new tokens", 16, 1000, 300, 16)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.05)
    top_k = st.slider("Top-k", 1, 200, 50, 1)

prompt = st.text_area("Prompt", value="The history of", height=120)

if st.button("Generate", type="primary"):
    with st.spinner("Generating ..."):
        with torch.no_grad():
            text = generate_text(
                model, tokenizer, prompt, max_new_tokens, temperature, top_k, device
            )
    st.subheader("Output")
    st.write(text)

if cfg.loss_curve_path.exists():
    st.divider()
    st.subheader("Training loss")
    st.image(str(cfg.loss_curve_path))
