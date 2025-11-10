"""
Gradio demo for the Marathi byte-level BPE tokenizer.

This Space-style app loads the tokenizer artifacts produced by `tokenizer.py`
and exposes two simple utilities:
  1. Encode Marathi (or general UTF-8) text to token IDs and token strings.
  2. Decode a comma-separated list of IDs back into text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import gradio as gr
from transformers import PreTrainedTokenizerFast


TOKENIZER_DIR = Path(__file__).parent / "artifacts" / "tokenizer"
TOKENIZER_JSON = TOKENIZER_DIR / "tokenizer.json"
SPECIAL_TOKENS_MAP = TOKENIZER_DIR / "special_tokens_map.json"

if not TOKENIZER_JSON.exists():
    raise FileNotFoundError(
        f"Tokenizer JSON not found at {TOKENIZER_JSON}. "
        "Run tokenizer.py to generate the artifacts before launching the app."
    )

tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER_JSON))

if SPECIAL_TOKENS_MAP.exists():
    special_tokens = json.loads(SPECIAL_TOKENS_MAP.read_text(encoding="utf-8"))
    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": list(special_tokens.values())})


def encode_text(text: str) -> Tuple[List[int], List[str]]:
    if not text.strip():
        return [], []
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return token_ids, tokens


def decode_tokens(token_ids_str: str) -> str:
    if not token_ids_str.strip():
        return ""
    try:
        token_ids = [int(tok.strip()) for tok in token_ids_str.split(",") if tok.strip()]
    except ValueError as exc:
        raise gr.Error("Token IDs must be integers separated by commas.") from exc
    return tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)


with gr.Blocks(title="Marathi Tokenizer") as demo:
    gr.Markdown(
        """
        # Marathi Byte-Level BPE Tokenizer

        This demo loads the tokenizer learned from the Maithili/Marathi corpus and lets you:

        - Encode text into token IDs.
        - Inspect the corresponding token strings.
        - Decode token ID sequences back to text.
        """
    )

    with gr.Tab("Encode"):
        input_text = gr.Textbox(label="Input Text", placeholder="Type Marathi text...")
        token_ids_output = gr.JSON(label="Token IDs")
        tokens_output = gr.JSON(label="Token Strings")
        encode_button = gr.Button("Encode")
        encode_button.click(encode_text, inputs=input_text, outputs=[token_ids_output, tokens_output])

    with gr.Tab("Decode"):
        token_ids_input = gr.Textbox(
            label="Token IDs (comma-separated)",
            placeholder="e.g. 286, 512, 299",
        )
        decoded_text = gr.Textbox(label="Decoded Text")
        decode_button = gr.Button("Decode")
        decode_button.click(decode_tokens, inputs=token_ids_input, outputs=decoded_text)


if __name__ == "__main__":
    demo.launch()

