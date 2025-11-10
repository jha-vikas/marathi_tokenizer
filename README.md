## Marathi Tokenizer

Byte-level BPE tokenizer training script and deployment assets for a Marathi text corpus.

### Contents
- `tokenizer.py`: trains the tokenizer and writes Hugging Face-compatible artifacts in `artifacts/tokenizer/`.
- `artifacts/tokenizer/`: generated files (`tokenizer.json`, `vocab.json`, `merges.txt`, etc.).
- `app.py`: Gradio demo for Hugging Face Spaces.
- `requirements.txt`: runtime dependencies for the Space.

### Training & Artifact Generation
1. Place your raw corpus at `raw_text/input.txt`.
2. Run the trainer:
   ```bash
   python tokenizer.py
   ```
3. Inspect the summary output (compression ratio, final vocabulary size). Artifacts land in `artifacts/tokenizer/`.

### Running the Demo Locally
```bash
pip install -r requirements.txt
python app.py
```
Visit the provided localhost URL to test encoding/decoding.

### Deploying to Hugging Face Spaces
1. Create a new Space (Gradio SDK).
2. Clone the Space repo and copy:
   - `app.py`
   - `requirements.txt`
   - the entire `artifacts/tokenizer/` folder
   - this `README.md` (optional but recommended)
3. Commit & push. The Space will build and expose the demo UI automatically.

### Using the Tokenizer Programmatically
```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained("path/to/artifacts/tokenizer")
ids = tokenizer.encode("नमस्कार", add_special_tokens=False)
text = tokenizer.decode(ids)
```

