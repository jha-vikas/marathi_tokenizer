"""
Simple byte-pair encoding (BPE) tokenizer prototype for Maithili text.

The script reads the novel text, converts the UTF-8 characters into byte IDs,
and then repeatedly merges the most frequent adjacent byte pairs to learn
new tokens. This mirrors the learning stage of a GPT-style tokenizer.
Now extended to emit Hugging Face-compatible artifacts (vocab/merges/tokenizer.json).
"""

import json
from pathlib import Path

RAW_TEXT_PATH = Path("./raw_text/input.txt")
ARTIFACT_DIR = Path("./artifacts/tokenizer")
VOCAB_SIZE = 5500  # the desired final vocabulary size


def bytes_to_unicode():
  """
  Replicates the byte-to-unicode map used by GPT-2 / ByteLevel BPE so every byte
  can be losslessly represented as a unique Unicode character.
  """
  bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
  cs = bs[:]
  n = 0
  for b in range(256):
    if b not in bs:
      bs.append(b)
      cs.append(256 + n)
      n += 1
  cs = [chr(n) for n in cs]
  return dict(zip(bs, cs))


def ensure_artifact_dir(path: Path):
  path.mkdir(parents=True, exist_ok=True)


def write_merges_file(path: Path, merge_order, token_strings):
  lines = ["#version: 0.2"]
  for pair in merge_order:
    left, right = pair
    lines.append(f"{token_strings[left]} {token_strings[right]}")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_vocab_file(path: Path, token_strings):
  vocab = {token_strings[idx]: idx for idx in sorted(token_strings)}
  path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")


def write_tokenizer_json(path: Path, token_strings, merge_order):
  merges_as_strings = [f"{token_strings[left]} {token_strings[right]}" for left, right in merge_order]
  vocab = {token_strings[idx]: idx for idx in sorted(token_strings)}
  tokenizer_json = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": {
      "type": "ByteLevel",
      "add_prefix_space": False,
      "trim_offsets": True,
      "use_regex": True,
    },
    "post_processor": None,
    "decoder": {
      "type": "ByteLevel",
      "add_prefix_space": False,
      "trim_offsets": False,
      "use_regex": True,
    },
    "model": {
      "type": "BPE",
      "dropout": None,
      "unk_token": None,
      "continuing_subword_prefix": "",
      "end_of_word_suffix": "",
      "fuse_unk": False,
      "vocab": vocab,
      "merges": merges_as_strings,
    },
  }
  path.write_text(json.dumps(tokenizer_json, ensure_ascii=False, indent=2), encoding="utf-8")


def write_auxiliary_configs(directory: Path):
  tokenizer_config = {
    "add_prefix_space": False,
    "clean_up_tokenization_spaces": False,
    "model_max_length": 2048,
    "tokenizer_class": "PreTrainedTokenizerFast",
  }
  (directory / "tokenizer_config.json").write_text(
    json.dumps(tokenizer_config, indent=2),
    encoding="utf-8",
  )
  (directory / "special_tokens_map.json").write_text("{}", encoding="utf-8")
  (directory / "added_tokens.json").write_text("[]", encoding="utf-8")


# --- Load and preprocess source text ---------------------------------------------------------
if not RAW_TEXT_PATH.exists():
  raise FileNotFoundError(f"Source text not found at {RAW_TEXT_PATH}")

with RAW_TEXT_PATH.open('r', encoding="utf-8") as f:
  text = f.read()

# Convert characters to raw UTF-8 bytes, then to integers in the range 0-255.
tokens = text.encode('utf-8')
tokens = list(map(int, tokens))

print('----')
print('length: ', len(text))
print('----')
print('length: ', len(tokens))

def get_stats(ids):
    """Count occurrences of every adjacent pair in the current token sequence."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  """
  Replace every occurrence of `pair` in the token list with the newly created token ID.

  Args:
      ids: Current flattened token stream.
      pair: 2-tuple of ints representing the byte/token pair to merge.
      idx:  New token ID assigned to the merged pair.
  """
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

# --- Run BPE merges --------------------------------------------------------------------------
vocab_size = VOCAB_SIZE
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
merge_order = []
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx
  merge_order.append(pair)


print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")

# --- Derive Hugging Face compatible artifacts -----------------------------------------------
byte_encoder = bytes_to_unicode()
token_strings = {i: byte_encoder[i] for i in range(256)}
for i, pair in enumerate(merge_order):
  new_token_id = 256 + i
  left, right = pair
  token_strings[new_token_id] = token_strings[left] + token_strings[right]

ensure_artifact_dir(ARTIFACT_DIR)
write_merges_file(ARTIFACT_DIR / "merges.txt", merge_order, token_strings)
write_vocab_file(ARTIFACT_DIR / "vocab.json", token_strings)
write_tokenizer_json(ARTIFACT_DIR / "tokenizer.json", token_strings, merge_order)
write_auxiliary_configs(ARTIFACT_DIR)

print(f"Hugging Face tokenizer artifacts saved to: {ARTIFACT_DIR.resolve()}")
print(f"final vocabulary size: {len(token_strings)}")

