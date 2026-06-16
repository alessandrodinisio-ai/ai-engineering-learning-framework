# Building a Tokenizer from Scratch

> Lesson 01 gave you a toy. This lesson gives you a weapon.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Time:** ~90 minutes

## Learning Objectives

- Build a production-grade BPE tokenizer that handles Unicode, whitespace normalization, and special tokens
- Implement byte-level fallback so the tokenizer can encode any input (including emoji, CJK characters, and code) without unknown tokens
- Add pre-tokenization regex patterns that split text at word boundaries before applying BPE merges
- Train a custom tokenizer on a corpus and compare its compression ratio against tiktoken on multilingual text

## The Problem

The BPE tokenizer you wrote in Lesson 01 works on English text. Now throw Japanese at it. Or emoji. Or a block of Python code with mixed tabs and spaces.

It breaks.

Not because BPE is wrong — because the implementation is incomplete. A production tokenizer handles raw bytes of arbitrary encoding, normalizes Unicode before splitting, manages special tokens that should never be merged, chains pre-tokenization with subword splitting, and runs fast enough to not bottleneck a training pipeline processing 15 trillion tokens.

GPT-2's tokenizer has 50,257 tokens. Llama 3 has 128,256. GPT-4 has roughly 100,000. These aren't toy numbers. The merge tables behind these vocabularies were trained on hundreds of gigabytes of text, and the surrounding machinery — normalization, pre-tokenization, special token injection, chat template formatting — is what separates a tokenizer that handles "hello world" from one that handles the entire internet.

You're going to build that machinery.

## The Concept

### The full pipeline

A production tokenizer isn't a single algorithm. It's a pipeline of five stages, each solving a different problem.

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

Each stage has a specific responsibility:

| Stage | What it does | Why it matters |
|-------|-------------|----------------|
| Normalize | NFKC Unicode, optional lowercasing, optional accent removal | The "fi" ligature (U+FB01) becomes "fi" (two characters). Without this, the same word gets different tokens. |
| Pre-Tokenize | Split text into chunks before BPE | Prevents BPE from merging across word boundaries. "the cat" should never produce an "e c" token. |
| BPE Merge | Apply learned merge rules to byte sequences | The core compression. Turns raw bytes into subword tokens. |
| Special Tokens | Inject [BOS], [EOS], [PAD], chat template markers | These tokens have fixed IDs. They never participate in BPE merges. The model relies on them for structure. |
| ID Mapping | Convert token strings to integer IDs | The model sees integers, not strings. |

### Byte-level BPE

The Lesson 01 tokenizer works on UTF-8 bytes. That choice is correct. But we skipped something important: what happens when those bytes aren't valid UTF-8?

Byte-level BPE solves this by treating every possible byte value (0-255) as a valid token. Your base vocabulary is exactly 256 entries. Any file — text, binary, corrupted — can be tokenized without unknown tokens.

GPT-2 added a cosmetic trick: mapping each byte to a printable Unicode character to keep the vocabulary human-readable. In its mapping, byte 0x20 (space) becomes character "G". This is purely aesthetic. The algorithm doesn't care.

The real power: byte-level BPE handles every language on earth. Chinese characters take 3 UTF-8 bytes each. Japanese can be 3-4 bytes. Arabic, Devanagari, emoji — all just byte sequences. The BPE algorithm finds patterns in these byte sequences the same way it finds patterns in English ASCII bytes.

### Pre-tokenization

Before BPE touches your text, you need to split it into chunks. This prevents the merge algorithm from creating tokens that span word boundaries.

GPT-2 uses a regex pattern to split text:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

This pattern splits at contractions ("don't" becomes "don" + "'t"), at words with optional leading spaces, at numbers, at punctuation, and at whitespace. Leading spaces are kept on the word — so "the cat" becomes [" the", " cat"], not ["the", " ", "cat"].

Llama uses SentencePiece, which skips regex entirely. It treats the raw byte stream as one long sequence and lets BPE figure out boundaries itself. This is simpler but gives BPE more freedom to create cross-word tokens.

The choice matters. GPT-2's regex prevents the tokenizer from learning that "the end of one word and the start of another should merge." SentencePiece allows this, sometimes producing more efficient compression but less interpretable tokens.

### Special tokens

Every production tokenizer reserves some token IDs for structural markers:

| Token | Purpose | Who uses it |
|-------|---------|---------|
| `[BOS]` / `<s>` | Beginning of sequence | Llama 3, GPT |
| `[EOS]` / `</s>` | End of sequence | All models |
| `[PAD]` | Padding for batch alignment | BERT, T5 |
| `[UNK]` | Unknown token (byte-level BPE eliminates it) | BERT, WordPiece |
| `<\|im_start\|>` | Chat message boundary start | ChatGPT, Qwen |
| `<\|im_end\|>` | Chat message boundary end | ChatGPT, Qwen |
| `<\|user\|>` | User turn marker | Llama 3 |
| `<\|assistant\|>` | Assistant turn marker | Llama 3 |

Special tokens are never split by BPE. They're matched exactly before the merge algorithm runs, replaced with their fixed IDs, and the surrounding text is tokenized normally.

### Chat templates

This is where most people get confused and most implementations go wrong.

When you send a message to a chat model, the API accepts a list of messages:

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

The model doesn't see JSON. It sees a flat sequence of tokens. The chat template converts messages to that flat sequence using special tokens. Each model does it differently:

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

Get the template wrong and the model produces garbage. It was trained on an exact format. Any deviation — a missing newline, a wrong token, an extra space — pushes the input out of training distribution.

### Speed

Python is too slow for production tokenization.

tiktoken (OpenAI) is written in Rust with Python bindings. HuggingFace tokenizers is also Rust. SentencePiece is C++. These achieve 10-100x speedups over pure Python.

For perspective: tokenizing 15 trillion tokens for Llama 3 pre-training at 1 million tokens/sec (fast Python) takes 174 days. At 100 million tokens/sec (Rust), it takes 1.7 days.

You build in Python to understand the algorithm. In production, you use a compiled implementation and only touch the Python wrapper.

## Build It

### Step 1: Byte-level encoding

The foundation. Convert any string to a byte sequence, map each byte to a printable character for display, and reverse the process.

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

Test on multilingual text to see byte counts:

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

`"hello"` is 5 bytes. `"你好"` is 6 bytes (3 bytes per character). The fire emoji is 4 bytes. A byte-level tokenizer doesn't care what language it is. Bytes are bytes.

### Step 2: Pre-tokenizer with regex

Use GPT-2's regex pattern to split text into chunks. Each chunk is tokenized by BPE independently.

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

The `regex` module supports Unicode property escapes (`\p{L}` for letters, `\p{N}` for numbers). The standard `re` module doesn't, so we fall back to ASCII character classes. Install `regex` for production multilingual tokenizers.

Try it:

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

Leading spaces stay on the word. Contractions split at the apostrophe. Punctuation becomes its own chunk. BPE will never merge tokens across these boundaries.

### Step 3: BPE on byte sequences

The core algorithm from Lesson 01, but now running on pre-tokenized chunks independently.

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Step 4: Special token handling

Special tokens need exact matching and fixed IDs. They bypass BPE entirely.

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Step 5: The complete tokenizer class

Wire everything together: normalize, split by special tokens, pre-tokenize, BPE merge, map to IDs.

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### Step 6: Multilingual test

The real test. Throw English, Chinese, emoji, and code at it.

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

Chinese characters produce 3 bytes each. Emoji produces 4 bytes. None of these break the tokenizer. None produce unknown tokens. That's the power of byte-level BPE.

## Use It

### Comparing real tokenizers

Load Llama 3, GPT-4, and Mistral's real tokenizers. See how each handles the same multilingual text.

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

Same text, different token counts. Llama 3's 128K vocabulary merges common patterns more aggressively. GPT-4's 100K is in between. Mistral's 32K produces more tokens but has a smaller embedding layer.

The tradeoff is always the same: larger vocabulary means shorter sequences but more parameters.

## Ship It

This lesson produces a prompt for building and debugging production tokenizers. See `outputs/prompt-tokenizer-builder.md`.

## Exercises

1. **Easy:** Add a `get_token_bytes(id)` method that shows the raw bytes for any token ID. Use it to inspect what your most common merged tokens actually represent.
2. **Medium:** Implement a Llama-style pre-tokenizer that splits on whitespace and digits but preserves leading spaces. Compare its vocabulary against the GPT-2 regex approach on the same corpus.
3. **Hard:** Add a chat template method that takes a list of `{"role": ..., "content": ...}` messages and produces the correct token sequence in Llama 3 chat format. Test it against HuggingFace's implementation.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|----------------------|
| Byte-level BPE | "tokenizer that works on bytes" | BPE with a base vocabulary of 256 byte values — handles any input without unknown tokens |
| Pre-tokenization | "splitting before BPE" | Regex or rule-based splitting that prevents BPE from merging across word boundaries |
| NFKC normalization | "Unicode cleanup" | Canonical decomposition followed by compatibility composition — "fi" ligature becomes "fi", fullwidth "Ａ" becomes "A" |
| Chat template | "how messages become tokens" | The exact format that converts role/content message lists into flat token sequences — differs per model and must match training format |
| Special tokens | "control tokens" | Reserved token IDs that bypass BPE — [BOS], [EOS], [PAD], chat markers — matched exactly before merging |
| Fertility | "tokens per word" | Ratio of output tokens to input words — GPT-4 is 1.3 for English, 2-3 for Korean, higher means more context wasted |
| tiktoken | "OpenAI's tokenizer" | Rust BPE implementation with Python bindings — 10-100x faster than pure Python |
| Merge table | "that vocabulary" | The ordered list of byte-pair merges learned during training — it is the tokenizer's learned knowledge |

## Further Reading

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- Rust BPE implementation used by GPT-3.5/4
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- Rust tokenizer library supporting BPE, WordPiece, Unigram
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- 128K vocabulary and tokenizer training details
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- Language-agnostic tokenization
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- The original byte-to-Unicode mapping
