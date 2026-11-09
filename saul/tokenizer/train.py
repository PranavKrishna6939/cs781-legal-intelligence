"""Legal byte-level BPE (report_saul.md 6).

* GPT-2-style pre-tokenization: only a leading space may attach to a word; letters, digits and punctuation never
  share a pre-token, so every letter/punctuation transition is a token boundary (strict Phase-2 spans).
* Single-digit splitting (`\\p{N}` without `+`): legal numbers are identifiers (years, sections, citations).
* No normalizer (NFKC/lowercasing would shift Phase-2 character offsets); cased.
* ByteLevel post-processor with trim_offsets=True so offsets exclude the leading space.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Iterator

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from saul.tokenizer.specials import BOS, EOS, MASK, PAD, zero_byte_specials

# Punctuation is one character, not `+`: a `+` merges ".[" / ")." into one pre-token and Phase-2
# citation spans, which end on the bracket, stop aligning.
GPT2_SINGLE_DIGIT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]|\s+(?!\S)|\s+"""
# cl100k-style (one punctuation char may glue to the next word) with single digits -- ablation only
CL100K_SINGLE_DIGIT = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)
REGEXES = {"gpt2": GPT2_SINGLE_DIGIT, "cl100k": CL100K_SINGLE_DIGIT}


def build_empty_tokenizer(regex: str = GPT2_SINGLE_DIGIT) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.normalizer = None
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(regex), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=True)
    return tok


def train_bpe(texts: Iterable[str], vocab_size: int, specials: list[str], min_frequency: int = 5,
              regex: str = GPT2_SINGLE_DIGIT, length: int | None = None) -> Tokenizer:
    if vocab_size < len(specials) + 256:
        raise ValueError(
            f"vocab_size {vocab_size} is smaller than {len(specials)} specials + 256 byte tokens"
        )
    tok = build_empty_tokenizer(regex)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[AddedToken(s, special=True, normalized=False) for s in specials],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer, length=length)
    # guarantee the exact vocab size (tensor-core friendly, and the model config depends on it)
    cur = tok.get_vocab_size(with_added_tokens=True)
    if cur < vocab_size:
        extra = [AddedToken(f"<|fill_{i}|>", special=True, normalized=False) for i in range(vocab_size - cur)]
        tok.add_special_tokens(extra)
    assert tok.get_vocab_size(with_added_tokens=True) == vocab_size, tok.get_vocab_size(with_added_tokens=True)
    for i, s in enumerate(specials):
        assert tok.token_to_id(s) == i, f"special {s} expected id {i}, got {tok.token_to_id(s)}"
    return tok


def to_hf(tok: Tokenizer):
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=BOS, eos_token=EOS, pad_token=PAD, mask_token=MASK,
        clean_up_tokenization_spaces=False,
        padding_side="left",
    )


def save_tokenizer(tok: Tokenizer, out_dir: str, specials: list[str], extra_meta: dict | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    to_hf(tok).save_pretrained(out_dir)
    meta = {"specials": specials, "vocab_size": tok.get_vocab_size(True), **(extra_meta or {})}
    with open(os.path.join(out_dir, "saul_tokenizer_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_hf_tokenizer(path: str):
    from transformers import PreTrainedTokenizerFast

    t = PreTrainedTokenizerFast.from_pretrained(path)
    t.clean_up_tokenization_spaces = False
    return t


def load_raw_tokenizer(path: str) -> Tokenizer:
    p = path if path.endswith(".json") else os.path.join(path, "tokenizer.json")
    return Tokenizer.from_file(p)


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-to-unicode map. Inlined so this does not depend on a transformers internal."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def token_byte_lengths(tok: Tokenizer, specials: list[str]) -> list[int]:
    """UTF-8 byte length of every token id (for bits-per-byte). Control specials count 0 bytes."""
    inv = {v: k for k, v in _bytes_to_unicode().items()}
    zero = zero_byte_specials(specials)
    vocab = tok.get_vocab(with_added_tokens=True)
    out = [0] * tok.get_vocab_size(True)
    added = {t.content for t in tok.get_added_tokens_decoder().values()} if hasattr(tok, "get_added_tokens_decoder") else set()
    for s, i in vocab.items():
        if s in zero:
            out[i] = 0
        elif s in added or s in specials:
            out[i] = len(s.encode("utf-8"))  # anonymization tags appear literally in the text
        else:
            out[i] = sum(1 for ch in s if ch in inv)
    return out


def sample_texts(shard_dirs: dict[str, list[str]], byte_targets: dict[str, float], source_bytes: dict[str, int],
                 seed_salt: str = "tokenizer") -> Iterator[str]:
    """Deterministically sample documents per source to approximately hit `byte_targets`."""
    from saul.utils.io import iter_jsonl
    from saul.utils.seed import stable_unit

    for src, paths in shard_dirs.items():
        p = min(1.0, byte_targets[src] / max(source_bytes[src], 1))
        for rec in iter_jsonl(paths):
            if stable_unit(rec["id"], seed_salt) < p:
                yield rec["text"]
