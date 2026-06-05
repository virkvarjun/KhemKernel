"""A small, dependency-free byte-pair-encoding tokenizer for IUPAC traces.

Why: the original IUPAC tokenizer split on whole words (``acetyloxybenzoic`` is
one token) and mapped anything seen < min_freq to ``<unk>`` — so the model
literally couldn't spell rare names. BPE starts from characters and learns
merges, so every name is representable (zero ``<unk>``) while keeping sequences
much shorter than pure character level.

Design
------
* Special tokens (``<pad> <start> <end> <unk>`` and the trace structure tokens
  ``<parent> … </name> ;``) are reserved up front and never split or merged.
* Everything else is encoded with learned character-level merges.
* encode→decode is lossless (spaces and punctuation are ordinary characters),
  so a decoded trace reconstructs the exact string the model emitted.

Vocab file format (JSON)::

    {"special_tokens": [...], "merges": [["a","b"], ...], "vocab": {tok: id, ...}}
"""
import json
import re
from collections import Counter

SPECIAL_TOKENS = ["<pad>", "<start>", "<end>", "<unk>"]
TRACE_TOKENS = [
    "<parent>", "</parent>", "<groups>", "</groups>",
    "<atoms>", "</atoms>", "<rings>", "</rings>",
    "<name>", "</name>", ";",
]


class BPETokenizer:
    def __init__(self, merges, vocab, special_tokens):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.vocab = dict(vocab)                       # token -> id
        self.itos = {i: t for t, i in self.vocab.items()}
        self.special_tokens = list(special_tokens)
        self.unk = self.vocab["<unk>"]
        # Regex that splits a string while keeping special tokens as standalone
        # pieces (longest first so "</parent>" wins over "<").
        present = sorted((t for t in special_tokens if not t.startswith("<pad")
                          and t not in ("<start>", "<end>", "<unk>")),
                         key=len, reverse=True)
        self._special_re = re.compile("(" + "|".join(re.escape(t) for t in present) + ")")
        self._special_set = set(special_tokens)
        self._cache = {}

    # ── encoding ──────────────────────────────────────────────────────────
    def _bpe(self, chunk):
        """Apply learned merges to a raw (non-special) string -> list of subword tokens."""
        if chunk in self._cache:
            return self._cache[chunk]
        word = list(chunk)
        while len(word) > 1:
            pairs = list(zip(word, word[1:]))
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            a, b = best
            merged, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        self._cache[chunk] = word
        return word

    def tokenize(self, text):
        """String -> list of token strings (no <start>/<end>)."""
        tokens = []
        for piece in self._special_re.split(text):
            if not piece:
                continue
            if piece in self._special_set:
                tokens.append(piece)
            else:
                tokens.extend(self._bpe(piece))
        return tokens

    def encode(self, text, add_bos_eos=True):
        ids = [self.vocab["<start>"]] if add_bos_eos else []
        for tok in self.tokenize(text):
            ids.append(self.vocab.get(tok, self.unk))
        if add_bos_eos:
            ids.append(self.vocab["<end>"])
        return ids

    def decode(self, ids):
        skip = {self.vocab["<start>"], self.vocab["<end>"], self.vocab["<pad>"]}
        return "".join(self.itos.get(int(i), "") for i in ids if int(i) not in skip)

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"special_tokens": self.special_tokens,
                       "merges": [list(m) for m in self.merges],
                       "vocab": self.vocab}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["merges"], d["vocab"], d["special_tokens"])

    # ── training ──────────────────────────────────────────────────────────
    @classmethod
    def train(cls, texts, vocab_size, special_tokens=None, verbose=False):
        """Learn BPE merges from an iterable of strings.

        Special tokens occupy the first ids; the base vocab is every character
        in the non-special text; merges are added (most-frequent-pair first,
        deterministic tie-break) until ``vocab_size`` is reached.
        """
        specials = list(special_tokens if special_tokens is not None
                        else SPECIAL_TOKENS + TRACE_TOKENS)
        special_set = set(specials)
        present = sorted((t for t in specials if t not in ("<pad>", "<start>", "<end>", "<unk>")),
                         key=len, reverse=True)
        split_re = re.compile("(" + "|".join(re.escape(t) for t in present) + ")") if present else None

        # Count non-special chunks (weighted BPE training over unique chunks).
        chunk_freq = Counter()
        for text in texts:
            pieces = split_re.split(text) if split_re else [text]
            for piece in pieces:
                if piece and piece not in special_set:
                    chunk_freq[piece] += 1

        # base vocab = all characters seen
        base_chars = set()
        for chunk in chunk_freq:
            base_chars.update(chunk)
        base_chars = sorted(base_chars)

        # words as char tuples -> frequency
        words = {tuple(chunk): f for chunk, f in chunk_freq.items()}

        vocab_tokens = list(specials) + base_chars
        n_merges = max(0, vocab_size - len(vocab_tokens))
        merges = []
        for step in range(n_merges):
            pair_counts = Counter()
            for word, f in words.items():
                for pair in zip(word, word[1:]):
                    pair_counts[pair] += f
            if not pair_counts:
                break
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            merges.append(best)
            new_tok = best[0] + best[1]
            vocab_tokens.append(new_tok)
            # apply merge to all words
            a, b = best
            new_words = {}
            for word, f in words.items():
                merged, i = [], 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        merged.append(new_tok)
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                new_words[tuple(merged)] = f
            words = new_words
            if verbose and (step + 1) % 500 == 0:
                print(f"  merge {step + 1}/{n_merges}: {best} (count {pair_counts[best]:,})")

        vocab = {tok: i for i, tok in enumerate(vocab_tokens)}
        return cls(merges, vocab, specials)
