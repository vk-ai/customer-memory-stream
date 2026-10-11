"""Embedders.

HashingEmbedder is the default: signed feature hashing over word unigrams,
word bigrams and character trigrams, L2-normalised. It needs no model download,
is deterministic, and embeds a sentence in tens of microseconds on CPU. It is a
lexical embedder - it does NOT understand synonyms the way a neural model does.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
STOP = frozenset(
    "a an the is are was were i my me to of and or in on at for it this that be with "
    "you your we our do does did have has had what which please can could would".split()
)


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


class HashingEmbedder:
    name = "hashing-v1"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def features(self, text: str) -> list[str]:
        toks = [t for t in _TOKEN.findall(text.lower()) if t not in STOP]
        feats = [f"w:{t}" for t in toks]
        feats += [f"b:{a}_{b}" for a, b in zip(toks, toks[1:])]
        for t in toks:
            p = f"#{t}#"
            feats += [f"c:{p[i:i+3]}" for i in range(len(p) - 2)]
        return feats

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for f in self.features(text):
            h = _h(f)
            w = 1.0 if f[0] == "w" else (0.7 if f[0] == "b" else 0.25)
            v[h % self.dim] += w if (h >> 63) & 1 == 0 else -w
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


class MiniLMEmbedder:
    """Optional neural embedder: sentence-transformers/all-MiniLM-L6-v2 (22M params)
    via plain `transformers` + mean pooling. Requires `pip install torch transformers`;
    not used in CI. ~384-d."""

    name = "all-MiniLM-L6-v2"

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.dim = self.model.config.hidden_size
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]
        with self._torch.no_grad():
            b = self.tok([text], return_tensors="pt", truncation=True, max_length=128)
            h = self.model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            v = ((h * m).sum(1) / m.sum(1))[0].numpy().astype(np.float32)
        v /= np.linalg.norm(v) or 1.0
        self._cache[text] = v
        return v
