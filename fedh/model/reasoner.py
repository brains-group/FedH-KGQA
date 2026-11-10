"""Relation-space reasoner: masked question -> per-hop relation distributions + hop count.

Two encoders:
  hf   a Hugging Face encoder (default roberta-base) with LoRA adapters; only the
       adapters and the heads are trained and exchanged in federated rounds
  bow  bag-of-words embedding + MLP; no download, used by the smoke test
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

_WORD = re.compile(r"\[ent\]|[a-z0-9']+")


class BowTokenizer:
    """Word-level tokenizer with a fixed vocabulary (0 = pad, 1 = unknown)."""

    def __init__(self, vocab: dict):
        self.vocab = vocab

    @classmethod
    def build(cls, texts, min_count: int = 1):
        counts = {}
        for t in texts:
            for w in _WORD.findall(t.lower()):
                counts[w] = counts.get(w, 0) + 1
        vocab = {"<pad>": 0, "<unk>": 1}
        for w, n in sorted(counts.items()):
            if n >= min_count:
                vocab[w] = len(vocab)
        return cls(vocab)

    def encode(self, text: str, max_len: int):
        ids = [self.vocab.get(w, 1) for w in _WORD.findall(text.lower())][:max_len]
        return ids or [1]


class BowEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.out_dim = dim

    def forward(self, input_ids, attention_mask):
        e = self.emb(input_ids) * attention_mask.unsqueeze(-1)
        h = e.sum(1) / attention_mask.sum(1, keepdim=True).clamp(min=1)
        return self.mlp(h)


class HFEncoder(nn.Module):
    def __init__(self, name: str, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1,
                 target_modules=("query", "value"), attn_implementation=None, mask4d: bool = False):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel
        kw = {"attn_implementation": attn_implementation} if attn_implementation else {}
        base = AutoModel.from_pretrained(name, **kw)
        self.mask4d = mask4d       # pass a [B, 1, 1, L] boolean mask (needed under torch.func.vmap for DP)
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                         target_modules=list(target_modules), bias="none")
        self.model = get_peft_model(base, cfg)
        self.out_dim = base.config.hidden_size

    def forward(self, input_ids, attention_mask):
        if self.mask4d:
            attention_mask = attention_mask.bool()[:, None, None, :]
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]


def relation_text(name: str) -> str:
    """'people.person.sibling_s__inv' -> 'reverse of people person sibling s'."""
    inv = name.endswith("__inv")
    base = name[:-5] if inv else name
    words = re.sub(r"[._/]+", " ", base).strip()
    return ("reverse of " + words) if inv else words


class RelTextHead(nn.Module):
    """Scores relations by their names: logit(r) = <W_q h_t, W_r f(r)> / sqrt(d) + b_r, where f(r) is a
    fixed feature vector of the relation's name. Relations seen rarely, or only at another client,
    still get sensible scores through the words they share with frequent ones."""

    def __init__(self, d_q: int, feats: torch.Tensor, d: int = 256):
        super().__init__()
        self.register_buffer("feats", feats.float(), persistent=False)
        self.proj_q = nn.Sequential(nn.Linear(d_q, d_q), nn.GELU(), nn.Linear(d_q, d))
        self.proj_r = nn.Linear(feats.shape[1], d)
        self.bias = nn.Parameter(torch.zeros(feats.shape[0]))
        self.scale = d ** -0.5

    def forward(self, hh):
        return self.proj_q(hh) @ self.proj_r(self.feats).t() * self.scale + self.bias


def bow_relation_features(names):
    """Bag-of-words counts over the words of relation names (used with any encoder)."""
    texts = [relation_text(n).lower().split() for n in names]
    vocab = {w: i for i, w in enumerate(sorted({w for t in texts for w in t}))}
    f = torch.zeros(len(names), len(vocab))
    for i, t in enumerate(texts):
        for w in t:
            f[i, vocab[w]] += 1.0
    return f / f.sum(1, keepdim=True).clamp(min=1.0)


@torch.no_grad()
def hf_relation_features(model_name: str, names, device, batch: int = 256):
    """Mean-pooled embeddings of relation names from the frozen pretrained encoder."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    texts = [relation_text(n) for n in names]
    for i in range(0, len(texts), batch):
        b = tok(texts[i: i + batch], padding=True, truncation=True, max_length=32, return_tensors="pt").to(device)
        h = enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).float().cpu())
    del enc
    return torch.cat(out)


class Reasoner(nn.Module):
    """Encoder + per-hop relation scores + a hop-count head.

    rel_feats=None: one learned output per relation id (fine when every relation has many examples).
    rel_feats given: relations are scored by their names (RelTextHead)."""

    def __init__(self, encoder: nn.Module, n_rel: int, k_max: int = 3, dropout: float = 0.1, rel_feats=None):
        super().__init__()
        self.encoder = encoder
        self.k_max = k_max
        self.n_rel = n_rel
        d = encoder.out_dim
        self.drop = nn.Dropout(dropout)
        self.hop_emb = nn.Parameter(torch.zeros(k_max, d))
        if rel_feats is None:
            self.rel_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_rel))
        else:
            self.rel_head = RelTextHead(d, rel_feats)
        self.k_head = nn.Linear(d, k_max)

    def forward(self, input_ids, attention_mask):
        h = self.drop(self.encoder(input_ids, attention_mask)).float()        # [B, d]
        hh = h.unsqueeze(1) + self.hop_emb.unsqueeze(0)                        # [B, K, d]
        rel_logits = self.rel_head(hh)                                         # [B, K, R]
        k_logits = self.k_head(h)                                              # [B, K]
        return rel_logits, k_logits

    def trainable_state(self):
        return {n: p.detach().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state):
        params = dict(self.named_parameters())
        with torch.no_grad():
            for n, v in state.items():
                params[n].copy_(v)


def reasoner_loss(rel_logits, k_logits, paths, path_mask, k_target):
    """Path-level marginal likelihood plus hop-count cross-entropy.

    paths:     [B, P, K] relation ids, -1 where a path is shorter than K
    path_mask: [B, P] 1 for a real label path
    k_target:  [B] hop count minus one, -1 when unknown
    A path of length 1 inside a longer question is a partial (first-hop) label.
    """
    logp = F.log_softmax(rel_logits, dim=-1)                                   # [B, K, R]
    B, P, K = paths.shape
    idx = paths.clamp(min=0)
    lp = torch.gather(logp.unsqueeze(1).expand(B, P, K, logp.size(-1)), 3, idx.unsqueeze(-1)).squeeze(-1)
    lp = (lp * (paths >= 0)).sum(-1)                                           # [B, P]
    lp = lp.masked_fill(path_mask == 0, float("-inf"))
    has = path_mask.sum(1) > 0
    path_nll = -torch.logsumexp(lp, dim=1)
    loss_path = path_nll[has].mean() if has.any() else rel_logits.sum() * 0
    kv = k_target >= 0
    loss_k = F.cross_entropy(k_logits[kv], k_target[kv]) if kv.any() else k_logits.sum() * 0
    return loss_path + loss_k, loss_path.detach(), loss_k.detach()
