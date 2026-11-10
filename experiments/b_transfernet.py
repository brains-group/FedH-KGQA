"""TransferNet (Shi et al., 2021) on the pooled KG: a centralized relation-following baseline.

At each step t the model attends to the question and gives every relation an independent score
p_t(r) = sigmoid(.); entity scores are transferred along all edges at once,
    e_t(x) = min(1, sum_{(h, r, x)} e_{t-1}(h) p_t(r)),
starting from the topic entity, and the answer scores mix the steps with a hop attention,
a(x) = sum_t pi_t e_t(x). Training uses answers only (no relation-path labels), end to end through
the transfer. Same encoder, masked question and answer rules as FedH-KGQA, so the comparison
isolates what our path labels and federation change.

  python experiments/b_transfernet.py --data-dir data/processed/metaqa/2hop/C3_rho0.5_balanced_s0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from b_kgqa import Partition, Tok, autocast, fmt, pad_batch, read_jsonl, score  # noqa: E402
from fedh.model.reasoner import BowEncoder, HFEncoder, Reasoner  # noqa: E402


def transfer(e0, rel_logits, k_logits, H, R, T, n_ent, no_return, topic_answer):
    B, K, _ = rel_logits.shape
    dev = rel_logits.device
    rows = torch.arange(B, device=dev)
    start = torch.zeros(B, n_ent, device=dev)
    start[rows, e0] = 1.0
    topic_mask = 1.0 - start
    p = torch.sigmoid(rel_logits.float())
    pi = torch.softmax(k_logits.float(), -1)
    e, out = start, 0.0
    for t in range(K):
        w = e[:, H] * p[:, t, R]                                   # [B, edges]
        e = torch.zeros(B, n_ent, device=dev).index_add(1, T, w).clamp(max=1.0)
        if no_return:
            e = e * topic_mask
        out = out + pi[:, t: t + 1] * e
    if not topic_answer:
        out = out * topic_mask
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="hf", choices=["hf", "bow"])
    ap.add_argument("--model-name", default="roberta-base")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dev-size", type=int, default=2000)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    part_name = os.path.basename(os.path.normpath(args.data_dir))
    hops = os.path.basename(os.path.dirname(os.path.normpath(args.data_dir)))
    tag = args.tag or f"{hops}_{part_name}_transfernet_s{args.seed}"
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)

    part = Partition(args.data_dir)
    no_return = False
    sp = os.path.join(args.data_dir, "stats.json")
    if os.path.exists(sp):
        with open(sp) as f:
            no_return = bool(json.load(f).get("exec_rules", {}).get("no_return", False))
    tri = np.concatenate(part.triples)
    H = torch.as_tensor(tri[:, 0], device=device)
    R = torch.as_tensor(tri[:, 1], device=device)
    T = torch.as_tensor(tri[:, 2], device=device)
    train = read_jsonl(os.path.join(args.data_dir, "train.jsonl"), args.max_train)
    dev = read_jsonl(os.path.join(args.data_dir, "dev.jsonl"))
    test = read_jsonl(os.path.join(args.data_dir, "test.jsonl"))
    if args.dev_size and len(dev) > args.dev_size:
        dev = random.Random(args.seed + 1).sample(dev, args.dev_size)
    print(f"[data] {part_name} ({hops}): {len(train)} train, {len(dev)} dev, {len(test)} test; {part.n_ent} entities, "
          f"{len(tri)} edges, {part.n_rel} relations; no_return={no_return} topic_answer={part.topic_answer}; "
          f"device {device}")

    tok = Tok(args, [q["question_masked"] for q in train])
    tr_ids, dv_ids, te_ids = tok.encode(train), tok.encode(dev), tok.encode(test)
    enc = BowEncoder(len(tok.tok.vocab), 256) if args.encoder == "bow" else \
        HFEncoder(args.model_name, args.lora_r, args.lora_alpha, 0.1)
    model = Reasoner(enc, part.n_rel, args.k_max).to(device)
    ex = [(x, q["topic_id"], [a for a in q["answers"] if part.topic_answer or a != q["topic_id"]])
          for q, x in zip(train, tr_ids)]
    ex = [e for e in ex if e[2]]

    @torch.no_grad()
    def predict(qs, ids):
        model.eval()
        top = []
        for i in range(0, len(qs), args.eval_batch_size):
            x, att = pad_batch(ids[i: i + args.eval_batch_size], tok.pad_id, device)
            e0 = torch.as_tensor([q["topic_id"] for q in qs[i: i + args.eval_batch_size]], device=device)
            with autocast(device):
                rl, kl = model(x, att)
            a = transfer(e0, rl, kl, H, R, T, part.n_ent, no_return, part.topic_answer)
            top.append(a.argmax(1).cpu().numpy())
        return np.concatenate(top) if top else np.zeros(0, dtype=np.int64)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = max(1, math.ceil(len(ex) / args.batch_size) * args.epochs)
    warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / steps))
    best, best_state, history = -1.0, model.trainable_state(), []
    for ep in range(args.epochs):
        ts = time.time()
        model.train()
        order = list(range(len(ex)))
        rng.shuffle(order)
        tot, n = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            b = [ex[j] for j in order[i: i + args.batch_size]]
            x, att = pad_batch([e[0] for e in b], tok.pad_id, device)
            e0 = torch.as_tensor([e[1] for e in b], device=device)
            y = torch.zeros(len(b), part.n_ent, device=device)
            for k, e in enumerate(b):
                y[k, e[2]] = 1.0
            with autocast(device):
                rl, kl = model(x, att)
            a = transfer(e0, rl, kl, H, R, T, part.n_ent, no_return, part.topic_answer)
            loss = ((a - y) ** 2).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
            n += len(b)
        dm, _ = score(dev, predict(dev, dv_ids))
        history.append({"epoch": ep + 1, "loss": tot / max(1, n), "dev": dm})
        print(f"[epoch {ep + 1}/{args.epochs}] loss {tot / max(1, n):.4f} | dev {fmt(dm)} ({time.time() - ts:.0f}s)")
        if dm["hits1"] > best:
            best, best_state = dm["hits1"], model.trainable_state()
    model.load_trainable_state(best_state)
    top = predict(test, te_ids)
    tm, hits = score(test, top)
    print(f"[test] {fmt(tm)} (n={tm['n']}, cross {tm['n_cross']})")
    res = {"tag": tag, "method": "transfernet", "args": vars(args), "data": part_name, "hops": hops,
           "history": history, "test": tm, "minutes": (time.time() - t0) / 60}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out_dir, "test_preds.jsonl"), "w") as f:
        for q, t, h in zip(test, top, hits):
            f.write(json.dumps({"qid": q["qid"], "top": int(t), "hit": float(h), "cross": bool(q.get("cross_client"))}) + "\n")
    print(f"[done] {out_dir}/metrics.json in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
