"""Embed the NoisyAG-News pool (50k texts shared by all 3 variants) and cache.

The three variants (best/med/worst) annotate the SAME 50k AG-News rows in the
same order (verified at prep time). We embed the pool ONCE with the frozen
all-mpnet-base-v2 encoder and cache per-variant real noisy labels + the
ground-truth oracle, so a downstream loader picks a variant without re-embedding
and without synthetic noise injection (the noise is the real crowd noise).

Outputs (pickle): data/noisyag_news/embeddings.pkl ->
  'x': (50000, dim) float32 embeddings
  'ground_truth': (50000,) int64  (original AG-News label = per-query oracle)
  'noisy_best' / 'noisy_med' / 'noisy_worst': (50000,) int64
  'dim': int, 'encoder': str
"""
from __future__ import annotations

from utils.paths import project_path

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

OUT = Path(project_path('data/noisyag_news/embeddings.pkl'))
MODEL_PATH = project_path('models/all-mpnet-base-v2')
BATCH = 512
def main() -> None:
    data_dir = Path(project_path('data/noisyag_news'))
    rows_all = {}
    for v in ("best", "med", "worst"):
        with open(data_dir / f"noisyag_{v}.jsonl") as f:
            rows_all[v] = [json.loads(line) for line in f]
    base = rows_all["best"]
    n = len(base)
    for v in ("med", "worst"):
        assert len(rows_all[v]) == n, f"{v} row count {len(rows_all[v])} != {n}"
        assert [r["text"] for r in rows_all[v]] == [r["text"] for r in base], (
            f"{v} text order differs from best; variants must share the pool"
        )
    print(f"shared pool verified: {n} texts, order identical across variants")

    enc = SentenceTransformer(MODEL_PATH)
    enc.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = enc.to(dev)

    texts = [r["text"] for r in base]
    emb = []
    for i in range(0, n, BATCH):
        e = enc.encode(texts[i:i + BATCH], batch_size=BATCH,
                       normalize_embeddings=False, convert_to_numpy=True,
                       show_progress_bar=False)
        emb.append(e)
    x = np.concatenate(emb, axis=0).astype(np.float32)
    assert x.shape[0] == n

    gt = np.asarray([r["ground_truth"] for r in base], dtype=np.int64)
    noisy = {v: np.asarray([r["noisy_label"] for r in rows_all[v]], dtype=np.int64)
             for v in ("best", "med", "worst")}
    for v in ("best", "med", "worst"):
        print(f"  {v}: flip vs oracle = {(noisy[v] != gt).mean():.4f}")

    payload = {
        "x": x, "ground_truth": gt,
        "noisy_best": noisy["best"], "noisy_med": noisy["med"],
        "noisy_worst": noisy["worst"],
        "dim": int(x.shape[1]), "encoder": "all-mpnet-base-v2",
    }
    with open(OUT, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.0f} MB), dim={payload['dim']}")
    print("ground-truth class dist:", np.bincount(gt))


if __name__ == "__main__":
    main()
