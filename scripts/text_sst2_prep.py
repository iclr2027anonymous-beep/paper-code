"""Embed SST-2 sentences with a frozen sentence-embedding encoder and cache to disk.

Prepares the text LSNPC input: clean+noisy label structure is handled later (noise
injection), here we only produce the frozen embeddings + clean labels + splits.

Outputs (pickle):
  data/sst2/embeddings.pkl -> {
    'train_x': (n_train, dim) float32, 'train_y': (n_train,) int64 (clean),
    'test_x': (n_test, dim) float32, 'test_y': (n_test,) int64 (clean),
    'train_text': [str], 'test_text': [str], 'dim': int,
    'encoder': str }
"""
from __future__ import annotations

from utils.paths import project_path

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

OUT = Path(project_path('data/sst2/embeddings.pkl'))
MODEL_PATH = project_path('models/all-mpnet-base-v2')
BATCH = 512
def main():
    data_dir = Path(project_path('data/sst2'))
    tr = pd.read_parquet(data_dir / "train-00000-of-00001.parquet")
    te = pd.read_parquet(data_dir / "test-00000-of-00001.parquet")
    tr_text = tr["sentence"].tolist(); tr_y = tr["label"].to_numpy()
    te_text = te["sentence"].tolist(); te_y = te["label"].to_numpy()
    print(f"train {len(tr_text)} / test {len(te_text)}")

    enc = SentenceTransformer(MODEL_PATH)
    enc.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = enc.to(dev)

    def embed(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), BATCH):
            e = enc.encode(texts[i:i + BATCH], batch_size=BATCH,
                           normalize_embeddings=False, convert_to_numpy=True,
                           show_progress_bar=False)
            out.append(e)
        return np.concatenate(out, axis=0).astype(np.float32)

    tr_x = embed(tr_text); te_x = embed(te_text)
    payload = {
        "train_x": tr_x, "train_y": tr_y.astype(np.int64),
        "test_x": te_x, "test_y": te_y.astype(np.int64),
        "train_text": tr_text, "test_text": te_text,
        "dim": int(tr_x.shape[1]), "encoder": "all-mpnet-base-v2",
    }
    with open(OUT, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB), dim={payload['dim']}")
    print("train y dist:", np.bincount(payload['train_y']))
    print("test y dist:", np.bincount(payload['test_y']))


if __name__ == "__main__":
    main()
