"""Embed the medical_abstracts pool with the frozen mpnet encoder and cache.

medical_abstracts (5-class, 14,438 PubMed abstracts) is a clean benchmark:
noise is injected synthetically downstream by the text entry point
(scripts/text_lsnpc_sst2.py), which reads the cached train-pool embeddings
here the same way it reads the SST-2 / AG-News caches.

Outputs (pickle): data/medical_abstracts/embeddings.pkl ->
  'x': (11550, dim) float32 embeddings of the train pool
  'y': (11550,) int64 clean labels (file labels 1-5 mapped to 0-4)
  'x_test': (2888, dim) float32 embeddings of the test pool
  'y_test': (2888,) int64 clean labels
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

OUT = Path(project_path('data/medical_abstracts/embeddings.pkl'))
MODEL_PATH = project_path('models/all-mpnet-base-v2')
BATCH = 512


def _read_pool(path: Path) -> tuple[list[str], np.ndarray]:
    rows = [json.loads(line) for line in path.open()]
    texts = [r["text"] for r in rows]
    y = np.asarray([r["label"] for r in rows], dtype=np.int64) - 1  # 1-5 -> 0-4
    return texts, y


def main() -> None:
    data_dir = Path(project_path('data/medical_abstracts'))
    tr_texts, tr_y = _read_pool(data_dir / "train.jsonl")
    te_texts, te_y = _read_pool(data_dir / "test.jsonl")
    print(f"train {len(tr_texts)} rows / test {len(te_texts)} rows; "
          f"classes {int(tr_y.min())}..{int(tr_y.max())}")

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

    x = embed(tr_texts)
    x_test = embed(te_texts)
    payload = {
        "x": x, "y": tr_y.astype(np.int64),
        "x_test": x_test, "y_test": te_y.astype(np.int64),
        "dim": int(x.shape[1]), "encoder": "all-mpnet-base-v2",
    }
    with open(OUT, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB), dim={payload['dim']}")
    print("train y dist:", np.bincount(payload["y"]))
    print("test  y dist:", np.bincount(payload["y_test"]))


if __name__ == "__main__":
    main()
