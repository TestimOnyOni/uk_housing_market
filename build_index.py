# build_index.py
import argparse
import os
import json
import numpy as np
import pandas as pd
import joblib
from rag_core import Embedder, row_to_doc


class VectorIndex:
    """Builds and saves either a FAISS or sklearn KNN index.
        Saved files in index_dir:
        - index.faiss OR sklearn_index.pkl
        - meta.parquet
        - config.json {dim:int, engine:"faiss"|"sklearn"}
    """
    def __init__(self, dim: int, index_dir: str):
        self.dim = dim
        self.index_dir = index_dir
        self.engine = None
        self.idx = None


    def build(self, embeddings: np.ndarray):
        try:
            import faiss # type: ignore
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            emb_norm = embeddings / norms
            self.idx = faiss.IndexFlatIP(self.dim)
            self.idx.add(emb_norm.astype(np.float32))
            self.engine = "faiss"
        except Exception:
            from sklearn.neighbors import NearestNeighbors
            self.idx = NearestNeighbors(n_neighbors=10, metric="cosine")
            self.idx.fit(embeddings.astype(np.float32))
            self.engine = "sklearn"


    def save(self, meta: pd.DataFrame):
        os.makedirs(self.index_dir, exist_ok=True)
        if self.engine == "faiss":
            import faiss
            faiss.write_index(self.idx, os.path.join(self.index_dir, "index.faiss"))
        else:
            joblib.dump(self.idx, os.path.join(self.index_dir, "sklearn_index.pkl"))
            meta.to_parquet(os.path.join(self.index_dir, "meta.parquet"), index=False)
        with open(os.path.join(self.index_dir, "config.json"), "w") as f:
            json.dump({"dim": self.dim, "engine": self.engine}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--index-dir", default="artifacts")
    ap.add_argument("--use-openai", action="store_true")
    ap.add_argument("--no-openai", action="store_true")
    args = ap.parse_args()

    use_openai = True
    if args.no_openai:
        use_openai = False
    if args.use_openai:
        use_openai = True

    df = pd.read_csv(args.csv)

    docs = [row_to_doc(r) for _, r in df.iterrows()]
    embedder = Embedder(use_openai=use_openai)
    embs = embedder.embed(docs)

    vi = VectorIndex(dim=embs.shape[1], index_dir=args.index_dir)
    vi.build(embs)
    vi.save(df.reset_index(drop=True))

    print(f"✅ Built {vi.engine} index on {len(docs)} docs → {args.index_dir}")


if __name__ == "__main__":
    main()