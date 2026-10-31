import json
import platform
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"

CORPUS_CSV = DATA_DIR / "physics_corpus.csv"
CORPUS_JSON = DATA_DIR / "physics_corpus.json"
CORPUS_STATS = DATA_DIR / "corpus_stats.json"
EMBEDDINGS_NPY = DATA_DIR / "embeddings_minilm.npy"
EMBEDDING_META = DATA_DIR / "embeddings_meta.json"

HF_DATASET = "TimSchopf/arxiv_categories"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
ENCODE_BATCH_SIZE = 64
TARGET_CORPUS_SIZE = 20_000
RANDOM_SEED = 42
MIN_ABSTRACT_CHARS = 100

PHYSICS_PREFIXES = (
    "astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
    "math-ph", "nlin", "nucl-ex", "nucl-th", "physics.", "quant-ph",
)
TITLE_CANDIDATES = ["title", "paper_title", "Title"]
ABSTRACT_CANDIDATES = ["abstract", "summary", "text", "Abstract"]
CATEGORY_CANDIDATES = ["categories", "category", "label", "labels", "terms"]


def environment_report():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["torch"] = None
    return info


def build_document_text(title, abstract):
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    return f"{title}. {abstract}".strip()


def pick_column(available, candidates, role):
    for c in candidates:
        if c in available:
            return c
    raise KeyError(f"no '{role}' column among {list(available)}")


def normalise_categories(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.replace(",", " ").split() if p.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if hasattr(value, "tolist"):
        return [str(v).strip() for v in value.tolist() if str(v).strip()]
    if hasattr(value, "__iter__"):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def is_physics(categories):
    if not categories:
        return False
    primary = categories[0]
    root = primary.split("->")[0].strip().lower()
    if root.startswith("physics"):
        return True
    return primary.startswith(PHYSICS_PREFIXES)


def coarse_domain(primary_category):
    parts = [p.strip() for p in primary_category.split("->") if p.strip()]
    if parts and parts[0].lower().endswith("archive"):
        parts = parts[1:]
    if not parts:
        root = primary_category.split("->")[0].strip()
        return root[: -len(" Archive")] if root.lower().endswith("archive") else root
    code = parts[0]
    if code.startswith("physics."):
        return code
    return code.split(".")[0]


def stratified_downsample(df, target_size, domain_col="domain", seed=RANDOM_SEED):
    if len(df) <= target_size:
        return df
    frac = target_size / len(df)
    parts = []
    for _, group in df.groupby(domain_col):
        n = max(1, int(round(len(group) * frac)))
        parts.append(group.sample(n=n, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > target_size:
        out = out.sample(n=target_size, random_state=seed).reset_index(drop=True)
    return out.reset_index(drop=True)


def build_corpus(out_csv=None, out_json=None, out_stats=None):
    from datasets import load_dataset

    out_csv = Path(out_csv or CORPUS_CSV)
    out_json = Path(out_json or CORPUS_JSON)
    out_stats = Path(out_stats or CORPUS_STATS)

    ds = load_dataset(HF_DATASET, split="train")
    cols = ds.column_names
    title_col = pick_column(cols, TITLE_CANDIDATES, "title")
    abstract_col = pick_column(cols, ABSTRACT_CANDIDATES, "abstract")
    category_col = pick_column(cols, CATEGORY_CANDIDATES, "categories")
    df = ds.to_pandas().rename(
        columns={title_col: "title", abstract_col: "abstract", category_col: "categories_raw"}
    )[["title", "abstract", "categories_raw"]]

    funnel = {"loaded": len(df)}
    df["categories"] = df["categories_raw"].apply(normalise_categories)
    df = df[df["categories"].map(len) > 0]
    df["primary_category"] = df["categories"].apply(lambda c: c[0])
    df = df[df["categories"].apply(is_physics)].copy()
    funnel["physics_primary"] = len(df)
    if len(df) == 0:
        raise ValueError("no physics papers found; check the category format of the dataset")

    df = df[df["abstract"].fillna("").str.len() >= MIN_ABSTRACT_CHARS]
    df = df.drop_duplicates(subset=["title"])
    funnel["after_filter"] = len(df)
    df["domain"] = df["primary_category"].apply(coarse_domain)
    df = stratified_downsample(df, TARGET_CORPUS_SIZE)
    funnel["final_sample"] = len(df)

    df = df.reset_index(drop=True)
    df.insert(0, "doc_id", range(len(df)))
    df["categories"] = df["categories"].apply(lambda c: " ".join(c))
    out = df[["doc_id", "title", "abstract", "primary_category", "domain", "categories"]]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    out.to_json(out_json, orient="records", indent=2)

    domain_counts = Counter(out["domain"])
    stats = {
        "n_documents": len(out),
        "n_domains": len(domain_counts),
        "domain_counts": dict(domain_counts.most_common()),
        "abstract_chars": {
            "mean": float(out["abstract"].str.len().mean()),
            "median": float(out["abstract"].str.len().median()),
            "min": int(out["abstract"].str.len().min()),
            "max": int(out["abstract"].str.len().max()),
        },
        "abstract_words": {
            "mean": float(out["abstract"].str.split().str.len().mean()),
            "median": float(out["abstract"].str.split().str.len().median()),
        },
        "funnel": funnel,
        "source_dataset": HF_DATASET,
        "random_seed": RANDOM_SEED,
    }
    out_stats.write_text(json.dumps(stats, indent=2))
    print(funnel)
    return out


def load_encoder(device=None):
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if (MODEL_DIR / "modules.json").exists():
        return SentenceTransformer(str(MODEL_DIR), device=device)
    model = SentenceTransformer(MODEL_NAME, device=device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR))
    return model


def encode_corpus(df, model, out_npy=None, out_meta=None):
    out_npy = Path(out_npy or EMBEDDINGS_NPY)
    out_meta = Path(out_meta or EMBEDDING_META)
    texts = [
        build_document_text(t, a)
        for t, a in zip(df["title"].fillna(""), df["abstract"].fillna(""))
    ]

    sample = texts[: min(500, len(texts))]
    tok_lens = np.array([len(model.tokenizer.encode(t, add_special_tokens=True)) for t in sample])
    n_trunc = int((tok_lens > model.max_seq_length).sum())
    pct_trunc = 100.0 * n_trunc / len(tok_lens)

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    elapsed = time.time() - t0

    assert embeddings.shape == (len(df), EMBEDDING_DIM)
    assert np.isfinite(embeddings).all()
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    np.save(out_npy, embeddings)
    meta = {
        "model_name": MODEL_NAME,
        "n_documents": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "normalized": True,
        "max_seq_length": int(model.max_seq_length),
        "batch_size": ENCODE_BATCH_SIZE,
        "encode_seconds": round(elapsed, 2),
        "text_format": "<title>. <abstract>",
        "row_alignment": "row i of embeddings == doc_id i in physics_corpus.csv",
        "truncation_sample": {
            "sample_size": int(len(tok_lens)),
            "mean_tokens": float(tok_lens.mean()),
            "median_tokens": float(np.median(tok_lens)),
            "max_tokens": int(tok_lens.max()),
            "pct_exceeding_max_seq_length": round(pct_trunc, 2),
        },
        "norm_check": {
            "min": float(norms.min()),
            "max": float(norms.max()),
            "mean": float(norms.mean()),
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"encoded {len(texts):,} documents in {elapsed:.1f}s; {pct_trunc:.1f}% exceed {model.max_seq_length} tokens")
    return embeddings


def verify_artifacts(corpus_csv=None, embeddings_npy=None, check_model=True):
    corpus_csv = Path(corpus_csv or CORPUS_CSV)
    embeddings_npy = Path(embeddings_npy or EMBEDDINGS_NPY)
    failures = []

    def check(label, ok, detail=""):
        print(("[PASS] " if ok else "[FAIL] ") + label + ("" if ok else "  " + detail))
        if not ok:
            failures.append(label)
        return ok

    df = pd.read_csv(corpus_csv)
    emb = np.load(embeddings_npy)
    required = ["doc_id", "title", "abstract", "primary_category", "domain"]
    check("required columns present", all(c in df.columns for c in required))
    check("no null titles or abstracts", bool(df["title"].notna().all() and df["abstract"].notna().all()))
    check("no duplicate titles", not df["title"].duplicated().any())
    check("doc_id is 0..N-1", np.array_equal(df["doc_id"].values, np.arange(len(df))))
    check("row count matches corpus", emb.shape[0] == len(df), f"{emb.shape[0]} vs {len(df)}")
    check(f"dimension is {EMBEDDING_DIM}", emb.shape[1] == EMBEDDING_DIM)
    check("dtype is float32", emb.dtype == np.float32)
    check("all values finite", bool(np.isfinite(emb).all()))
    norms = np.linalg.norm(emb, axis=1)
    check("rows are unit length", bool(np.allclose(norms, 1.0, atol=1e-4)))
    check("embeddings vary across documents", float(emb.std(axis=0).mean()) > 1e-3)
    n = min(200, len(emb))
    sims = emb[:n] @ emb[:n].T
    check("pairwise similarities are not all near 1", float(sims[~np.eye(n, dtype=bool)].mean()) < 0.95)

    if check_model and (MODEL_DIR / "modules.json").exists():
        model = load_encoder(device="cpu")
        text = build_document_text(df.loc[0, "title"], df.loc[0, "abstract"])
        vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        check("re-encoding doc 0 reproduces the stored vector", float(vec @ emb[0]) > 0.999)
    else:
        print("[SKIP] local model not found; re-encoding check skipped")

    print("ALL CHECKS PASSED" if not failures else f"FAILED: {failures}")
    return failures


def tokenize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


@dataclass
class BM25Index:
    doc_ids: list
    bm25: BM25Okapi = field(repr=False)

    @classmethod
    def build(cls, doc_ids, texts):
        return cls(doc_ids=list(doc_ids), bm25=BM25Okapi([tokenize(t) for t in texts]))

    def scores(self, query):
        return self.bm25.get_scores(tokenize(query))

    def search(self, query, k=10):
        order = np.argsort(-self.scores(query))[:k]
        return [self.doc_ids[i] for i in order]


@dataclass
class DenseIndex:
    doc_ids: list
    embeddings: np.ndarray
    model: object = field(repr=False)

    @classmethod
    def build(cls, doc_ids, embeddings, model):
        assert embeddings.shape[0] == len(doc_ids)
        return cls(doc_ids=list(doc_ids), embeddings=embeddings, model=model)

    def scores(self, query):
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        return self.embeddings @ q.astype(np.float32)

    def search(self, query, k=10):
        order = np.argsort(-self.scores(query))[:k]
        return [self.doc_ids[i] for i in order]


def merge_ranked_lists(ranked_lists, k):
    hits, best = {}, {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            hits[doc_id] = hits.get(doc_id, 0) + 1
            if doc_id not in best or rank < best[doc_id]:
                best[doc_id] = rank
    return sorted(hits, key=lambda d: (-hits[d], best[d]))[:k]


def build_indices(corpus, embeddings=None, encoder=None):
    texts = [build_document_text(t, a) for t, a in zip(corpus["title"], corpus["abstract"])]
    doc_ids = corpus["doc_id"].tolist()
    indices = {"bm25": BM25Index.build(doc_ids, texts)}
    if embeddings is not None and encoder is not None:
        indices["dense_minilm"] = DenseIndex.build(doc_ids, embeddings, encoder)
    return indices
