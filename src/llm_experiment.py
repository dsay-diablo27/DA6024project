import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import corpus_retrieval as cr

DATA_DIR = cr.DATA_DIR
EVAL_AUTO_JSON = DATA_DIR / "eval_auto.json"
RESULTS_JSON = DATA_DIR / "results.json"
ANALYSIS_JSON = DATA_DIR / "analysis.json"
ANNOTATIONS_JSON = DATA_DIR / "case_annotations.json"
PERSONAL_CSV = DATA_DIR / "eval_personal.csv"
MANUAL_JSON = DATA_DIR / "manual_eval_results.json"

EVAL_N = 60
TOP_K = 10
TOP_K_MANUAL = 5
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://localhost:11434"
ANTHROPIC_MODEL = "claude-haiku-5-5"

DECOMPOSE_SYSTEM = (
    "You read an informal description of a research difficulty and split it into distinct sub-problems. "
    "One confused paragraph often contains more than one difficulty, and each needs its own literature search. "
    "If the description is really only one difficulty, return a single sub-problem. "
    'Reply with ONLY a JSON object: {"sub_problems": ["...", ...]}'
)
QUERY_GEN_SYSTEM = (
    "You turn a research sub-problem into 2-3 short search queries phrased the way the scientific literature "
    "would phrase them - i.e. using the formal technical vocabulary a paper's title or abstract would use, "
    "not the informal words a stuck researcher would use. "
    'Reply with ONLY a JSON object: {"queries": ["...", "...", ...]}'
)
EXPLAIN_SYSTEM = (
    "You explain, in one or two sentences, why a retrieved paper is relevant to a researcher's problem. "
    "Base the explanation ONLY on the given abstract - do not invent content the abstract does not support. "
    'Reply with ONLY a JSON object: {"explanation": "..."}'
)
SYNTH_PROBLEM_SYSTEM = (
    "You will be shown the title and abstract of a physics paper. Write ONE short paragraph (2-4 sentences) in the "
    "voice of a researcher describing a difficulty they are stuck on, that this paper's method would resolve - "
    "written the way someone who has NOT read this paper and does NOT know its title-level vocabulary would describe "
    "their confusion, in informal, symptom-level language. Do not use the paper's title, named methods, or "
    "distinctive technical terms from the abstract. "
    'Reply with ONLY a JSON object: {"problem_description": "..."}'
)


class LLMClient:
    def complete(self, system, user, max_tokens=512):
        raise NotImplementedError


@dataclass
class AnthropicLLM(LLMClient):
    model: str = ANTHROPIC_MODEL
    max_retries: int = 3

    def __post_init__(self):
        import anthropic

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic(api_key=key)

    def complete(self, system, user, max_tokens=512):
        last = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return "".join(b.text for b in resp.content if b.type == "text")
            except Exception as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Anthropic call failed after {self.max_retries} attempts: {last}")


@dataclass
class OllamaLLM(LLMClient):
    model: str = OLLAMA_MODEL
    host: str = OLLAMA_HOST
    max_retries: int = 3
    temperature: float = 0.3

    def complete(self, system, user, max_tokens=512):
        import requests

        last = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "stream": False,
                        "options": {"num_predict": max_tokens, "temperature": self.temperature},
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json()["message"]["content"]
            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"cannot reach Ollama at {self.host}; install it, start it, and run: ollama pull {self.model}"
                ) from e
            except Exception as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Ollama call failed after {self.max_retries} attempts: {last}")


class MockLLM(LLMClient):
    def complete(self, system, user, max_tokens=512):
        if "sub_problems" in system:
            return json.dumps({"sub_problems": ["mock sub-problem A", "mock sub-problem B"]})
        if "queries" in system:
            return json.dumps({"queries": ["mock query one", "mock query two"]})
        if "explanation" in system:
            return json.dumps({"explanation": "mock explanation"})
        if "problem_description" in system:
            return json.dumps({"problem_description": "I have a mock problem I cannot name."})
        return json.dumps({})


def make_llm_client(backend="ollama"):
    if backend == "ollama":
        return OllamaLLM()
    if backend == "anthropic":
        return AnthropicLLM()
    if backend == "mock":
        return MockLLM()
    raise ValueError(f"unknown backend {backend!r}")


def check_backend(backend="ollama"):
    if backend == "ollama":
        import requests

        try:
            models = [m["name"] for m in requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5).json().get("models", [])]
        except Exception as e:
            print(f"cannot reach Ollama: {e}")
            return False
        ok = any(m.split(":")[0] == OLLAMA_MODEL.split(":")[0] for m in models)
        print(f"Ollama reachable; models: {models}" + ("" if ok else f"; run: ollama pull {OLLAMA_MODEL}"))
        return ok
    if backend == "anthropic":
        ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
        print("ANTHROPIC_API_KEY set" if ok else "ANTHROPIC_API_KEY not set")
        return ok
    return True


def parse_json(raw):
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def call_json(client, system, user, max_tokens=512):
    raw = client.complete(system, user, max_tokens=max_tokens)
    try:
        return parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        raw = client.complete(
            system,
            user + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object, nothing else.",
            max_tokens=max_tokens,
        )
        return parse_json(raw)


def decompose(client, description):
    result = call_json(client, DECOMPOSE_SYSTEM, f"Problem description:\n{description}")
    return result.get("sub_problems") or [description]


def generate_queries(client, sub_problem):
    result = call_json(client, QUERY_GEN_SYSTEM, f"Sub-problem:\n{sub_problem}")
    return result.get("queries") or [sub_problem]


def explain(client, description, title, abstract):
    result = call_json(
        client,
        EXPLAIN_SYSTEM,
        f"Researcher's problem:\n{description}\n\nRetrieved paper title: {title}\nAbstract: {abstract}",
    )
    return result.get("explanation", "")


def mediated_queries(client, description):
    queries = []
    for sub in decompose(client, description):
        queries.extend(generate_queries(client, sub))
    return queries


def rewrite_only_query(client, description):
    queries = generate_queries(client, description)
    return queries[0] if queries else description


def synthetic_problem_description(client, title, abstract):
    result = call_json(client, SYNTH_PROBLEM_SYSTEM, f"Title: {title}\nAbstract: {abstract}", max_tokens=300)
    return result.get("problem_description", "")


def build_eval_set(corpus, client, n=EVAL_N, seed=cr.RANDOM_SEED, out=None):
    out = Path(out or EVAL_AUTO_JSON)
    sample = corpus.sample(n=min(n, len(corpus)), random_state=seed).reset_index(drop=True)
    records = []
    t0 = time.time()
    for i, row in sample.iterrows():
        description = synthetic_problem_description(client, row["title"], row["abstract"])
        records.append(
            {
                "gold_doc_id": int(row["doc_id"]),
                "source_title": row["title"],
                "problem_description": description,
            }
        )
        if (i + 1) % 10 == 0 or (i + 1) == len(sample):
            print(f"{i + 1}/{len(sample)} descriptions, {time.time() - t0:.0f}s")
    out.write_text(json.dumps(records, indent=2))
    return records


def run_experiment(corpus, indices, eval_set, client, k=TOP_K, out=None):
    out = Path(out or RESULTS_JSON)
    eval_set = [e for e in eval_set if e.get("problem_description", "").strip()]
    hits = {name: {"direct": [], "rewrite": [], "mediated": []} for name in indices}
    disagreements, per_example = [], []
    t0 = time.time()
    for i, ex in enumerate(eval_set):
        description, gold = ex["problem_description"], ex["gold_doc_id"]
        rewritten = rewrite_only_query(client, description)
        queries = mediated_queries(client, description)
        record = {"gold_doc_id": gold, "rewrite_query": rewritten, "mediated_queries": queries, "hits": {}}
        for name, index in indices.items():
            direct = index.search(description, k)
            rewrite = index.search(rewritten, k)
            sub_lists = [index.search(q, k) for q in queries] or [direct]
            mediated = cr.merge_ranked_lists(sub_lists, k)
            h = {
                "direct": int(gold in direct[:k]),
                "rewrite": int(gold in rewrite[:k]),
                "mediated": int(gold in mediated[:k]),
            }
            for cond, value in h.items():
                hits[name][cond].append(value)
            record["hits"][name] = h
            if h["mediated"] != h["direct"]:
                disagreements.append(
                    {
                        "retriever": name,
                        "problem_description": description,
                        "gold_title": ex["source_title"],
                        "mediated_queries": queries,
                        "mediated_recall": h["mediated"],
                        "direct_recall": h["direct"],
                    }
                )
        per_example.append(record)
        if (i + 1) % 10 == 0 or (i + 1) == len(eval_set):
            print(f"{i + 1}/{len(eval_set)} descriptions, {time.time() - t0:.0f}s")
    results = {
        "k": k,
        "n_examples": len(eval_set),
        "summary_recall": {
            name: {c: round(float(np.mean(v)), 4) for c, v in conds.items()} for name, conds in hits.items()
        },
        "n_disagreements": len(disagreements),
        "disagreement_examples": disagreements[:15],
        "per_example": per_example,
    }
    out.write_text(json.dumps(results, indent=2))
    return results


def summary_table(results):
    table = pd.DataFrame(results["summary_recall"]).T * 100
    table.columns = [c.capitalize() for c in table.columns]
    table.index.name = f"Recall@{results['k']} (%), n={results['n_examples']}"
    return table.round(1)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def binom_two_sided(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def headline(results):
    n = results["n_examples"]
    out = {}
    for retriever, conds in results["summary_recall"].items():
        out[retriever] = {}
        for cond, value in conds.items():
            hits = round(value * n)
            lo, hi = wilson(hits, n)
            out[retriever][cond] = {"hits": hits, "n": n, "recall": hits / n, "ci95": [lo, hi]}
    return out


def paired_counts(results):
    n = results["n_examples"]
    retrievers = list(results["summary_recall"])
    out = {}
    if "per_example" in results:
        out["exact"] = True
        for r in retrievers:
            wins = sum(e["hits"][r]["mediated"] > e["hits"][r]["direct"] for e in results["per_example"])
            losses = sum(e["hits"][r]["mediated"] < e["hits"][r]["direct"] for e in results["per_example"])
            out[r] = {
                "mediated_wins": wins,
                "mediated_losses": losses,
                "ties": n - wins - losses,
                "mcnemar_p": binom_two_sided(wins, losses),
            }
        return out

    stored = results["disagreement_examples"]
    wins = {r: sum(1 for d in stored if d["retriever"] == r and d["mediated_recall"] == 1) for r in retrievers}
    losses = {r: sum(1 for d in stored if d["retriever"] == r and d["mediated_recall"] == 0) for r in retrievers}
    net = {
        r: round(results["summary_recall"][r]["mediated"] * n) - round(results["summary_recall"][r]["direct"] * n)
        for r in retrievers
    }
    rest = results["n_disagreements"] - len(stored)
    solutions = []
    if len(retrievers) == 2:
        a, b = retrievers
        for aw in range(rest + 1):
            for al in range(rest + 1 - aw):
                for bw in range(rest + 1 - aw - al):
                    bl = rest - aw - al - bw
                    if wins[a] + aw - losses[a] - al == net[a] and wins[b] + bw - losses[b] - bl == net[b]:
                        solutions.append((aw, al, bw, bl))
    out["exact"] = len(solutions) == 1
    out["n_completions"] = len(solutions)
    if solutions:
        a, b = retrievers
        aw, al, bw, bl = solutions[0]
        for r, (w, l) in {a: (wins[a] + aw, losses[a] + al), b: (wins[b] + bw, losses[b] + bl)}.items():
            out[r] = {"mediated_wins": w, "mediated_losses": l, "ties": n - w - l, "mcnemar_p": binom_two_sided(w, l)}
    return out


STOP = set(
    "the a an of and to in for on with is are was were i im my me it its this that these those how what which be been "
    "as by or at from can could would should but not no have has had trying figure out about into than then them they "
    "their there here also just very really much many more most some such like need want know think stuck problem "
    "having trouble tried using use way ways get getting make makes made see seem seems something anything everything "
    "doing done does".split()
)


def content_words(text):
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOP}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def bm25_rank_analysis(corpus, eval_set, index):
    ids = np.array(corpus["doc_id"].tolist())
    rows = {int(r.doc_id): r for r in corpus.itertuples()}
    ranks, cover_abs, cover_title = [], [], []
    for ex in eval_set:
        order = np.argsort(-index.scores(ex["problem_description"]))
        ranks.append(int(np.where(ids[order] == ex["gold_doc_id"])[0][0]) + 1)
        words = content_words(ex["problem_description"])
        gold = rows[ex["gold_doc_id"]]
        cover_abs.append(len(words & content_words(gold.title + " " + gold.abstract)) / max(len(words), 1))
        title_words = content_words(gold.title)
        cover_title.append(len(title_words & words) / max(len(title_words), 1))
    ranks = np.array(ranks)
    return {
        "n": len(ranks),
        "ranks": ranks.tolist(),
        "recall_at": {str(k): int((ranks <= k).sum()) for k in (1, 5, 10, 50, 100, 500, 1000)},
        "median_rank": float(np.median(ranks)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "coverage_abstract": cover_abs,
        "coverage_title": cover_title,
        "mean_coverage_abstract": float(np.mean(cover_abs)),
        "mean_coverage_title": float(np.mean(cover_title)),
        "max_coverage_title": float(np.max(cover_title)),
        "spearman_coverage_vs_rank": spearman(np.array(cover_abs), ranks.astype(float)),
    }


def reciprocal_rank_fusion(lists, k=10, c=60):
    scores = {}
    for lst in lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (c + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])[:k]


def merge_check(results, corpus, index):
    title_to_id = dict(zip(corpus["title"], corpus["doc_id"]))
    seen, rows = set(), []
    for d in results["disagreement_examples"]:
        if d["gold_title"] in seen:
            continue
        seen.add(d["gold_title"])
        gold = int(title_to_id[d["gold_title"]])
        lists = [index.search(q, 10) for q in d["mediated_queries"]]
        merged = cr.merge_ranked_lists(lists, 10)
        top1 = {lst[0] for lst in lists}
        rows.append(
            {
                "gold_title": d["gold_title"],
                "n_queries": len(lists),
                "queries_containing_gold": int(sum(gold in lst for lst in lists)),
                "direct": int(gold in index.search(d["problem_description"], 10)),
                "hit_count_merge": int(gold in merged),
                "rrf_merge": int(gold in reciprocal_rank_fusion(lists)),
                "top1_share_of_merged": sum(x in top1 for x in merged) / len(merged),
            }
        )
    return {
        "n_cases": len(rows),
        "direct_hits": sum(r["direct"] for r in rows),
        "hit_count_hits": sum(r["hit_count_merge"] for r in rows),
        "rrf_hits": sum(r["rrf_merge"] for r in rows),
        "mean_top1_share_of_merged_top10": float(np.mean([r["top1_share_of_merged"] for r in rows])),
        "rows": rows,
    }


def taxonomy(annotations):
    cases = annotations["cases"]
    counts = Counter(label for c in cases for label in c["group_labels"])
    total = sum(counts.values())
    return {
        "n_cases": len(cases),
        "n_groups": total,
        "counts": dict(counts),
        "share": {k: v / total for k, v in counts.items()},
        "cases_with_drift_or_filler": sum(1 for c in cases if any(l in ("DRIFT", "FILLER") for l in c["group_labels"])),
        "subproblem_count_distribution": {
            str(k): v for k, v in sorted(Counter(c["n_queries"] // 3 for c in cases).items())
        },
    }


def knn_purity(embeddings, corpus, k=10, chunk=2000):
    codes = corpus["domain"].astype("category")
    dom = np.asarray(codes.cat.codes)
    names = list(codes.cat.categories)
    per_doc = np.zeros(len(embeddings))
    for i in range(0, len(embeddings), chunk):
        sims = embeddings[i : i + chunk] @ embeddings.T
        sims[np.arange(sims.shape[0]), np.arange(i, i + sims.shape[0])] = -1.0
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        per_doc[i : i + chunk] = (dom[idx] == dom[i : i + chunk, None]).mean(axis=1)
    return {
        "k": k,
        "mean_purity": float(per_doc.mean()),
        "per_domain": {names[c]: float(per_doc[dom == c].mean()) for c in range(len(names))},
    }


def eval_set_stats(eval_set, corpus):
    words = [len(e["problem_description"].split()) for e in eval_set]
    domains = corpus.set_index("doc_id").loc[[e["gold_doc_id"] for e in eval_set], "domain"]
    return {
        "n": len(eval_set),
        "words_mean": float(np.mean(words)),
        "words_min": int(min(words)),
        "words_max": int(max(words)),
        "mention_paper": int(sum("paper" in e["problem_description"].lower() for e in eval_set)),
        "gold_domain_counts": {k: int(v) for k, v in domains.value_counts().items()},
    }


def run_analysis(corpus, embeddings, eval_set, results, indices, annotations_path=None, out=None):
    out = Path(out or ANALYSIS_JSON)
    annotations_path = Path(annotations_path or ANNOTATIONS_JSON)
    eval_set = [e for e in eval_set if e.get("problem_description", "").strip()]
    analysis = {"headline": headline(results), "paired_mediated_vs_direct": paired_counts(results)}
    if annotations_path.exists():
        analysis["taxonomy"] = taxonomy(json.loads(annotations_path.read_text()))
    bm25 = indices["bm25"]
    analysis["bm25_direct"] = bm25_rank_analysis(corpus, eval_set, bm25)
    analysis["merge_check_bm25"] = merge_check(results, corpus, bm25)
    analysis["knn_purity_minilm"] = knn_purity(embeddings, corpus)
    analysis["eval_set"] = eval_set_stats(eval_set, corpus)
    out.write_text(json.dumps(analysis, indent=2))
    return analysis


def print_summary(analysis):
    print("Recall@10 with 95% Wilson intervals")
    for retriever, conds in analysis["headline"].items():
        for cond, v in conds.items():
            lo, hi = v["ci95"]
            print(f"  {retriever:13s} {cond:9s} {v['hits']:2d}/{v['n']}  {100 * v['recall']:5.1f}%  [{100 * lo:.0f}, {100 * hi:.0f}]")
    paired = analysis["paired_mediated_vs_direct"]
    print(f"\nMediated vs direct (exact counts: {paired.get('exact')})")
    for retriever in analysis["headline"]:
        if retriever in paired:
            p = paired[retriever]
            print(f"  {retriever:13s} wins {p['mediated_wins']:2d}  losses {p['mediated_losses']:2d}  ties {p['ties']:2d}  McNemar p = {p['mcnemar_p']:.4f}")
    b = analysis["bm25_direct"]
    print(f"\nBM25 direct: median gold rank {b['median_rank']:.0f}, MRR {b['mean_reciprocal_rank']:.3f}, hits at k {b['recall_at']}")
    print(f"  Spearman(word overlap, rank) = {b['spearman_coverage_vs_rank']:.3f}")
    m = analysis["merge_check_bm25"]
    print(f"\nMerge check on {m['n_cases']} stored cases (BM25): direct {m['direct_hits']}, hit count {m['hit_count_hits']}, RRF {m['rrf_hits']}")
    if "taxonomy" in analysis:
        t = analysis["taxonomy"]
        print(f"\nSub-problem labels: {t['counts']}; {t['cases_with_drift_or_filler']}/{t['n_cases']} cases with drift or filler")
    print(f"\nMiniLM 10-NN domain purity: {analysis['knn_purity_minilm']['mean_purity']:.3f}")


def show_example(corpus, indices, example, k=3):
    titles = dict(zip(corpus["doc_id"], corpus["title"]))
    print("gold paper :", example["gold_title"])
    print("description:", example["problem_description"][:260], "...")
    queries = example["mediated_queries"]
    print(f"{len(queries)} generated queries, for example: {queries[:3]}")
    for name, index in indices.items():
        direct = index.search(example["problem_description"], k)
        merged = cr.merge_ranked_lists([index.search(q, 10) for q in queries], k)
        print(f"\n{name} direct   :", [titles[d][:55] for d in direct])
        print(f"{name} mediated :", [titles[d][:55] for d in merged])


def plot_recall(analysis):
    import matplotlib.pyplot as plt

    h = analysis["headline"]
    retrievers = [(k, n) for k, n in [("bm25", "BM25"), ("dense_minilm", "MiniLM (dense)")] if k in h]
    conds = [("direct", "Direct", "#7F8C9B"), ("rewrite", "Rewrite only", "#E0A458"), ("mediated", "Mediated", "#2A6F97")]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    width = 0.25
    for j, (ck, label, colour) in enumerate(conds):
        xs, ys, lo, hi = [], [], [], []
        for i, (rk, _) in enumerate(retrievers):
            v = h[rk][ck]
            xs.append(i + (j - 1) * width)
            ys.append(100 * v["recall"])
            lo.append(100 * (v["recall"] - v["ci95"][0]))
            hi.append(100 * (v["ci95"][1] - v["recall"]))
        ax.bar(xs, ys, width * 0.92, color=colour, label=label, yerr=[lo, hi], capsize=2)
        for x, y, e in zip(xs, ys, hi):
            ax.text(x, y + e + 1.2, f"{y:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(retrievers)))
    ax.set_xticklabels([n for _, n in retrievers])
    ax.set_ylabel("Recall@10 (%)")
    ax.set_ylim(0, 62)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.show()


def plot_rank(analysis):
    import matplotlib.pyplot as plt

    b = analysis["bm25_direct"]
    ranks = np.array(b["ranks"])
    cover = np.array(b["coverage_abstract"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    xs = np.sort(ranks)
    a1.step(xs, 100 * np.arange(1, len(xs) + 1) / len(xs), where="post", color="#2A6F97", lw=1.6)
    a1.set_xscale("log")
    a1.axhline(100 * (ranks <= 10).mean(), color="#C8553D", ls="--", lw=0.9, label=f"Recall@10 = {100 * (ranks <= 10).mean():.1f}%")
    a1.axvline(b["median_rank"], color="#6B6B6B", ls="--", lw=0.9, label=f"median rank = {b['median_rank']:,.0f}")
    a1.set_xlabel("rank of the gold paper (log scale)")
    a1.set_ylabel("% of descriptions")
    a1.set_title("BM25, direct query")
    a1.legend(frameon=False, loc="upper left")
    hit = ranks <= 10
    a2.scatter(100 * cover[~hit], ranks[~hit], s=14, color="#6B6B6B", label="not in top 10")
    a2.scatter(100 * cover[hit], ranks[hit], s=22, color="#2E8B6F", label="in top 10")
    a2.set_yscale("log")
    a2.set_xlabel("description words found in the gold abstract (%)")
    a2.set_ylabel("rank of the gold paper")
    a2.set_title(f"overlap vs. rank (Spearman {b['spearman_coverage_vs_rank']:.2f})")
    a2.legend(frameon=False, loc="lower left")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.show()


def plot_taxonomy(annotations):
    import matplotlib.pyplot as plt

    colours = {"ON": "#2E8B6F", "PARTIAL": "#E0C45A", "DRIFT": "#C8553D", "FILLER": "#6B6B6B"}
    cases = annotations["cases"]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for i, c in enumerate(cases):
        for j, label in enumerate(c["group_labels"]):
            ax.barh(i, 0.9, left=j, color=colours[label], edgecolor="white", height=0.78)
        o = c["mediated_vs_direct"]
        tag = []
        if "bm25" in o:
            tag.append("B" + ("+" if o["bm25"] == "win" else "-"))
        if "dense_minilm" in o:
            tag.append("D" + ("+" if o["dense_minilm"] == "win" else "-"))
        ax.text(4.15, i, " ".join(tag), va="center", fontsize=8)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([c["id"] for c in cases])
    ax.invert_yaxis()
    ax.set_xticks([0.45, 1.45, 2.45, 3.45])
    ax.set_xticklabels(["sub 1", "sub 2", "sub 3", "sub 4"])
    ax.set_xlim(0, 4.9)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[k]) for k in colours]
    ax.legend(handles, list(colours), frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_all(analysis):
    plot_recall(analysis)
    plot_rank(analysis)
    if ANNOTATIONS_JSON.exists():
        plot_taxonomy(json.loads(ANNOTATIONS_JSON.read_text()))


def ask_relevant(i, n, title, abstract):
    print(f"\n[{i}/{n}] {title}\n    {abstract[:220]}{'...' if len(abstract) > 220 else ''}")
    while True:
        answer = input("    relevant? [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def manual_check(corpus, indices, client, personal_csv=None, k=TOP_K_MANUAL, retrievers=None, out=None):
    out = Path(out or MANUAL_JSON)
    problems = pd.read_csv(personal_csv or PERSONAL_CSV)["problem_description"].dropna().tolist()
    chosen = {name: idx for name, idx in indices.items() if retrievers is None or name in retrievers}
    titles = dict(zip(corpus["doc_id"], corpus["title"]))
    abstracts = dict(zip(corpus["doc_id"], corpus["abstract"]))
    results = {"k": k, "judgements": []}
    for p_idx, problem in enumerate(problems, 1):
        print(f"\nPROBLEM {p_idx}/{len(problems)}: {problem}")
        queries = mediated_queries(client, problem)
        for name, index in chosen.items():
            ranked = cr.merge_ranked_lists([index.search(q, k) for q in queries], k)
            print(f"\n-- {name} --")
            marks = []
            for i, doc_id in enumerate(ranked, 1):
                relevant = ask_relevant(i, len(ranked), titles[doc_id], abstracts[doc_id])
                marks.append({"doc_id": int(doc_id), "title": titles[doc_id], "relevant": relevant})
            results["judgements"].append(
                {
                    "problem": problem,
                    "retriever": name,
                    "queries": queries,
                    "results": marks,
                    "precision_at_k": round(sum(m["relevant"] for m in marks) / max(len(marks), 1), 4),
                }
            )
            out.write_text(json.dumps(results, indent=2))
    return results


def manual_summary(path=None):
    path = Path(path or MANUAL_JSON)
    if not path.exists():
        print("no manual relevance check saved yet")
        return None
    data = json.loads(path.read_text())
    table = {}
    for j in data["judgements"]:
        row = table.setdefault(j["retriever"], {"problems": 0, "sum_p": 0.0, "any_hit": 0})
        row["problems"] += 1
        row["sum_p"] += j["precision_at_k"]
        row["any_hit"] += int(any(m["relevant"] for m in j["results"]))
    out = pd.DataFrame(table).T
    out["mean_P@k"] = (out["sum_p"] / out["problems"]).round(3)
    out[["problems", "any_hit"]] = out[["problems", "any_hit"]].astype(int)
    return out[["problems", "mean_P@k", "any_hit"]]
