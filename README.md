quantumQuery

quantumQuery tests whether a large language model placed between a researcher and a search engine helps when the researcher can only describe a problem informally. An informal description is split into sub-problems, each is rewritten as literature-style search queries, the queries are run against a 20,000-paper physics corpus built from arXiv using BM25 or a MiniLM dense retriever, and the results are merged. The notebook builds the corpus, encodes it, creates 60 test descriptions, compares direct, rewrite-only and mediated retrieval with Recall@10, and runs the statistics and error analysis. The language model is Llama 3.1 8B, served locally by Ollama.

What each item contains

quantumquery_pipeline.ipynb
  The whole project from start to end, in Part 1 to Part 6.

src/
  corpus_retrieval.py   corpus building, encoding, checks, BM25 and dense search
  llm_experiment.py     language model calls, test set, experiment, statistics, plots, manual check
  requirements.txt      Python packages

data/
  corpus (csv and json), embeddings (npy), test descriptions and results (json), problem list for the manual check (csv)

model/
  the saved all-MiniLM-L6-v2 sentence encoder used by the dense retriever. It is a pretrained model and was not trained or fine-tuned here.

README.md
  this file

How to reproduce

1. Install Python 3.10 or newer, then run: pip install -r src/requirements.txt
   For a GPU, install the CUDA build of PyTorch first (see pytorch.org).
2. Install Ollama (ollama.com), then run: ollama pull llama3.1:8b
3. Open quantumquery_pipeline.ipynb in Jupyter and run all cells. With the data folder as provided, it loads the saved results and repeats the analysis in a few minutes.
4. To rebuild everything from scratch (download, encode, generate the test set, rerun the experiment), set REBUILD = True in Part 1. This takes about 30 minutes on a GPU.
5. For the manual relevance check, set MANUAL_CHECK = True in Part 1 and answer y or n for each paper shown.
6. If the model folder is empty, the encoder is downloaded into it the first time it is needed.
