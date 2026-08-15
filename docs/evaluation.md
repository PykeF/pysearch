# Retrieval evaluation and benchmarks

Everything measured about this project, with the caveats that make the numbers
readable.

> **Read this first.** The retrieval evaluation uses a small synthetic corpus
> written for this project: **67 documents, 32 labelled queries**. It supports
> statements about *these* queries and nothing wider. It is not MS MARCO, not
> BEIR, and not a benchmark. Performance figures are local development
> measurements on a single machine and say nothing about production throughput.

- [Dataset construction](#dataset-construction)
- [The development / evaluation split](#the-development--evaluation-split)
- [Metrics](#metrics)
- [Parameter development](#parameter-development)
- [Held-out results](#held-out-results)
- [Why fusion lost, exactly](#why-fusion-lost-exactly)
- [Per-query results](#per-query-results)
- [What retrieval costs](#what-retrieval-costs)
- [Startup recovery cost](#startup-recovery-cost)
- [Distributed hybrid latency](#distributed-hybrid-latency)
- [Reproducing all of this](#reproducing-all-of-this)
- [Limitations](#limitations)

## Dataset construction

The corpus lives in `scripts/evaluation_data.py` and is written to make the
three retrieval modes distinguishable rather than to flatter any of them. 67
documents across topic clusters:

| Cluster | Documents | Purpose |
| --- | --- | --- |
| Vehicles | `veh-1…8` | paraphrase targets with no shared vocabulary |
| Information retrieval | `ir-1…8` | domain topic |
| Distributed systems | `sys-1…8` | domain topic |
| Cooking | `cook-1…6` | topically distant; a source of lexical false positives |
| Error codes | `err-1…6` | `ERR_CONN_RESET_1042/1043/1044` — deliberate near-duplicates |
| Part numbers | `part-1…6` | `PX-9174-Q/9175-Q/9176-Q` — identifier discrimination |
| RFC references | `rfc-1…5` | exact-token retrieval |
| Version numbers | `ver-1…4` | `4.2.0 / 4.2.1 / 4.3.0 / 5.0.0` distractors |
| Product names | `prod-1…5` | Halberd, Marlinspike, Tanglefoot, Blackthorn, Wintergreen |
| Operations | `ops-1…6` | mixed vocabulary |
| General distractors | `gen-1…5` | unrelated content that shares common words |

The near-duplicates matter: a corpus where every document is obviously distinct
would make the `distractor` category meaningless.

Each labelled query declares its relevant documents and a **category**:

| Category | Meaning |
| --- | --- |
| `semantic` | Paraphrase — little or no vocabulary shared with the answer |
| `lexical` | Exact term or identifier match |
| `mixed` | Partial vocabulary overlap |
| `distractor` | Correct answer must be picked out from near-identical neighbours |

Dataset integrity is enforced by tests in `tests/unit/test_evaluation.py`: every
labelled document exists, the two query sets do not overlap, every category is
represented in the evaluation set with at least four queries, and the deliberate
near-duplicates are present.

## The development / evaluation split

The queries are split **before any measurement was taken**, and the split is
respected:

- **12 development queries** chose the RRF constant and the candidate depth.
- **20 evaluation queries** were then measured once, with those values frozen.

Tuning on the queries you report is how a measurement becomes an advertisement.
The development set is optimistic by construction — it is scored on the queries
that selected the parameters — and is reported separately and labelled as such.

**No evaluation query was rewritten, removed or re-labelled after seeing a
result.** The four queries where hybrid lost are still in the set, and they are
analysed below rather than deleted.

## Metrics

| Metric | Definition |
| --- | --- |
| **Recall@k** | Fraction of a query's relevant documents appearing in the top `k` |
| **MRR** | Mean reciprocal rank of the *first* relevant document (`1/rank`, or 0) |

Both are computed over a retrieval depth of 20. MRR is the headline because most
of these queries have a single correct answer, where MRR is exactly "how far
down the list was it".

## Parameter development

Development queries only. Baselines on this set: BM25 **MRR 0.771**, semantic
**MRR 0.885**.

| rrf_k | depth | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | --- | --- |
| 10 | 10 | 0.833 | 0.903 | 0.838 |
| 30 | 10 | 0.833 | 0.861 | 0.838 |
| 60 | 10 | 0.833 | 0.861 | 0.838 |
| 10 | 20 | 0.833 | 0.861 | 0.833 |
| 30 | 20 | 0.833 | 0.861 | 0.833 |
| 60 | 20 | 0.833 | 0.861 | 0.833 |
| 10 | 50 | 0.833 | 0.861 | 0.833 |
| 30 | 50 | 0.833 | 0.861 | 0.833 |
| 60 | 50 | 0.833 | 0.861 | 0.833 |

**The sweep is flat.** MRR spans 0.833–0.838 across all nine settings — a
difference of one rank position on one of twelve queries. Recall@5 is identical
everywhere.

So the measurement did not choose the defaults; it showed there was nothing to
choose. `rrf_k = 60` (the value from the original RRF paper) and a depth of
`5 × limit` were kept for stability rather than to chase noise. Reporting this as
"tuning improved hybrid" would be false — the honest reading is that on a corpus
this small, these parameters do not matter.

Note also that hybrid at its best development setting (0.838) already trails
semantic alone (0.885) here, before the held-out set was ever touched.

## Held-out results

20 evaluation queries, parameters frozen at `rrf_k=60`, candidate depth 50.

| Mode | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | --- |
| BM25 | 0.725 | 0.725 | 0.732 |
| semantic | 0.925 | 0.933 | **1.000** |
| hybrid | 0.858 | 0.933 | 0.865 |

### Hybrid did not beat semantic alone on this set

That is the headline finding, and it is not being softened. Semantic retrieval
scored a perfect 1.000 MRR — it placed a relevant document first for all twenty
queries — and fusing BM25 into it made the ranking *worse*, not better.

By category (MRR; sample sizes are small, so read these as direction only):

| Category | n | BM25 | semantic | hybrid |
| --- | --- | --- | --- | --- |
| semantic | 6 | 0.19 | 1.00 | **0.55** |
| lexical | 6 | 1.00 | 1.00 | 1.00 |
| mixed | 4 | 1.00 | 1.00 | 1.00 |
| distractor | 4 | 0.88 | 1.00 | **1.00** |

The category breakdown is where the result becomes informative rather than
merely disappointing:

- On **lexical**, **mixed** and **distractor** queries, hybrid matched the best
  input every time — and on `distractor` it *recovered* a BM25 failure
  (`"upgrading without downtime"`, BM25 0.50 → hybrid 1.00).
- All of the damage is in the **semantic** category: 4 of 6 paraphrases got
  worse.

Hybrid ranked below its best single-mode input on **4 of 20 queries**, all
`semantic`.

## Why fusion lost, exactly

The mechanism is worth stating because it is the honest lesson of the phase.
Take `"searching by meaning rather than keywords"`:

```text
BM25      cook-4 (5.87)  cook-6 (5.64)  gen-4 (5.64)  ...   ir-3 absent
semantic  ir-3 (0.435)   rfc-1 (0.221)  ir-2 (0.219)  ...
hybrid    cook-4 0.0313 (lex 1, sem 7)  ...  ir-3 0.0164 (lex none, sem 1)
```

`cook-4` is *"season in layers **rather than** all at the end"*. It matches on
`rather` and `than`. BM25 here is not merely unhelpful — it is **confidently
wrong**, and RRF weighs both retrievers equally, so three wrong documents each
contributing `1/(60 + small rank)` outrank the correct document's single
contribution.

The contrast with `"making dinner"` makes it sharp: there BM25 matched **nothing
at all**, contributed nothing, and hybrid equalled semantic exactly (1.00).

> **BM25 finding nothing is harmless to fusion. BM25 finding the wrong thing is
> what hurts.**

The root cause traces directly to a deliberate Phase 1 decision: **there is no
stop-word filtering**, so common words like `rather` and `than` carry real IDF
weight and produce spurious lexical matches. RRF is scale-free but it is not
quality-aware — it cannot tell a confident retriever from a correct one.

This limitation is documented rather than patched. Adding a stop-word list to
make hybrid win would be tuning the system to the evaluation set, which is the
thing the dev/eval split exists to prevent.

### Two further caveats

- Semantic scored a **perfect 1.000 MRR** on this corpus, which leaves fusion
  nothing to gain and everything to lose. A corpus this small and this cleanly
  separated flatters the semantic path.
- The evaluation contains **no query where BM25 beats semantic**, so it cannot
  demonstrate the case where fusion would help most. That is a limitation of the
  dataset, not evidence that lexical retrieval is redundant.

Both mean this evaluation is better at showing *how* fusion fails than at
estimating how well it would do on a realistic corpus.

## Per-query results

Reciprocal rank of the first relevant document. `←` marks the four losses.

| Query | Category | BM25 | semantic | hybrid |
| --- | --- | --- | --- | --- |
| `fixing a broken engine` | semantic | 0.09 | 1.00 | 0.17 ← |
| `searching by meaning rather than keywords` | semantic | 0.00 | 1.00 | 0.14 ← |
| `surviving the loss of a machine` | semantic | 0.05 | 1.00 | 0.50 ← |
| `splitting data across machines` | semantic | 0.00 | 1.00 | 0.50 ← |
| `making dinner` | semantic | 0.00 | 1.00 | 1.00 |
| `making retries safe` | semantic | 1.00 | 1.00 | 1.00 |
| `ERR_CHECKSUM_3301` | lexical | 1.00 | 1.00 | 1.00 |
| `ERR_CONN_RESET_1044` | lexical | 1.00 | 1.00 | 1.00 |
| `PX-9176-Q` | lexical | 1.00 | 1.00 | 1.00 |
| `RFC 7519` | lexical | 1.00 | 1.00 | 1.00 |
| `Blackthorn` | lexical | 1.00 | 1.00 | 1.00 |
| `release 5.0.0` | lexical | 1.00 | 1.00 | 1.00 |
| `PX-9174-Q battery failure` | mixed | 1.00 | 1.00 | 1.00 |
| `Halberd node draining` | mixed | 1.00 | 1.00 | 1.00 |
| `RFC 9111 response reuse` | mixed | 1.00 | 1.00 | 1.00 |
| `combining two rankings` | mixed | 1.00 | 1.00 | 1.00 |
| `battery replacement procedure` | distractor | 1.00 | 1.00 | 1.00 |
| `running out of disk space` | distractor | 1.00 | 1.00 | 1.00 |
| `display calibration` | distractor | 1.00 | 1.00 | 1.00 |
| `upgrading without downtime` | distractor | 0.50 | 1.00 | **1.00** |

The last row is the one case where fusion did what it is supposed to do: BM25
placed the answer second, and fusion promoted it to first.

### Observed live, through the coordinator

On the 7-process cluster with the real pinned model:

```text
"car maintenance"              /search          total=1   prod-1  (matches "maintenance")
                               /search/semantic veh-1 0.601, veh-2 0.525
                               /search/hybrid   prod-1 first  <- the failure mode above

"ERR_CONN_RESET_1044"          /search          err-3 11.15, err-1 7.63, err-2 7.63
                               /search/semantic err-3 0.793, err-2 0.763, err-1 0.760
                               /search/hybrid   err-3 first (lex 1, sem 1)

"PX-9174-Q battery failure"    /search/hybrid   part-1 first (lex 1, sem 1)
```

## What retrieval costs

Local development measurements on one machine (Apple Silicon, Python 3.13) with
synthetic 40-word documents, from `scripts/semantic_benchmark.py`. **Not a
benchmark.**

| Documents | Index (s) | Rebuild: lexical | Rebuild: semantic | Vector search | Vector memory |
| --- | --- | --- | --- | --- | --- |
| 100 | 0.052 | 0.002 s | 0.004 s | 0.033 ms | 0.1 MB |
| 1,000 | 0.537 | 0.027 s | 0.040 s | 0.259 ms | 1.0 MB |
| 5,000 | 2.779 | 0.158 s | 0.181 s | 1.978 ms | 5.1 MB |

Model load 0.209 s; embedding ~19,400 documents/s batched; query embedding
0.052 ms. Vector memory is `N × d × 4` bytes for float32 at 256 dimensions,
excluding Python overhead.

Two design decisions were made *pending these numbers*:

- **Re-embedding at startup costs about the same as the lexical rebuild**
  (0.181 s vs 0.158 s at 5,000 documents), which is why vectors are not
  persisted. If that ratio changed by an order of magnitude, persisting them
  would start to pay.
- **Vector search grows linearly** — 2 ms at 5,000 documents implies tens of
  milliseconds at a hundred thousand. That column is what would eventually
  justify an approximate index.

## Startup recovery cost

From `scripts/rebuild_benchmark.py`, lexical only, synthetic 40-word documents:

| Documents | Index (s) | Rebuild (s) | Documents/s rebuilt |
| --- | --- | --- | --- |
| 100 | 0.037 | 0.003 | 36,706 |
| 1,000 | 0.394 | 0.023 | 43,546 |
| 10,000 | 3.860 | 0.276 | 36,185 |

Rebuild is roughly linear and runs an order of magnitude faster than the original
indexing, because indexing pays one fsync per document while recovery is a single
sequential read. Snapshotting the index would only become worth its consistency
cost once this column stopped being acceptable.

## Distributed hybrid latency

Measured in-process against a live 7-process cluster (3 primaries, 3 replicas,
1 coordinator), comparing the real concurrent implementation against a
sequential path using the same lock, the same network and the same code:

| Execution | Median latency |
| --- | --- |
| Sequential | 10.59 ms |
| Concurrent (`asyncio.gather`) | 9.60 ms |

**9.3% median saving**, range −1.4% to +15.4%. Rankings were confirmed identical
between the two strategies.

The saving is real but well short of the `max(lexical, semantic)` an ideal
overlap would give, because the shard nodes serialize the two overlapping
requests behind their own per-node locks. That is the honest reading: the
coordinator overlaps its waiting, and the shards then partly re-serialize it.

## Reproducing all of this

Every number above comes from a committed script. None of them require editing
source code, and none of them modify the evaluation data.

```bash
uv sync --extra semantic
```

Held-out evaluation — the results table, category table and per-query table:

```bash
uv run --extra semantic python scripts/evaluate_retrieval.py
```

Parameter development — the RRF-k and candidate-depth sweep:

```bash
uv run --extra semantic python scripts/evaluate_retrieval.py --develop
```

Embedding, rebuild and vector-search costs:

```bash
uv run --extra semantic python scripts/semantic_benchmark.py
```

Lexical startup-recovery cost (no model needed):

```bash
uv run python scripts/rebuild_benchmark.py
```

The evaluation scripts build an in-memory engine and never write to
`scripts/evaluation_data.py`, so the held-out set cannot drift as a side effect
of running them.

## Limitations

Stated together, because any one of them alone would be misleading:

- **67 documents.** Real corpora are six or more orders of magnitude larger, and
  BM25's statistics behave differently at scale.
- **Synthetic, single-author.** The documents and the queries were written by
  the same person, which is a known way to accidentally encode the answer.
- **32 queries** total, and per-category samples of 4–6. A single query moving
  one rank position visibly changes a category average.
- **Single relevance judgement per query**, binary, with no graded relevance.
- **No query where BM25 beats semantic**, so the strongest case for fusion is
  untested.
- **Semantic scores 1.000**, leaving no headroom; a harder corpus would be
  needed to see whether fusion helps when neither input is already perfect.
- **The embedding model is static** (`model2vec`), measurably weaker than a
  transformer. Results would shift with a stronger encoder.
- **Performance numbers are single-machine, single-run** development
  measurements without statistical treatment beyond a median.

The right conclusion from this evaluation is *"on this small set, fusion hurt
paraphrase queries via BM25 false positives, and the mechanism is understood"* —
not *"hybrid search does not work"*.
