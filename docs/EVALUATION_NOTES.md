# Evaluation notes: Tasks 2.2 to 2.4

Every metric below comes from a saved results file in `results/`. Spot checks
that are qualitative are labelled as such. Anything not yet measured says so.

## 2.2 Confidence and citations

**What changed**

* The original prompt asked for `source_chunks_used` but never showed the model
  any chunk id, so any chunk citation it produced was invented. Prompt `v2`
  labels every chunk, graph path and image with a bracketed id.
* Every citation is checked against the packed context. An entry counts as
  present if it is a shown id or a verbatim quote of at least 12 characters
  from the context; everything else is dropped and counted as fabricated.
* Confidence is computed from retrieval evidence (retriever agreement, top-hit
  margin, graph support) with thresholds fixed before evaluation. The model's
  self-report is kept as `confidence_self_reported`.

**Measured so far (retrieval only, no LLM)**

On the 50 golden questions, evidence confidence from text evidence alone
(`evidence_confidence_text_only` in `results/retrieval_fixed_r1.json`) is
compared with the original self-report of 49 high / 0 medium / 1 low. With the
default RRF fusion it is 23 high / 27 medium / 0 low.

Limitation: the margin component depends on the fusion score scale. The same
questions score 48 high / 2 medium under weighted fusion and 50 low under the
original broken fusion. Evidence confidence is only comparable within one
fusion mode. Whether it tracks answer quality needs the answer-level run.

**Not yet measured (needs a Groq key)**

* `fabricated_citation_rate` for prompts v1 (baseline) and v2.
* The calibration plot: `graphrag eval --mode answer --out results/answer.json`
  then `graphrag calibration results/answer.json`.

## 2.3 Conflict detection

The original check flags a conflict when a chunk contains the literal string
`not <entity>` or `unlike <entity>`. It is replaced by an NLI cross-encoder
(`cross-encoder/nli-deberta-v3-small`) over pairs of claims that come from
different chunks, mention the same entity, and are semantically related.

Measured with `graphrag conflicts` (results: `results/conflicts.json`). Both
methods see the same top-5 chunks per golden question and the same entities,
extracted from the question. The notebook used graph node names as entities;
that needs Neo4j and was not used.

| Method | Corpus flags (50 questions) | Planted contradictions caught | Planted paraphrases flagged |
|---|---|---|---|
| Substring check (original) | 0 | 0 / 10 | 0 / 10 |
| NLI, no relatedness gate | 8 | 8 / 10 | 0 / 10 |
| NLI, gate 0.6 (default) | 5 | 8 / 10 | 0 / 10 |
| NLI, gate 0.7 | 1 | 5 / 10 | 0 / 10 |
| NLI, gate 0.8 | 0 | 4 / 10 | 0 / 10 |

How to read this honestly:

* **The corpus flags are false positives.** All flags at every gate were read
  manually; none is a genuine contradiction. Lecture slides from one course
  rarely contradict each other, so the corpus can show false alarms but cannot
  show whether real contradictions are caught.
* **The planted set measures recall.** `data/golden_qa/planted_conflicts.json`
  is synthetic: ten claims quoted from the corpus, each paired with a
  contradicting sentence written for the test, plus ten consistent
  paraphrases. It is small, so 8 / 10 has wide uncertainty.
* **How the default was chosen.** 0.6 is the highest gate that kept planted
  recall at the ungated level. That choice used only the planted set.
* **Two defects were fixed before the sweep, after reading the first run's
  flags.** Claim splitting broke slide sentences at PDF line wraps, producing
  fragments; and the model labelled unrelated pairs as contradictions in one
  direction. Lines are now re-joined, claims need at least six words, and a
  pair must be contradictory in both directions. The first run, with neither
  fix, flagged 19 corpus pairs, none genuine.

## 2.4 Scalability

* **Image dedup** uses a BK-tree over Hamming distance. On 500 random 64-bit
  hashes it returns exactly the linear scan's answers at radii 0, 5, 9 and 20
  (`tests/test_dedupe.py`). `graphrag ingest` on the original corpus still
  keeps 14 of 136 images. For random 64-bit hashes
  at radius 9 its pruning is modest; no large speed-up is claimed.
* **Entity linking** goes lexical (Neo4j full-text index) to semantic (cached
  node-name embeddings) to fuzzy only for misses, and node names are fetched
  once per process instead of on every query. Unit-tested against a fake
  driver; not yet run against a live graph.
* **Query entities** come from GLiNER plus the quoted-span and acronym regexes.
  Qualitative spot check, not saved: for "What is the difference between local trust and global trust
  on Twitter?" GLiNER returns *local trust*, *global trust* and *Twitter*; for
  the MBTI question it returns only *social media*, and the acronym regex adds
  *MBTI*. Its effect on graph recall is not yet measured.
* **Path expansion** ranks by hop count then edge support before `LIMIT`, and
  renders every node and edge direction. The original rendered a 2-hop path as
  `start -[first relation]-> end`, hiding the middle node.
* **Triple extraction** is async with bounded concurrency, exponential backoff
  with jitter, and a completed-index checkpoint. Unit-tested with a fake LLM;
  wall-clock speed-up against Groq not yet measured.

## Task 3: ablation, reranker, router

### Ablation (`graphrag ablation`, raw results in `results/ablation.json`)

Retrieval-only rows; 50 golden questions; top 5 text chunks; 3 runs per row;
all caches off; CPU only. Retrieval metrics are identical across runs, so only
latency has spread.

| Config | Recall@5 | MRR | nDCG@5 | Context keyword recall | p50 latency |
|---|---|---|---|---|---|
| dense only | 0.773 | 0.934 | 0.775 | 0.723 | 242 ms ± 4 |
| BM25 only | 0.653 | 0.789 | 0.623 | 0.691 | 1 ms |
| hybrid, broken fusion (original) | 0.650 | 0.581 | 0.525 | 0.647 | 256 ms ± 3 |
| hybrid, RRF (fixed) | 0.787 | 0.930 | 0.779 | 0.732 | 259 ms ± 17 |
| hybrid + rerank | 0.787 | 0.863 | 0.754 | 0.728 | 9263 ms ± 379 |

Rows with the graph and images are pending (Neo4j, image captions); answer-level
columns (keyword recall on the answer, faithfulness) are pending (Groq key).

**Ground truth is provisional.** Recall@5, MRR and nDCG@5 use the automatic
relevance suggestions in `data/golden_qa/relevance_candidates.json`, not yet
reviewed by a person. Candidates were ranked by similarity between each
question's ideal answer and each chunk using `all-MiniLM-L6-v2`, a model no
retriever uses; identical slides form one group. Because that ranking is
semantic, it may favour dense retrieval over BM25.

What the table says:

* **The original hybrid was worse than dense retrieval alone** on every
  metric: MRR 0.581 against 0.934. The broken fusion did not just fail to help;
  concatenating two independently scored lists pushed the best dense hit down.
* **The fix:** Recall@5 0.650 to 0.787, MRR 0.581 to 0.930, nDCG@5 0.525 to
  0.779, context keyword recall 0.647 to 0.732.
* **Fixed hybrid is close to dense alone** on this corpus (Recall@5 +0.014,
  nDCG@5 +0.004, MRR -0.004). BM25 adds little here; slides are short and the
  dense model already matches their vocabulary.
* **The reranker does not pay for itself here.** `BAAI/bge-reranker-base` over
  30 candidates leaves Recall@5 unchanged, lowers MRR by 0.067 and costs about
  36x the latency on CPU. It stays off by default and remains in the ablation.

A first ablation run reported 13 to 16 ms medians for runs 2 and 3. That was a
measurement bug: the embedder's in-memory query cache was still on, so those
runs skipped query embedding. The cache is now disabled in the ablation and the
table above comes from the corrected run.

### Router (`graphrag route-eval`, `results/router.json`)

Zero-shot nearest-template routing agrees with the golden question type on
0.58 of questions (lookup 16/31, relational 11/15, multi-hop 2/4). The
templates were written after seeing the golden questions, so this is
indicative, not held-out. A misrouted lookup only costs graph latency; a
misrouted multi-hop question loses decomposition. Not tuned further on these
labels to avoid overfitting; an LLM router is worth comparing once a key is
available.

### Not yet measured

Answer-level keyword recall and the before/after against the notebook's 0.588,
fabricated-citation rate, calibration plot, supported-sentence rate, LLM-judge
faithfulness and its agreement with human labels, multi-hop recall with the
router and decomposition, HyDE, the graph and image ablation rows, and
`docker compose up`.

### Image captions (`graphrag index --skip-text`)

With Tesseract 5.4 installed: 14 images survive dedup and the caption filter
(10 OCR, 4 BLIP), matching the notebook's 14 image chunks.

* An earlier run without Tesseract sent all 14 to BLIP. Two of those captions
  were degenerate repetition ("gender gender gender ...") and text-heavy tables
  got vague one-liners. With OCR, those images now carry their real text, for
  example the antisocial-behaviour label table and the crisis-tweet taxonomy.
* The caption filter now also rejects repetitive captions (six or more words,
  fewer than half distinct). On this corpus it no longer fires, because OCR
  handles the images that produced the repetition.
* Still wrong: BLIP describes the university logo as other universities
  (2 captions). OCR text is noisy on diagrams. Neither is filtered.
