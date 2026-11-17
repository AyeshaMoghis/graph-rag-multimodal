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
