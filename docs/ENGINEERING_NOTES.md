# Engineering notes: the hybrid-fusion bug

## What it was

The notebook's hybrid retriever blended a dense score (BGE embeddings in
ChromaDB) with a BM25 score: `alpha * dense + (1 - alpha) * bm25`. To do that it
put each retriever's candidates in a dictionary keyed by chunk id, then took the
union of the two dictionaries.

The two sides used different keys:

```python
# dense side
cid = meta.get("chunk_id", doc[:30])     # chunk_id was never written to metadata
# BM25 side
cid = c.get("chunk_id", f"bm_{idx}")     # real chunk id
```

When the Chroma collection was built, the metadata dict held `doc_id`, `page`,
`page_type`, `source` and `word_count`, but not `chunk_id`. The chunk id was
only the Chroma record id. So `meta.get("chunk_id", ...)` always fell back to
the first 30 characters of the chunk text. A dense key looked like
`"Cyberbullying is the use of di"` and a BM25 key looked like
`"lecture3.pdf_p012_c00"`. They could never be equal.

Consequences:

1. **No chunk ever received both signals.** Every candidate was scored as
   `alpha * dense + 0` or `0 + (1 - alpha) * bm25`. The "hybrid" retriever was
   two independent rankers concatenated and re-sorted.
2. **Duplicate slots.** A chunk found by both retrievers entered the merged list
   twice, once under each key, and could take two of the five context slots.
3. **Prefix collisions.** Two chunks whose text started with the same 30
   characters, such as repeated slide headers, collapsed into one dense entry.
4. **Scale mismatch decided the ranking.** Dense scores were raw similarities
   on a 0 to 1 scale. BM25 scores were divided by the top BM25 score, so the best
   BM25 hit always scored exactly `1 - alpha = 0.5`, however well it matched.

A second, independent bug sat in the BM25 tokenization. The corpus was indexed
with `text.split()`, keeping case and punctuation, while queries were tokenized
with `query.lower().split()`. A capitalized document token such as `MBTI` could
never match the lower-cased query token `mbti`. A token with trailing
punctuation such as `networks,` never matched `networks` either.

## How it was found

The key mismatch was spotted by reading the notebook's `hybrid_retrieve` next
to the cell that built the Chroma collection: the metadata written there has
no `chunk_id`, so the dense-side lookup could only ever hit its fallback. It
was then confirmed with a faithful reproduction of the old logic
(`fusion=legacy`), pinned by `test_legacy_mode_reproduces_the_bug`.

The tokenization bug was found by code review while porting BM25 into its own
module, and is pinned by `test_legacy_tokenizer_cannot_match_capitalised_terms`.

## The fix

* Chunk ids are written to Chroma metadata. Fusion keys on the id Chroma
  returns for each result, which is the chunk id. Only the deliberate `legacy`
  reproduction mode still uses the broken keys.
* The default fusion is now Reciprocal Rank Fusion,
  `score(d) = sum over retrievers of 1 / (60 + rank(d))`. RRF only uses ranks,
  so it needs no score normalization. Min-max normalization over a truncated
  top-N list does not fix the scale mismatch: the normalized value depends on
  which other candidates happened to make the cut.
* Weighted fusion is still available with `GRAPHRAG_FUSION=weighted`, now keyed
  correctly.
* BM25 lower-cases and splits on word characters for both documents and
  queries.
* `GRAPHRAG_FUSION=legacy` with `GRAPHRAG_BM25_TOKENIZER=legacy`, or
  `graphrag eval --preset original`, reproduces the old behavior exactly so the
  before number is measured on the same code path, not quoted from memory.

## What it cost in recall

Not yet measured. This section will be filled from the saved results of
`graphrag eval --preset original` and `graphrag eval --preset fixed` on the same
corpus and golden set.
