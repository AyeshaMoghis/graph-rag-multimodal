| Config | Recall@5 | MRR | nDCG@5 | Context keyword recall | Keyword recall (answer) | Faithfulness | p50 latency |
|---|---|---|---|---|---|---|---|
| dense only | 0.773 | 0.934 | 0.775 | 0.723 | pending (LLM) | pending (LLM) | 242 ms ± 4 |
| BM25 only | 0.653 | 0.789 | 0.623 | 0.691 | pending (LLM) | pending (LLM) | 1 ms ± 0 |
| hybrid (broken fusion, original) | 0.650 | 0.581 | 0.525 | 0.647 | pending (LLM) | pending (LLM) | 256 ms ± 3 |
| hybrid (RRF, fixed) | 0.787 | 0.930 | 0.779 | 0.732 | pending (LLM) | pending (LLM) | 259 ms ± 17 |
| hybrid + rerank | 0.787 | 0.863 | 0.754 | 0.728 | pending (LLM) | pending (LLM) | 9263 ms ± 379 |
| hybrid + rerank + graph | pending | | | | | | needs Neo4j (NEO4J_* not set) |
| full (+ images) | pending | | | | | | needs Neo4j (NEO4J_* not set); needs image captions (run `graphrag index` without --skip-images) |

*Retrieval labels are automatic suggestions not yet reviewed by a human; Recall@5, MRR and nDCG@5 are provisional.*

50 questions, 3 runs per row, mean ± std (std omitted when zero).
