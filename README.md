# graph-rag-multimodal

Knowledge-graph-augmented multimodal retrieval-augmented generation over PDF
corpora. Text, image captions and a Neo4j concept graph are retrieved together
and packed into a token budget for a Llama-3.1-8B answer.

> Work in progress: this repository is a production rewrite of a research
> notebook. Results tables will be added once every number has been measured.

## Quickstart (development)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                  # fill in GROQ_API_KEY and NEO4J_*

graphrag ingest --corpus-dir data/sample
graphrag index
graphrag build-graph
graphrag ask "What is retrieval-augmented generation?"
graphrag eval --mode retrieval --preset fixed
```

Run the tests with `pytest`. Unit tests make no network calls.
