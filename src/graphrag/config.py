"""Single source of configuration, loaded from the environment and `.env`.

Credentials use their conventional names (GROQ_API_KEY, NEO4J_URI, ...).
Every other setting can be overridden with a ``GRAPHRAG_`` prefixed variable,
e.g. ``GRAPHRAG_FUSION=weighted``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

FusionMode = Literal["rrf", "weighted", "legacy", "dense", "sparse"]
TokenizerMode = Literal["normalized", "legacy"]
GraphModeSetting = Literal["ranked", "legacy"]
EntityExtractorMode = Literal["ner", "legacy"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GRAPHRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Credentials (never given defaults) ───────────────────────────────
    groq_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_API_KEY")
    )
    neo4j_uri: str | None = Field(default=None, validation_alias=AliasChoices("NEO4J_URI"))
    neo4j_user: str | None = Field(default=None, validation_alias=AliasChoices("NEO4J_USER"))
    neo4j_password: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("NEO4J_PASSWORD")
    )
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY")
    )

    # ── Paths ────────────────────────────────────────────────────────────
    corpus_dir: Path = Path("data/sample")
    output_dir: Path = Path("output")
    golden_qa_path: Path = Path("data/golden_qa/golden_qa_set_complete.json")

    # ── Models ───────────────────────────────────────────────────────────
    embed_model: str = "BAAI/bge-large-en-v1.5"
    embed_batch_size: int = 64
    caption_model: str = "Salesforce/blip-image-captioning-base"
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    prompt_version: Literal["v1", "v2"] = "v2"
    conflict_mode: Literal["nli", "naive"] = "nli"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_threshold: float = 0.8
    nli_max_pairs: int = 64
    # Chosen on the synthetic planted set: the highest gate that kept recall at the
    # ungated level (see results/conflicts.json and docs/EVALUATION_NOTES.md).
    nli_min_similarity: float = 0.6

    # ── Ingest ───────────────────────────────────────────────────────────
    phash_threshold: int = 10
    chunk_min_words: int = 40
    chunk_dense_line_count: int = 6
    chunk_min_chunk_words: int = 25
    ocr_min_alpha_words: int = 15
    caption_min_words: int = 4

    # ── Retrieval ────────────────────────────────────────────────────────
    fusion: FusionMode = "rrf"
    bm25_tokenizer: TokenizerMode = "normalized"
    rrf_k: int = 60
    hybrid_alpha: float = 0.5
    candidate_multiplier: int = 3
    rerank: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_candidates: int = 30
    router: bool = False
    decompose: bool = False
    hyde: bool = False
    text_top_k: int = 5
    image_top_k: int = 3
    graph_hops: int = 2
    graph_top_n_entities: int = 5
    graph_fuzzy_threshold: int = 70
    graph_path_limit: int = 30
    graph_mode: GraphModeSetting = "ranked"
    entity_extractor: EntityExtractorMode = "ner"
    ner_model: str = "urchade/gliner_small-v2.1"
    ner_threshold: float = 0.4
    link_lexical_threshold: int = 90
    link_embedding_threshold: float = 0.80

    # ── Context packing ──────────────────────────────────────────────────
    context_budget_tokens: int = 6000
    graph_budget_fraction: float = 0.40
    text_budget_fraction: float = 0.40
    image_budget_fraction: float = 0.20

    # ── Knowledge graph extraction ───────────────────────────────────────
    triple_confidence_threshold: float = 0.65
    triple_text_char_limit: int = 2000
    extraction_sleep_seconds: float = 1.0
    extraction_checkpoint_every: int = 50

    # ── Eval ─────────────────────────────────────────────────────────────
    eval_sleep_seconds: float = 1.5
    seed: int = 42

    # ── Derived paths ────────────────────────────────────────────────────
    @property
    def documents_dir(self) -> Path:
        return self.output_dir / "documents"

    @property
    def chroma_path(self) -> Path:
        return self.output_dir / "chroma_db"

    @property
    def embeddings_dir(self) -> Path:
        return self.output_dir / "embeddings"

    @property
    def bm25_path(self) -> Path:
        return self.output_dir / "bm25_index.json"

    @property
    def captions_path(self) -> Path:
        return self.output_dir / "image_captions.json"

    @property
    def graph_checkpoint_path(self) -> Path:
        return self.output_dir / "graph_meta.json"

    @property
    def graph_enabled(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_user and self.neo4j_password)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
