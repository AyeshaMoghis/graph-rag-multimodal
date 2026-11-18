"""Post-hoc checks on generated answers.

* :func:`validate_citations` drops any citation that does not point at something
  that was actually in the prompt, and counts it as fabricated.
* :func:`evidence_confidence` scores confidence from retrieval evidence instead
  of trusting the model's self-report.
* :func:`detect_conflicts` flags contradictions between retrieved chunks with
  an NLI model, replacing the notebook's substring check.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from graphrag.reasoning.prompts import graph_path_id
from graphrag.retrieval.fusion import PackedContext

CITATION_FIELDS = ("source_chunks_used", "graph_paths_used", "relevant_images")
MIN_QUOTE_CHARS = 12

ConfidenceLevel = Literal["high", "medium", "low"]


def _norm(text: object) -> str:
    return " ".join(str(text).lower().split())


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


# ── Citations ───────────────────────────────────────────────────────────────


@dataclass
class CitationReport:
    kept: dict[str, list[str]] = field(default_factory=dict)
    dropped: dict[str, list[str]] = field(default_factory=dict)

    @property
    def cited(self) -> int:
        return sum(len(v) for v in self.kept.values()) + self.fabricated

    @property
    def fabricated(self) -> int:
        return sum(len(v) for v in self.dropped.values())

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cited": self.cited, "fabricated": self.fabricated}


def _is_present(entry: str, ids: set[str], texts: Sequence[str]) -> bool:
    """An entry is present if it is a shown id, or a verbatim quote of shown text."""
    bare = entry.strip().strip("[]").strip()
    if bare in ids:
        return True
    quote = _norm(bare)
    return len(quote) >= MIN_QUOTE_CHARS and any(quote in t for t in texts)


def validate_citations(answer: dict[str, Any], context: PackedContext) -> CitationReport:
    """Split each citation field into entries present in ``context`` and fabricated ones."""
    sources: dict[str, tuple[set[str], list[str]]] = {
        "source_chunks_used": (
            set(context.chunk_ids),
            [_norm(getattr(c, "text", c)) for c in context.text_chunks],
        ),
        "graph_paths_used": (
            {graph_path_id(i) for i in range(len(context.kept_graph_paths))},
            [_norm(p) for p in context.kept_graph_paths],
        ),
        "relevant_images": (
            set(context.image_ids),
            [
                _norm(f"{getattr(i, 'caption', '')} {getattr(i, 'image_path', '')}")
                for i in context.images
            ],
        ),
    }
    report = CitationReport()
    for name in CITATION_FIELDS:
        ids, texts = sources[name]
        entries = _as_list(answer.get(name))
        report.kept[name] = [e for e in entries if _is_present(e, ids, texts)]
        report.dropped[name] = [e for e in entries if not _is_present(e, ids, texts)]
    return report


# ── Evidence-based confidence ───────────────────────────────────────────────

HIGH_THRESHOLD = 0.60
MEDIUM_THRESHOLD = 0.35


@dataclass
class EvidenceConfidence:
    score: float
    level: ConfidenceLevel
    components: dict[str, float]


def evidence_confidence(
    text_hits: Sequence[Any],
    graph_paths_in_context: int | None,
) -> EvidenceConfidence:
    """Average of up to three components, each in [0, 1].

    * ``agreement``: share of retrieved chunks returned by both dense and BM25.
    * ``margin``: relative score drop from the first to the last retrieved
      chunk, doubled and capped at 1. A clear winner scores high.
    * ``graph``: 1 if at least one graph path made it into the prompt. Omitted
      when the graph layer is disabled (``graph_paths_in_context is None``).

    The weights and the thresholds (0.60 high, 0.35 medium) were fixed before
    any evaluation; the calibration report checks whether they track recall.
    """
    components: dict[str, float] = {}
    if text_hits:
        components["agreement"] = sum(
            1.0 for h in text_hits if len(getattr(h, "retrieved_by", ())) == 2
        ) / len(text_hits)
        top, last = float(text_hits[0].score), float(text_hits[-1].score)
        components["margin"] = min(1.0, 2 * (top - last) / top) if top > 0 else 0.0
    else:
        components["agreement"] = components["margin"] = 0.0
    if graph_paths_in_context is not None:
        components["graph"] = 1.0 if graph_paths_in_context > 0 else 0.0

    score = round(sum(components.values()) / len(components), 4)
    level: ConfidenceLevel = (
        "high" if score >= HIGH_THRESHOLD else "medium" if score >= MEDIUM_THRESHOLD else "low"
    )
    return EvidenceConfidence(score, level, components)


# ── Contradiction detection (Task 2.3) ──────────────────────────────────────
# The notebook flagged a conflict when a chunk contained the literal string
# "not <node>" or "unlike <node>". That fires on ordinary contrastive phrasing
# and misses real contradictions. Here, claims (sentences or bullets) from
# different chunks that mention the same entity are scored by an NLI
# cross-encoder, and pairs classified as contradiction above a threshold are
# reported. Restricting to shared-entity pairs keeps the number of model calls
# small.

_BULLET_LINE = re.compile(r"^\s*[•*\-–→▪◦❑○■●]+\s*")  # noqa: RUF001 - real bullet glyphs
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_claims(text: str, min_words: int = 6, max_words: int = 60) -> list[str]:
    """Split slide or page text into claims.

    PDF text wraps lines mid-sentence, so lines are re-joined first; a new claim
    starts only at a bullet glyph or a sentence end. Fragments shorter than
    ``min_words`` (slide titles, headers) are dropped.
    """
    blocks: list[str] = []
    for line in text.splitlines():
        if _BULLET_LINE.match(line) or not blocks:
            blocks.append(_BULLET_LINE.sub("", line))
        else:
            blocks[-1] += " " + line
    claims = []
    for block in blocks:
        for part in _SENTENCE_END.split(" ".join(block.split())):
            if min_words <= len(part.split()) <= max_words:
                claims.append(part.strip())
    return claims


class NLIScorer(Protocol):
    def contradiction_probs(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


class EntailmentScorer(Protocol):
    def entailment_probs(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


class NLIModel(NLIScorer, EntailmentScorer, Protocol):
    """An NLI model that can score both contradiction and entailment."""


class CrossEncoderNLI:
    """``cross-encoder/nli-deberta-v3-small``; the label order is read from the model config.

    Pairs are (premise, hypothesis).
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._labels: dict[str, int] = {}

    def _probs(self, pairs: Sequence[tuple[str, str]], label: str) -> list[float]:
        if not pairs:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self._labels = {v.lower(): int(k) for k, v in self._model.config.id2label.items()}
        idx = next(i for name, i in self._labels.items() if name.startswith(label))
        probs = self._model.predict(list(pairs), apply_softmax=True, show_progress_bar=False)
        return [float(row[idx]) for row in probs]

    def contradiction_probs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return self._probs(pairs, "contradiction")

    def entailment_probs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return self._probs(pairs, "entailment")


@dataclass
class Conflict:
    entity: str
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    probability: float

    def describe(self) -> str:
        return (
            f"Possible contradiction about '{self.entity}' between {self.source_a} "
            f'("{self.claim_a}") and {self.source_b} ("{self.claim_b}").'
        )


def candidate_pairs(
    chunks: Sequence[Any], entities: Sequence[str], max_pairs: int = 64
) -> list[tuple[str, tuple[str, str], tuple[str, str]]]:
    """(entity, (source, claim), (source, claim)) for claims from different chunks sharing an entity."""
    claims = [
        (str(getattr(c, "chunk_id", i)), claim)
        for i, c in enumerate(chunks)
        for claim in split_claims(str(getattr(c, "text", c)))
    ]
    out: list[tuple[str, tuple[str, str], tuple[str, str]]] = []
    seen: set[tuple[str, str]] = set()
    for entity in sorted({e.lower() for e in entities if len(e) > 2}):
        mentioning = [(src, cl) for src, cl in claims if entity in cl.lower()]
        for i, a in enumerate(mentioning):
            for b in mentioning[i + 1 :]:
                key = (min(a[1], b[1]), max(a[1], b[1]))
                if a[0] == b[0] or a[1] == b[1] or key in seen:
                    continue
                seen.add(key)
                out.append((entity, a, b))
                if len(out) >= max_pairs:
                    return out
    return out


def relatedness_filter(
    pairs: list[tuple[str, tuple[str, str], tuple[str, str]]],
    encoder: Any,
    min_similarity: float,
) -> list[tuple[str, tuple[str, str], tuple[str, str]]]:
    """Keep pairs whose claims are about the same thing (cosine >= ``min_similarity``).

    A contradiction is only meaningful between claims about the same
    proposition; NLI scores for unrelated claims are noise.
    """
    if not pairs or encoder is None or min_similarity <= 0:
        return pairs
    texts = sorted({c for _, a, b in pairs for c in (a[1], b[1])})
    index = {t: i for i, t in enumerate(texts)}
    vecs = encoder.encode(texts)  # unit-normalised, so the dot product is cosine
    return [
        p for p in pairs if float(vecs[index[p[1][1]]] @ vecs[index[p[2][1]]]) >= min_similarity
    ]


def detect_conflicts(
    chunks: Sequence[Any],
    entities: Sequence[str],
    nli: NLIScorer,
    threshold: float = 0.8,
    max_pairs: int = 64,
    encoder: Any = None,
    min_similarity: float = 0.0,
) -> list[Conflict]:
    pairs = relatedness_filter(
        candidate_pairs(chunks, entities, max_pairs), encoder, min_similarity
    )
    # Score both directions and keep the minimum: a real contradiction holds either
    # way round, while small NLI models often label unrelated statements
    # "contradiction" in one direction only.
    forward = nli.contradiction_probs([(a[1], b[1]) for _, a, b in pairs])
    backward = nli.contradiction_probs([(b[1], a[1]) for _, a, b in pairs])
    probs = [min(f, r) for f, r in zip(forward, backward, strict=True)]
    found = [
        Conflict(entity, a[1], a[0], b[1], b[0], round(p, 4))
        for (entity, a, b), p in zip(pairs, probs, strict=True)
        if p >= threshold
    ]
    return sorted(found, key=lambda c: -c.probability)


# ── Sentence-level citations (Task 3.5) ─────────────────────────────────────
# With prompt v3 every answer sentence ends with the ids that support it. Each
# sentence is checked: its ids must exist in the prompt, and the NLI model must
# judge that at least one cited source entails the sentence. Everything else is
# listed as an unsupported claim.

_CITATION = re.compile(r"\[([^\[\]]+)\]")
_ANSWER_SENTENCE_END = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z0-9\"'(])")


@dataclass
class SentenceSupport:
    text: str
    cited: list[str]
    valid_ids: list[str]
    fabricated_ids: list[str]
    entailment: float
    supported: bool
    sources: list[dict[str, Any]] = field(default_factory=list)


def split_answer(answer: str) -> list[tuple[str, list[str]]]:
    """(sentence without citation markers, cited ids) for each answer sentence."""
    out = []
    for raw in _ANSWER_SENTENCE_END.split(answer.strip()):
        ids = [
            part.strip()
            for group in _CITATION.findall(raw)
            for part in re.split(r"[,;]", group)
            if part.strip()
        ]
        text = " ".join(_CITATION.sub("", raw).split()).replace(" .", ".")
        if text:
            out.append((text, ids))
    return out


def verify_sentences(
    answer: str,
    context: PackedContext,
    nli: EntailmentScorer,
    threshold: float = 0.5,
) -> list[SentenceSupport]:
    """Check each sentence's citations against the prompt and with NLI entailment."""
    premises: dict[str, tuple[str, dict[str, Any]]] = {}
    for c in context.text_chunks:
        cid = str(getattr(c, "chunk_id", ""))
        meta = {"id": cid, "doc_id": getattr(c, "doc_id", ""), "page": getattr(c, "page", 0)}
        premises[cid] = (str(getattr(c, "text", "")), meta)
    for i, path in enumerate(context.kept_graph_paths):
        premises[graph_path_id(i)] = (path, {"id": graph_path_id(i), "graph_path": path})
    for img in context.images:
        iid = str(getattr(img, "image_id", ""))
        premises[iid] = (
            str(getattr(img, "caption", "")),
            {"id": iid, "doc_id": getattr(img, "doc_id", ""), "page": getattr(img, "page", 0)},
        )

    sentences = split_answer(answer)
    pairs: list[tuple[str, str]] = []
    owners: list[int] = []
    for i, (text, ids) in enumerate(sentences):
        for cid in ids:
            if cid in premises:
                pairs.append((premises[cid][0], text))
                owners.append(i)
    probs = nli.entailment_probs(pairs)
    best = [0.0] * len(sentences)
    for i, p in zip(owners, probs, strict=True):
        best[i] = max(best[i], p)

    out = []
    for i, (text, ids) in enumerate(sentences):
        valid = [cid for cid in ids if cid in premises]
        out.append(
            SentenceSupport(
                text=text,
                cited=ids,
                valid_ids=valid,
                fabricated_ids=[cid for cid in ids if cid not in premises],
                entailment=round(best[i], 4),
                supported=bool(valid) and best[i] >= threshold,
                sources=[premises[cid][1] for cid in valid],
            )
        )
    return out
