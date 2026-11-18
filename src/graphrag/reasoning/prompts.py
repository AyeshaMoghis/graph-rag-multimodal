"""Prompt templates, verbatim from the original notebook.

Placeholders are replaced with ``str.replace`` rather than ``str.format``
because the templates contain literal JSON braces.
"""

TRIPLE_PROMPT = """You are a knowledge graph extraction assistant.
Given a text passage, extract all factual relationships as triples.

STRICT RULES:
- Only use these node types: Concept, Algorithm, Dataset, Architecture
- Only use these relation types: USES, IS_A, CONTRASTS_WITH, PART_OF, DESCRIBED_IN
- Assign a confidence score 0.0 to 1.0 to each triple
- Return ONLY a valid JSON array — no markdown fences, no explanation, no backticks
- If no valid triples found, return empty array: []

Output format:
[{"subject": "...", "subject_type": "...", "relation": "...", "object": "...", "object_type": "...", "confidence": 0.0}]

TEXT:
REPLACE_TEXT_HERE
"""

REASONING_PROMPT = """You are an expert assistant. Use the provided multi-source context to answer the question.

GRAPH RELATIONSHIPS (structured knowledge):
{graph_context}

TEXT EXCERPTS (detailed content):
{text_chunks}

RELEVANT DIAGRAMS / IMAGES (described):
{image_captions}

{conflict_note}

INSTRUCTIONS:
- Use graph relationships to structure your answer logically.
- Use text excerpts to explain concepts in detail.
- Reference diagrams where they add visual clarity.
- If sources conflict, state the conflict explicitly.
- Answer step-by-step where appropriate.
- Be concise and grounded — only use what is in the provided context.

Respond ONLY with a JSON object in this exact format (no markdown fences):
{"answer": "...", "referenced_concepts": ["..."], "source_chunks_used": ["..."], "graph_paths_used": ["..."], "relevant_images": ["..."], "confidence": "high|medium|low"}

QUESTION: REPLACE_QUERY_HERE
"""


def render_reasoning_prompt(
    query: str,
    graph_context: str,
    text_chunks: list[str],
    image_captions: list[str],
    conflicts: list[str],
) -> str:
    conflict_note = ("POTENTIAL CONFLICTS DETECTED:\n" + "\n".join(conflicts)) if conflicts else ""
    prompt = REASONING_PROMPT.replace("{graph_context}", graph_context)
    prompt = prompt.replace("{text_chunks}", "\n---\n".join(text_chunks) or "(none)")
    prompt = prompt.replace("{image_captions}", "\n".join(image_captions) or "(none)")
    prompt = prompt.replace("{conflict_note}", conflict_note)
    return prompt.replace("REPLACE_QUERY_HERE", query)


# ── v2: every piece of context carries an id the model can cite ─────────────
# v1 asked for source_chunks_used without ever showing chunk ids, so any chunk
# citation it produced was invented. v2 labels each item and restricts
# citations to those labels, which makes citations checkable.

REASONING_PROMPT_V2 = """You are an expert assistant. Answer the question using ONLY the context below.

GRAPH RELATIONSHIPS (structured knowledge):
{graph_context}

TEXT EXCERPTS (detailed content):
{text_chunks}

RELEVANT DIAGRAMS / IMAGES (described):
{image_captions}

{conflict_note}

INSTRUCTIONS:
- Use graph relationships to structure your answer and text excerpts for detail.
- If sources conflict, state the conflict explicitly.
- Be concise and grounded: do not add facts that are not in the context.
- Cite only ids that appear in square brackets above. Never invent an id.
  If you did not use any graph relationship, return an empty graph_paths_used list.

Respond ONLY with a JSON object in this exact format (no markdown fences):
{"answer": "...", "referenced_concepts": ["..."], "source_chunks_used": ["<text id>"], "graph_paths_used": ["<G id>"], "relevant_images": ["<image id>"], "confidence": "high|medium|low"}

QUESTION: REPLACE_QUERY_HERE
"""


def graph_path_id(i: int) -> str:
    return f"G{i + 1}"


def render_reasoning_prompt_v2(
    query: str,
    graph_paths: list[str],
    text_items: list[tuple[str, str, int, str]],
    image_items: list[tuple[str, str]],
    conflicts: list[str],
    template: str | None = None,
) -> str:
    """``text_items`` are (chunk_id, doc_id, page, text); ``image_items`` are (image_id, caption)."""
    graph = (
        "\n".join(f"[{graph_path_id(i)}] {p}" for i, p in enumerate(graph_paths))
        or "(no graph paths found)"
    )
    texts = (
        "\n---\n".join(
            f"[{cid}] ({doc}, page {page})\n{text}" for cid, doc, page, text in text_items
        )
        or "(none)"
    )
    images = "\n".join(f"[{iid}] {cap}" for iid, cap in image_items) or "(none)"
    conflict_note = ("POTENTIAL CONFLICTS DETECTED:\n" + "\n".join(conflicts)) if conflicts else ""
    prompt = (template or REASONING_PROMPT_V2).replace("{graph_context}", graph)
    prompt = prompt.replace("{text_chunks}", texts)
    prompt = prompt.replace("{image_captions}", images)
    prompt = prompt.replace("{conflict_note}", conflict_note)
    return prompt.replace("REPLACE_QUERY_HERE", query)


# v3: v2 plus a citation at the end of every answer sentence (Task 3.5).
REASONING_PROMPT_V3 = REASONING_PROMPT_V2.replace(
    "- Cite only ids that appear in square brackets above.",
    "- End EVERY sentence of the answer with the ids that support it, in square\n"
    '  brackets, e.g. "Credibility is a node property [SC Unit 2-1.pdf_p013_c00]."\n'
    "- Cite only ids that appear in square brackets above.",
)
