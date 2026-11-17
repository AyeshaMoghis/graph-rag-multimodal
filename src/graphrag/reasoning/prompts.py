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
