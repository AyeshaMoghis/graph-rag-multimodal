from __future__ import annotations

from graphrag.graph.render import path_edges
from graphrag.graph.retrieve import render_path


def test_path_edges_round_trips_render_path() -> None:
    path = render_path(["A", "B", "C"], ["USES", "IS_A"], ["->", "<-"])
    assert path_edges(path) == [("A", "USES", "B"), ("C", "IS_A", "B")]


def test_path_edges_handles_multi_word_names() -> None:
    assert path_edges("User Credibility -[PART_OF]-> Social Media Graph") == [
        ("User Credibility", "PART_OF", "Social Media Graph")
    ]
