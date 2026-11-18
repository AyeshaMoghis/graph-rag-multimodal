"""Turn rendered graph paths back into edges, and edges into an interactive pyvis page."""

from __future__ import annotations

import re
from collections.abc import Sequence

_STEP = re.compile(r"\s(-\[([A-Z_]+)\]->|<-\[([A-Z_]+)\]-)\s")


def path_edges(path: str) -> list[tuple[str, str, str]]:
    """``"A -[USES]-> B <-[IS_A]- C"`` -> ``[("A", "USES", "B"), ("C", "IS_A", "B")]``."""
    parts = _STEP.split(path)
    # split() yields: node, full_step, fwd_rel, back_rel, node, ...
    nodes = parts[0::4]
    edges = []
    for i, (fwd, back) in enumerate(zip(parts[2::4], parts[3::4], strict=True)):
        left, right = nodes[i].strip(), nodes[i + 1].strip()
        edges.append((left, fwd, right) if fwd else (right, back, left))
    return edges


def subgraph_html(
    paths: Sequence[str], highlight: Sequence[str] = (), height: str = "420px"
) -> str:
    """Self-contained HTML for the subgraph used in an answer (needs the ``demo`` extra)."""
    from pyvis.network import Network

    net = Network(height=height, width="100%", directed=True, cdn_resources="in_line")
    seen: set[str] = set()
    marked = {h.lower() for h in highlight}
    for path in paths:
        for src, rel, dst in path_edges(path):
            for node in (src, dst):
                if node not in seen:
                    seen.add(node)
                    net.add_node(
                        node, label=node, color="#e45756" if node.lower() in marked else "#4c72b0"
                    )
            net.add_edge(src, dst, label=rel, arrows="to")
    return str(net.generate_html())
