"""Streamlit demo for the graph-rag-multimodal API.

Run the API first (``uvicorn graphrag.api.app:app_factory --factory``), then:

    streamlit run demo/app.py

Shows the answer with per-sentence support, the retrieved chunks, matched
images, and an interactive render of the graph paths that were in context.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st
import streamlit.components.v1 as components

from graphrag.graph.render import subgraph_html

API_URL = os.environ.get("GRAPHRAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Graph RAG demo", layout="wide")
st.title("Graph-augmented multimodal RAG")

question = st.text_input("Ask a question about the corpus", placeholder="What are the three types of user credibility?")
if not question:
    st.stop()

with st.spinner("Retrieving and answering..."):
    try:
        resp = httpx.post(f"{API_URL}/ask", json={"question": question, "stream": False}, timeout=120)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        st.error(f"API error: {exc}")
        st.stop()
data = resp.json()
answer, retrieval = data["answer"], data["retrieval"]

left, right = st.columns([3, 2])
with left:
    st.subheader("Answer")
    if answer["sentences"]:
        for s in answer["sentences"]:
            badge = "✅" if s["supported"] else "⚠️"
            cites = ", ".join(s["valid_ids"]) or "no valid citation"
            st.markdown(f"{badge} {s['text']}  \n<small>{cites} · entailment {s['entailment']:.2f}</small>", unsafe_allow_html=True)
    else:
        st.write(answer["answer"])
    st.caption(
        f"Confidence (retrieval evidence): **{answer['confidence']}** · "
        f"model self-report: {answer['confidence_self_reported']} · "
        f"fabricated citations dropped: {answer['citations'].get('fabricated', 0)}"
    )
    if answer["unsupported_claims"]:
        with st.expander(f"Unsupported claims ({len(answer['unsupported_claims'])})"):
            for claim in answer["unsupported_claims"]:
                st.write("- " + claim)
    if answer["conflicts"]:
        with st.expander("Possible conflicts between sources"):
            for c in answer["conflicts"]:
                st.write("- " + c)

    st.subheader("Retrieved chunks")
    for c in retrieval["chunks"]:
        with st.expander(f"{c['chunk_id']} · {c['doc_id']} p.{c['page']} · via {', '.join(c['retrieved_by'])}"):
            st.write(c["text"])

with right:
    st.subheader("Graph paths in context")
    if retrieval["graph_paths"]:
        components.html(subgraph_html(retrieval["graph_paths"], retrieval["entities"]), height=440)
    else:
        st.info("No graph paths (graph disabled, or the router chose a lookup route).")
    st.subheader("Matched images")
    for img in retrieval["images"]:
        st.markdown(f"**{img['image_id']}** ({img['doc_id']} p.{img['page']}): {img['caption']}")
    st.caption(f"Route: {retrieval['route'] or 'none'} · timings (ms): {data['timings_ms']}")
