import os
from pathlib import Path
import streamlit as st

from ingest import ingest_pdfs, ingest_urls
from retriever import answer

st.set_page_config(page_title="Clarifile Demo")

ROOT = Path(__file__).parent.resolve()
DEMO_DIR = ROOT / "data" / "Demo"
URLS_TXT = ROOT / "urls.txt"

def list_demo_pdfs() -> list[str]:
    return [str(p) for p in sorted(DEMO_DIR.glob("*.pdf"))]

def read_urls_txt() -> list[str]:
    if not URLS_TXT.exists():
        return []
    urls = []
    for line in URLS_TXT.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        urls.append(s)
    seen, deduped = set(), []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped

def build_kb_if_needed():
    if st.session_state.get("kb_ready"):
        return
    ingested_files, ingested_urls, errors = [], [], []
    try:
        pdfs = list_demo_pdfs()
        if pdfs:
            ingest_pdfs(pdfs)
            ingested_files = [Path(p).name for p in pdfs]
    except Exception as e:
        errors.append(f"PDF ingest error: {e}")
    try:
        urls = read_urls_txt()
        if urls:
            ingest_urls(urls)
            ingested_urls = urls
    except Exception as e:
        errors.append(f"URL ingest error: {e}")
    st.session_state.kb_ready = True
    st.session_state.ingested_files = ingested_files
    st.session_state.ingested_urls = ingested_urls
    st.session_state.kb_errors = errors

if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "ingested_urls" not in st.session_state:
    st.session_state.ingested_urls = []
if "kb_errors" not in st.session_state:
    st.session_state.kb_errors = []

build_kb_if_needed()

st.title("Clarifile Demo")

st.subheader("Knowledge Base Summary")
files = st.session_state.ingested_files
urls = st.session_state.ingested_urls
errs = st.session_state.kb_errors

left, right = st.columns(2)
with left:
    st.metric("Demo PDFs loaded", len(files))
with right:
    st.metric("URLs loaded from urls.txt", len(urls))

if files:
    with st.expander("View PDF files"):
        for name in files:
            st.markdown(f"- `{name}`")
else:
    st.info("No PDFs found in `data/Demo/`.")

if urls:
    with st.expander("View URLs"):
        for u in urls:
            st.markdown(f"- {u}")
else:
    st.info("No URLs found in `urls.txt` (or file missing).")

if errs:
    with st.expander("Ingestion Warnings / Errors"):
        for e in errs:
            st.warning(e)

st.markdown("---")

st.subheader("Ask a Question")
q = st.text_area("Question", height=120, placeholder="Ask about the ingested PDFs/URLs…")

if st.button("Ask"):
    if not q.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and answering..."):
            try:
                res = answer(q)
                a = res.get("answer") if isinstance(res, dict) else str(res)
            except Exception as e:
                a = f"Error: {e}"
        st.text_area("Answer", value=a, height=360)
