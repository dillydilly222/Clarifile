import os
from pathlib import Path
import streamlit as st

from retriever import answer

st.set_page_config(page_title="Clarifile Demo")

ROOT = Path(__file__).parent.resolve()
DEMO_DIR = ROOT / "data" / "Demo"
URLS_TXT = ROOT / "data" / "Demo" / "urls.txt"
STORAGE_DIR = ROOT / "storage"

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

st.title("Clarifile Demo")

st.subheader("Knowledge Base Summary")
files = [Path(p).name for p in list_demo_pdfs()]
urls = read_urls_txt()
storage_exists = STORAGE_DIR.exists() and STORAGE_DIR.is_dir()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Demo PDFs detected", len(files))
with col2:
    st.metric("URLs detected in urls.txt", len(urls))
with col3:
    st.metric("Storage folder present", int(storage_exists))

if files:
    with st.expander("View PDF files"):
        for name in files:
            st.markdown(f"- `{name}`")
else:
    st.info("No PDFs found in 'data/Demo/'.")

if urls:
    with st.expander("View URLs"):
        for u in urls:
            st.markdown(f"- {u}")
else:
    st.info("No URLs found in 'urls.txt' (or file missing).")

if not storage_exists:
    st.error("Missing 'storage/' directory. Ensure the Chroma collection is prebuilt in ./storage before asking questions.")

st.markdown("---")

st.subheader("Ask a Question")
q = st.text_area("Question", height=120, placeholder="Ask about the ingested PDFs/URLs…")

if st.button("Ask"):
    if not q.strip():
        st.warning("Please enter a question.")
    elif not storage_exists:
        st.error("No vector store found. Build the knowledge base into ./storage first.")
    else:
        with st.spinner("Retrieving and answering..."):
            try:
                res = answer(q)
                a = res.get("answer") if isinstance(res, dict) else str(res)
            except Exception as e:
                a = f"Error: {e}"
        st.text_area("Answer", value=a, height=360)
st.image("Flow Diagram.png", caption="Clarifile Flow Diagram", use_column_width=True)
