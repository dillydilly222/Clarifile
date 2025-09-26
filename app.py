import os
from pathlib import Path
import streamlit as st

from ingest import ingest_pdfs
from retriever import answer

st.set_page_config(page_title="Clarifile Demo")

# Paths
ROOT = Path(__file__).parent.resolve()
DEMO_DIR = ROOT / "data" / "Demo"

# Preload all demo PDFs
def preload_demo_pdfs():
    pdfs = [str(p) for p in DEMO_DIR.glob("*.pdf")]
    if not pdfs:
        return []
    try:
        count = ingest_pdfs(pdfs)
        return pdfs
    except Exception as e:
        st.error(f"Error ingesting demo PDFs: {e}")
        return []

# Preload knowledge base at app start
if "demo_files" not in st.session_state:
    st.session_state.demo_files = preload_demo_pdfs()

st.title("Clarifile Demo")

# Show summary of loaded files
if st.session_state.demo_files:
    st.subheader("Knowledge Base Summary")
    st.write("The following demo PDFs are loaded into the knowledge base:")
    for f in st.session_state.demo_files:
        st.markdown(f"- `{Path(f).name}`")
else:
    st.warning("No demo PDFs found in `data/Demo/`.")

st.markdown("---")

# Question box
st.subheader("Ask a Question")
q = st.text_area("Question", height=120, placeholder="Type your question about the ingested docs…")

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
