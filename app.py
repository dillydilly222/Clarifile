import os
from pathlib import Path
import streamlit as st
from retriever import answer
from ingest import ingest_pdfs, ingest_urls

st.set_page_config(page_title="Clarifile Demo")

DEMO_PDFS = [
    "data/Demo/RAGInfo1.pdf",
    "data/Demo/RAGInfo2.pdf",
    "data/Demo/RAGInfo3.pdf",
    "data/Demo/MLInfo1.pdf",
    "data/Demo/AIInfo1.pdf",
]
ONLINE_URLS = [
    "https://atos.net/wp-content/uploads/2024/08/atos-retrieval-augmented-generation-ai-whitepaper.pdf",
    "https://www.tutorialspoint.com/machine_learning/machine_learning_tutorial.pdf",
    "https://www.oajaiml.com/uploads/archivepdf/63501191.pdf",
    "https://medium.com/@Hiadore/building-a-simple-rag-question-answering-system-from-your-own-pdf-for-free-no-framework-used-part-bd60bf370c52",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://jsonplaceholder.typicode.com/todos/1",
    "https://huggingface.co/api/models/bert-base-uncased",
]

if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False

def save_uploads(files):
    dest = Path("data/uploads")
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files or []:
        p = dest / f.name
        with open(p, "wb") as out:
            out.write(f.read())
        paths.append(str(p))
    return paths

st.title("Clarifile Demo")

with st.sidebar:
    st.subheader("Knowledge Base")
    use_demo = st.checkbox("Include bundled demo PDFs", value=True)
    use_urls = st.checkbox("Also fetch and ingest URLs", value=False)
    uploads = st.file_uploader("Add your PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Build / Refresh KB"):
        uploaded_paths = save_uploads(uploads)
        try:
            if use_demo:
                ingest_pdfs(DEMO_PDFS)
            if uploaded_paths:
                ingest_pdfs(uploaded_paths)
            if use_urls:
                ingest_urls(ONLINE_URLS)
            st.session_state.kb_ready = True
            st.success("Knowledge base ready.")
        except Exception as e:
            st.session_state.kb_ready = False
            st.error(f"Build failed: {e}")

st.markdown("### Ask a question")
q = st.text_area("Question", height=120, placeholder="Type your question about the ingested docs…")
if st.button("Ask"):
    if not st.session_state.kb_ready:
        st.error("Build the knowledge base first.")
    elif not q.strip():
        st.warning("Enter a question.")
    else:
        try:
            res = answer(q)
            a = res.get("answer") if isinstance(res, dict) else str(res)
            st.text_area("Answer", value=a, height=360)
        except Exception as e:
            st.error(f"Error: {e}")
