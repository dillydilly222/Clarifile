import os
from pathlib import Path
import traceback
import streamlit as st

# your modules
from retriever import answer
from ingest import ingest_pdfs, ingest_urls

st.set_page_config(page_title="Clarifile Demo")

ROOT = Path(__file__).parent.resolve()
DEMO_DIR = ROOT / "data" / "Demo"
DEMO_PDFS = [
    DEMO_DIR / "RAGInfo1.pdf",
    DEMO_DIR / "RAGInfo2.pdf",
    DEMO_DIR / "RAGInfo3.pdf",
    DEMO_DIR / "MLInfo1.pdf",
    DEMO_DIR / "AIInfo1.pdf",
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
UPLOAD_DIR = ROOT / "data" / "uploads"
MARKER_PATH = ROOT / ".kb_built"

if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = MARKER_PATH.exists()
if "log" not in st.session_state:
    st.session_state.log = []

def log(line: str):
    st.session_state.log.append(line)

def save_uploads(files):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files or []:
        p = UPLOAD_DIR / f.name
        p.write_bytes(f.read())
        paths.append(str(p))
    return paths

def verify_demo_pdfs():
    missing = [str(p) for p in DEMO_PDFS if not p.exists()]
    valid = [str(p) for p in DEMO_PDFS if p.exists()]
    return valid, missing

def build_kb(include_demo: bool, include_urls: bool, uploaded_paths: list[str]):
    st.session_state.log = []
    try:
        if include_demo:
            valid, missing = verify_demo_pdfs()
            if missing:
                log("Missing demo PDFs:")
                for m in missing:
                    log(f"- {m}")
            if valid:
                log(f"Ingesting {len(valid)} demo PDFs")
                ingest_pdfs(valid)
            else:
                log("No valid demo PDFs found; skipping demo ingest")

        if uploaded_paths:
            log(f"Ingesting {len(uploaded_paths)} uploaded PDFs")
            ingest_pdfs(uploaded_paths)

        if include_urls:
            log(f"Ingesting {len(ONLINE_URLS)} URLs (this may be slow)")
            ingest_urls(ONLINE_URLS)

        MARKER_PATH.write_text("ok")
        st.session_state.kb_ready = True
        log("Knowledge base ready")
        return True
    except Exception as e:
        st.session_state.kb_ready = False
        log("Build failed with exception:")
        log("".join(traceback.format_exception_only(type(e), e)).strip())
        # If you want the full stacktrace in the UI, uncomment:
        # log(traceback.format_exc())
        return False

st.title("Clarifile Demo")

with st.sidebar:
    st.subheader("Knowledge Base")
    use_demo = st.checkbox("Include bundled demo PDFs", value=True)
    use_urls = st.checkbox("Also fetch and ingest URLs (slower)", value=False)
    uploads = st.file_uploader("Add your PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("Build / Refresh KB"):
        with st.spinner("Building knowledge base..."):
            uploaded_paths = save_uploads(uploads)
            ok = build_kb(use_demo, use_urls, uploaded_paths)
        if ok:
            st.success("Knowledge base ready")
        else:
            st.error("Build failed; see logs below")

st.markdown("#### Build logs")
st.code("\n".join(st.session_state.log) or "(no logs yet)")

st.markdown("---")
st.markdown("### Ask a question")
q = st.text_area("Question", height=120, placeholder="Type your question about the ingested docs…")

if st.button("Ask"):
    if not st.session_state.kb_ready:
        st.error("Build the knowledge base first.")
    elif not q.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Retrieving and answering..."):
            try:
                res = answer(q)
                a = res.get("answer") if isinstance(res, dict) else str(res)
            except Exception as e:
                a = f"Error from answer(): {e}"
        st.text_area("Answer", value=a, height=360)
