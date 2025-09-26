import os, json, requests
from pathlib import Path
import streamlit as st
from retriever import answer
from ingest import ingest_pdfs, ingest_urls

if "GROQ_API_KEY" in st.secrets: os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "MODEL_ID" in st.secrets:     os.environ["MODEL_ID"]     = st.secrets["MODEL_ID"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_ID = os.getenv("MODEL_ID", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_orig_post = requests.post

class _FakeResponse:
    def __init__(self, text, model, stream=False):
        self._text = text or ""
        self._model = model
        self._stream = stream
        self.status_code = 200
        self.ok = True
        self.headers = {"content-type": "application/json"}
    def json(self):
        return {"response": self._text, "done": True, "model": self._model}
    def iter_lines(self, chunk_size=1024, decode_unicode=None, delimiter=None):
        s = self._text
        step = max(1, min(64, len(s)//40 or 1))
        for i in range(0, len(s), step):
            yield (json.dumps({"model": self._model, "response": s[i:i+step], "done": False}) + "\n").encode("utf-8")
        yield (json.dumps({"model": self._model, "response": "", "done": True}) + "\n").encode("utf-8")
    def raise_for_status(self): return
    def close(self): return
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False

def _ollama_proxy_post(url, *args, **kwargs):
    try:
        if url.startswith("http://localhost:11434") and url.endswith("/api/generate"):
            if not GROQ_API_KEY:
                return _FakeResponse("GROQ_API_KEY not set", MODEL_ID, stream=False)
            payload = kwargs.get("json") or {}
            prompt = payload.get("prompt") or payload.get("input") or ""
            opts = payload.get("options") or {}
            max_tokens = int(opts.get("num_predict", payload.get("max_tokens", 256)) or 256)
            temperature = float(opts.get("temperature", payload.get("temperature", 0.7)) or 0.7)
            top_p = float(opts.get("top_p", payload.get("top_p", 0.9)) or 0.9)
            want_stream = bool(payload.get("stream", False) or kwargs.get("stream", False))
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            body = {
                "model": MODEL_ID,
                "messages": [
                    {"role": "system", "content": "You are a concise, helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stream": False
            }
            r = _orig_post(GROQ_URL, headers=headers, json=body, timeout=60)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            return _FakeResponse(text, MODEL_ID, stream=want_stream)
        return _orig_post(url, *args, **kwargs)
    except Exception as e:
        return _FakeResponse(f"Proxy error: {e}", MODEL_ID, stream=False)

requests.post = _ollama_proxy_post

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

def _log(s): st.session_state.log.append(s)

def _save_uploads(files):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files or []:
        p = UPLOAD_DIR / f.name
        p.write_bytes(f.read())
        paths.append(str(p))
    return paths

def _verify_demo():
    missing = [str(p) for p in DEMO_PDFS if not p.exists()]
    valid = [str(p) for p in DEMO_PDFS if p.exists()]
    return valid, missing

def build_kb(include_demo, include_urls, uploaded_paths):
    st.session_state.log = []
    try:
        if include_demo:
            valid, missing = _verify_demo()
            if missing:
                for m in missing: _log(f"Missing: {m}")
            if valid:
                _log(f"Ingesting {len(valid)} demo PDFs")
                ingest_pdfs(valid)
        if uploaded_paths:
            _log(f"Ingesting {len(uploaded_paths)} uploaded PDFs")
            ingest_pdfs(uploaded_paths)
        if include_urls:
            _log(f"Ingesting {len(ONLINE_URLS)} URLs")
            ingest_urls(ONLINE_URLS)
        MARKER_PATH.write_text("ok")
        st.session_state.kb_ready = True
        _log("Knowledge base ready")
        return True
    except Exception as e:
        st.session_state.kb_ready = False
        _log(f"Build failed: {e}")
        return False

st.title("Clarifile Demo")

with st.sidebar:
    st.subheader("Knowledge Base")
    use_demo = st.checkbox("Include bundled demo PDFs", value=True)
    use_urls = st.checkbox("Also fetch and ingest URLs (slower)", value=False)
    uploads = st.file_uploader("Add your PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Build / Refresh KB"):
        with st.spinner("Building knowledge base..."):
            uploaded_paths = _save_uploads(uploads)
            ok = build_kb(use_demo, use_urls, uploaded_paths)
        if ok: st.success("Knowledge base ready")
        else:   st.error("Build failed; see logs below")

st.markdown("Build logs")
st.code("\n".join(st.session_state.log) or "(no logs yet)")

st.markdown("---")
st.markdown("Ask a question")
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
