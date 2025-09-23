# Clarifile

Clarifile is a compact RAG (Retrieve–Augment–Generate) mini-app. It ingests PDFs and web pages, cleans and chunks the text, and builds a **persistent** vector index (Chroma) for fast, citation-friendly retrieval. The codebase is small and interview-ready, showcasing practical AI integration (parsing, embeddings, vector search).

## Prerequisites
- Python **3.10+**
- macOS, Linux, or Windows
- (Optional) An OpenAI API key for the later “generate” step

## Installation

```bash
# 1) Clone and enter the repo
git clone <YOUR_FORK_OR_REPO_URL> clarifile
cd clarifile

# 2) Create and activate a virtual environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\activate

# 3) Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt