# Clarifile

Clarifile is a lightweight Retrieval-Augmented Generation (RAG) system built with ChromaDB, SentenceTransformers, and an LLM (local Ollama Llama 3.1 or a hosted provider). It retrieves relevant chunks from a prebuilt vector store and generates grounded answers with citations only when sources are used. A minimal Streamlit app is included.
## Features
- PDF & URL ingestion with text cleaning and chunking  
- Persistent vector storage via ChromaDB  
- Top-k semantic search with scores  
- Context construction and deduplication  
- LLM query with local 'llama3.1:8b' 
- Automatic source citations
- Minimal Streamlit UI: KB summary + question box

## Structure
- ingest.py # Ingest PDFs/URLs 
- retriever.py # Retrieval + QA flow
- data/
    - (pdf files)
- storage/
    - (collection data)

## Requires
- Python ≥ 3.10
- SentenceTransformers all-MiniLM-L6-v2
- ChromaDB (persistent client)
- One LLM path:
    - Local: Ollama with llama3.1:8b
    - Hosted: Groq API (e.g., llama-3.1-8b-instant)
  
## Prebuild the Knowledge Base (Required)
The app expects a ready vector store in ./storage/. Preload all PDFs in data/Demo/ and URLs from data/Demo/urls.txt:

## Configure the LLM
Clarifile routes all generations through retriever.answer() → call_llm().
- Hosted (Streamlit Cloud / public demo):
    - Set secrets/env:
    - GROQ_API_KEY — Groq API key
    - MODEL_ID — e.g., llama-3.1-8b-instant
- Local (developer laptop):
    - Install Ollama and pull the model:
    - ollama pull llama3.1:8b


Leave GROQ_API_KEY unset to use http://localhost:11434.
## Installation
'''bash
pip install -r requirements.txt

## Usage
'''python
from ingest import ingest_pdfs, ingest_urls
from retriever import answer

ingest_pdfs(["data/testPDF1p.pdf", "data/testPDF3p.pdf"])
ingest_urls([
    "https://www.example.com",
    "https://jsonplaceholder.typicode.com/todos/1"
])

print(answer("What does the example site contain?"))

## Run the App
'''bash
streamlit run app.py
- The app lists detected demo PDFs and URLs.
- It requires ./storage/ to exist; if missing, build it first (see Prebuild).
- Ask a question; answers cite sources only when relevant chunks were used.
  or
- Scan the qr code to test app
<img width="322" height="329" alt="Clarifile-Demo" src="https://github.com/user-attachments/assets/38de6f07-bc5c-4f2f-9ddf-82c31d7ae75e" />


## Notices

**Built with Llama 3.1**

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.  
See the [Llama 3.1 Community License](https://llama.meta.com/llama-downloads) and [Acceptable Use Policy](https://llama.meta.com/llama3_1/use-policy) for more information.
