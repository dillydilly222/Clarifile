# Clarifile

Clarifile is a lightweight Retrieval-Augmented Generation (RAG) system using [ChromaDB](https://www.trychroma.com/), [SentenceTransformers](https://www.sbert.net/), and a local [Ollama](https://ollama.ai/) Llama 3.1 model. It ingests PDFs and URLs, retrieves relevant chunks, and generates context-grounded answers with citations.

## Features
- PDF & URL ingestion with text cleaning and chunking  
- Persistent vector storage via ChromaDB  
- Top-k semantic search with scores  
- Context construction and deduplication  
- LLM query with local `llama3.1:8b`  
- Automatic source citations  

## Structure
- ingest.py # Ingest PDFs/URLs 
- retriever.py # Retrieval + QA flow
- data/
--testPDF1p.pdf
--testPDF3p.pdf

##Requires
Ollama with llama3.1:8b
python version >= 3.13

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

## Notices

**Built with Llama 3.1**

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.  
See the [Llama 3.1 Community License](https://llama.meta.com/llama-downloads) and [Acceptable Use Policy](https://llama.meta.com/llama3_1/use-policy) for more information.
