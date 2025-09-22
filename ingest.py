from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = "storage"
EMBED = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name = "all-MiniLM-L6-v2"
)

#Split text into overlapping character windows
def chunk_text(text: str, chunk_chars: int = 1200, overlap_chars: int = 200) -> list[str]:
    if (not text or not text.strip()):
        return []
    elif(len(text) <= chunk_chars):
        text = text.strip()

    chunks: list[str] = []
    i = 0
    step_size = max(1, chunk_chars - overlap_chars)

    while (i < len(text)):
        j = i + chunk_chars
        segment = text[i:j]
        segment = segment.strip()
        if (segment):
            chunks.append(segment)
        i += step_size
    return chunks
        


