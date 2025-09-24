from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
import json
import io
import hashlib
import time
from chromadb.api.models import Collection

PERSIST_DIR = "./storage"
EMBED = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name = "all-MiniLM-L6-v2"
)

def clean_text(text: str | None) -> str:
    if (not text):
        return ""
    cleaned_text = text.replace("\u00A0", " ")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text


def chunk_text(text: str, chunk_chars: int = 1200, overlap_chars: int = 200) -> list[str]:
    """
    Split a string into overlapping character-based chunks.

    This function divides the given text into segments of up to 
    `chunk_chars` characters, with each segment overlapping the 
    previous one by `overlap_chars` characters. Useful for tasks 
    like text processing or preparing input for models with 
    maximum context length.

    Parameters:
        text (str): The input text to split.
        chunk_chars (int, optional): The maximum number of characters 
            per chunk. Default is 1200.
        overlap_chars (int, optional): The number of characters 
            each chunk should overlap with the previous one. 
            Default is 200.

    Returns:
        list[str]: A list of overlapping text chunks.
    """
    #Prepares the text for segmentation
    if (not text or not text.strip()):
        return []
    elif(len(text) <= chunk_chars):
        return [text.strip()]

    #Create vars needed for segmentation
    chunks: list[str] = []
    i = 0
    step_size = max(1, chunk_chars - overlap_chars)

    #Loops through the text and creates segements that are chunk_chars long
    while (i < len(text)):
        j = i + chunk_chars
        segment = text[i:j]
        segment = segment.strip()
        if (segment):
            chunks.append(segment)
        i += step_size
    return chunks

def load_pdf(file_obj) -> str:
    """
    Extract text from all pages of a PDF file.

    This function reads a PDF from the given file-like object and 
    concatenates the extracted text from each page into a single 
    string.

    Parameters:
        file_obj: A file-like object (e.g., an open file handle or 
            BytesIO stream) representing the PDF to read.

    Returns:
        str: The extracted text from the PDF, combined into a single 
        string. If no text is found, returns an empty string.
    """

    pdf_text: list[str] = []
    try: 
        #Checks to see what kind of file_obj was given and acts accordingly
        if (isinstance(file_obj, str)):
            file = open(file_obj, "rb")
        else:
            #Sees if it has .seek, if it does sets it to start at 0
            if hasattr (file_obj, "seek"):
                file_obj.seek(0)
            file = file_obj

        #Create the pdf_reader and the text on each page
        pdf_reader = PdfReader(file)
        pdf_text = [page.extract_text() or "" for page in pdf_reader.pages]

    #Errors that could occur
    except FileNotFoundError:
        raise ValueError(f"PDF {file} does not exist")
    except Exception as e:
        raise ValueError(f"Could not read PDF: {e}")
    finally:
        file.close()
    

    #Put page text into one string, each page is sepperated by \n
    return "\n".join(pdf_text)

def load_url(url: str) -> str:
    """
    Fetch a URL and return its visible text content.

    This function downloads the web page at the given URL and extracts
    all visible text using BeautifulSoup. The text blocks are joined
    with single spaces to preserve readability. No further cleaning or
    chunking is done here — callers are responsible for that.

    Parameters:
        url (str): The web address to fetch.

    Returns:
        str: The raw visible text content from the web page, or an empty
             string if the request fails.

    Raises:
        ValueError: If the request fails due to a network error or
                    unreachable URL.
    """

    try:
        #Check what kind of url was given and grab text accordingly
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        
        match content_type:
            case content_type if content_type.partition(";")[0].strip() == "application/pdf" or url.lower().endswith(".pdf"):
                pdf_bytes = io.BytesIO(response.content)
                return load_pdf(pdf_bytes)
            case content_type if "html" in content_type:
                soup = BeautifulSoup(response.text, "html.parser")
                return soup.get_text(" ")
            case content_type if "json" in content_type:
                return json.dumps(response.json(), indent=2)
            case content_type if "text" in content_type:
                return response.text
            case _:
                return ""
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to load URL: {url}") from e          
        
def build_or_get_collection(name: str = "docs") -> Collection:
    """
    Open or create a persistent Chroma collection using the global embedding function.

    Purpose:
        Ensures that a Chroma collection with the given name exists and is ready 
        for storing/retrieving vector embeddings. If the collection does not 
        already exist, it will be created. If it does exist, it will be opened 
        and reused.

    Args:
        name (str, optional): Name of the collection to open or create.
            Defaults to "docs".

    Returns:
        chromadb.api.models.Collection.Collection: A persistent Chroma collection 
        object backed by the configured embedding function (EMBED).
    """

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(name=name, embedding_function=EMBED)

def ingest_pdfs(files, col_name="docs") -> int:
    """
    Ingest one or more PDF files into a persistent Chroma collection.

    Purpose:
        Extracts text from PDF files, cleans and chunks the content, 
        and stores the resulting text chunks in a Chroma collection 
        along with metadata and unique IDs.

    Args:
        files (Iterable): A sequence of PDF file-like objects or paths 
            to PDF files. Each must be readable by load_pdf. If a file 
            object has a .name attribute, it will be used as the base 
            name; otherwise "upload.pdf" is used as a fallback.
        col_name (str, optional): Name of the Chroma collection where 
            chunks will be stored. Defaults to "docs".

    Returns:
        int: The total number of text chunks (documents) added to the 
        collection.
    
    Notes:
        IDs are generated in the format "<base_name>-<chunk_index>" 
        and must be unique within the collection. Re-ingesting the 
        same file with the same IDs will raise an error. During 
        development, either delete the storage/ folder between runs 
        or adjust the ID scheme (e.g., add a hash or timestamp).
    """
    #Collection to store pdf files and total number of documents
    collection = build_or_get_collection(col_name)
    total_docs = 0

    #Grab all pdf files and convert them to add to database
    for file_obj in files:
        #Create the lists needed to add pdfs to database
        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []
        if isinstance(file_obj, (str, Path)):
            base_name = Path(file_obj).name
        else:
            base_name = Path(getattr(file_obj, "name", "upload.pdf")).name
        
        raw_text = load_pdf(file_obj)
        cleaned_text = clean_text(raw_text)
        if not cleaned_text:
            continue
        text_parts = chunk_text(cleaned_text, chunk_chars=1200, overlap_chars=200)

        for i, text_part in enumerate(text_parts) :
            documents.append(text_part)
            metadatas.append({"source": base_name, "type": "pdf", "chunk": i})
            ids.append(f"{base_name}-{i}-{int(time.time())%100000}")
        
        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            total_docs += len(documents)
    
    return total_docs

def ingest_urls(urls, col_name="docs") -> int:
    """
    Ingest one or more URLs into a persistent Chroma collection.

    Purpose:
        Downloads and extracts visible text from URLs, cleans and chunks 
        the content, and stores the resulting text chunks in a Chroma 
        collection along with metadata and unique IDs.

    Args:
        urls (Iterable[str]): A sequence of URL strings to fetch and ingest.
            Each URL should point to an HTML page or other text-readable 
            resource. If the URL cannot be retrieved or parsed, it should 
            be skipped or logged as an error.
        col_name (str, optional): Name of the Chroma collection where 
            chunks will be stored. Defaults to "docs".

    Returns:
        int: The total number of text chunks (documents) added to the 
        collection.
    
    Notes:
        IDs are generated in the format "<base_name>-<chunk_index>", 
        where <base_name> is derived from the URL (e.g., the domain or 
        path). IDs must be unique within the collection. Re-ingesting 
        the same URL with the same IDs will raise an error. During 
        development, either delete the storage/ folder between runs 
        or adjust the ID scheme (e.g., add a hash or timestamp).
    """

    collection = build_or_get_collection(col_name)
    total_docs = 0

    for url_idx, url in enumerate(urls):
        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        source_url = str(url)

        try:
            raw_text = load_url(source_url)
        except ValueError:
            continue

        cleaned_text = clean_text(raw_text)
        if not cleaned_text:
            continue

        text_parts = chunk_text(cleaned_text, chunk_chars=1200, overlap_chars=200)

        #Create a readable shorter id for each url
        short = source_url.replace("https://", "").replace("http://", "").strip("/")
        short = short.replace("/", "-")[:80]

        for i, text_part in enumerate(text_parts):
            documents.append(text_part)
            metadatas.append({"source": source_url, "type": "url", "chunk": i})
            ids.append(f"{short}-{url_idx}-{i}")

        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            total_docs += len(documents)

    return total_docs

