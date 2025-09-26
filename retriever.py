from ingest import build_or_get_collection
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests


DEFAULT_SYSTEM = (
    "You must answer strictly from the provided context. "
    "If the answer is not in the context, say you don't know."
)
PERSONA = (
    "You are an assistant who will answer user questions,"
    "using the provided context try and answer the query to the best of your ability, "
    "if you can not answer the query with the provided context, "
    "then answer with your own personal knowledge instead,"
    "ignoring the context provided (seeing as it has no corelation to the query),"
    "if you still can not answer the query, then reply with I do not know"
)

def retrieve_chunks(query, k=5, col_name="docs") -> list[dict]:
    """
    Retrieve the top-k most relevant documents from a vector collection.

    This function queries a vector database collection with the given input query
    and returns a list of records containing documents, metadata, and similarity scores.  
    Results are sorted by ascending distance (or equivalently descending score).

    Args:
        query (str): The input query string to search against the collection.
        k (int, optional): The number of top results to return. Defaults to 5.
        col_name (str, optional): The name of the collection to query.
            Defaults to "docs".

    Raises:
        ValueError: If `query` is None or an empty string.

    Returns:
        list[dict]: A list of result records, where each record is a dictionary with:
            document (str): The matched document text.
            source (str | None): The source of the document, if available in metadata.
            chunk (int | None): The chunk index of the document, if available.
            type (str | None): The type of document, if available in metadata.
            id (str): The unique identifier of the document.
            distance (float): The raw distance between the query and the document vector.
            score (float): A similarity score in [0.0, 1.0], defined as (1 - distance).
    """
    #Make sure a query was asked and ensure k >= 1
    if (not query or not str(query).strip()):
        raise ValueError("No query found")
    k = max(1, int(k))

    #Open the collection and query to find results
    collection = build_or_get_collection(col_name)
    result = collection.query(query_texts=[query], n_results=k, include=["documents","metadatas","distances"])


    #Sort and flatten the results and build the correct sorted records
    if (not result or not result.get("documents") or not result["documents"]):
        return []
    docs = result["documents"][0] or []
    metadatas = result.get("metadatas", [[]])[0] or []
    distances = result.get("distances", [[]])[0] or []
    ids_matrix = result.get("ids")  # may or may not exist in your version
    ids_ = ids_matrix[0] if ids_matrix else [None] * len(docs)

    records = []
    for i, (doc, metadata, distance) in enumerate(zip(docs, metadatas, distances)):
        #Ensure score is [0,1]
        dist = float(distance)
        score = 1.0 - dist
        if score < 0.0: score = 0.0
        if score > 1.0: score = 1.0

        #Make sure ther eis an id
        _id = ids_[i] if i < len(ids_) else f"{metadata.get('source')}-{metadata.get('chunk')}"
        #Add the data to the records dict
        metadata = metadata or {}
        records.append({
            "document": doc,
            "source": metadata.get("source"), 
            "chunk": metadata.get("chunk"),
            "type": metadata.get("type"),
            "id": _id,
            "distance": float(distance),
            "score": score,
        })
    records.sort(key=lambda r: r["distance"])  # or: reverse sort by r["score"]
    return records

def build_context(chunks: list[dict], max_chars: int = 6000) -> tuple[str, list[dict]]:
    """
    Build a consolidated context string from retrieved chunks.

    This function concatenates the text of retrieved chunks into a single context
    block, respecting a character budget, and returns both the context and the
    subset of chunks that were included (useful for citations).

    Args:
        chunks (list[dict]): The retrieved result records produced by 'retrieve_chunks'.
        max_chars (int, optional): Maximum number of characters allowed in the
            final context string. Defaults to 6000.

    Raises:
        ValueError: If `chunks` is None.

    Returns:
        tuple[str, list[dict]]: A tuple containing:
            context (str): The concatenated context text assembled from chunks.
            used_chunks (list[dict]): The subset of input chunks that fit within
                the 'max_chars' limit, preserving original order.
    """
    #Check if chunks has something and max_chars is not less than zero
    if (chunks is None):
        raise ValueError("chunks is None")
    if (max_chars < 0):
        max_chars = 0

    #Set up needed lists and sets
    used_chunks: list[dict] = []
    chunk_parts: list[str] = []
    total = 0
    seen_chunks: set[tuple[str | None, str | None, int | None]] = set()

    #Loop through the chunks and grab needed info to piece back together for prompt
    for chunk in chunks:
        #Make sure its not a duplicate
        if not isinstance(chunk, dict):
            continue

        #Grab keys needed and make sure they are not duplicates
        key = (chunk.get("id"), chunk.get("source"), chunk.get("chunk"))
        if (key in seen_chunks):
            continue
        seen_chunks.add(key)

        #Grab the used doc and make sure it is not empty
        doc = (chunk.get("document") or "").strip()
        if (not doc):
            continue
        
        #Grab needed info and put it all together
        source = chunk.get("source")
        chunk_index = chunk.get("chunk")
        block = (f"Source: {source} (chunk {chunk_index})\n{doc}\n\n")
        if (total + len(block) > max_chars):
            break
        chunk_parts.append(block)
        used_chunks.append(chunk)
        total += len(block)
    return "".join(chunk_parts), used_chunks
    
def make_prompt(query: str, context: str, system_msg: str = DEFAULT_SYSTEM) -> str:
    """
    Construct a prompt for the language model using the query and context.

    This function formats a prompt that instructs the model to answer strictly
    from the provided context and responds with the user's question.

    Args:
        query (str): The user's natural language question.
        context (str): The consolidated context text built from retrieved chunks.
        system_msg (str, optional): System-level instruction that sets answer policy.
            Defaults to 'DEFAULT_SYSTEM'.

    Raises:
        ValueError: If 'query' or 'context' is None or empty.

    Returns:
        str: A single prompt string combining the system instruction, context,
            and the user's question, ready to be sent to an LLM.
    """
    #Make sure that there is a query, context
    if(not query):
        raise ValueError("No query found")
    if(not context):
        raise ValueError("No context found")

    #Strip qury and context, and make sure they are still not empty
    q = query.strip()
    c = context.strip()
    if(not q):
        raise ValueError("Query must be non-empty")
    if(not c):
        raise ValueError("Context must be non-empty")
    #See if custom system_msg was added if not result to default
    sys = (system_msg or "").strip()
    if not sys:
        sys = DEFAULT_SYSTEM
    
    return f"{sys}\n\nContext:\n{c}\n\nQuestion: {q}\nAnswer:"

def call_llm(prompt: str) -> str:
    """
    Execute a language model call to produce an answer from a prompt.

    This function should invoke the Llama3 model,
    passing the constructed prompt and returning the resulting answer text.

    Args:
        prompt (str): The prompt string created by 'make_prompt'.

    Raises:
        NotImplementedError: If the method is not yet implemented.
        RuntimeError: If the underlying model call fails or returns no output.

    Returns:
        str: The model-generated answer text (without appended citations).
    """
    #Grab groq api key and model name for streamlit app
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
    try:
        #Using the streamlit demo
        if groq_key:
            headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
            body = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": PERSONA},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 512,
                "top_p": 0.9,
                "stream": False,
            }
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              headers=headers, json=body, timeout=60)
            r.raise_for_status()
            data = r.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        else:
            #Grab the response from the local LLM call and get just the answer portion
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.1:8b", "prompt": f"{PERSONA}\nUser:{prompt}", "stream": False},
                timeout=300  # 5 min timeout for long generations
            )
            response.raise_for_status()
            data = response.json()
            text = (data.get("response") or "").strip()
            #Make sure an answer was given
            if not text:
                raise RuntimeError("empty response from Llama 3.1:8b")
            return text
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")
    
def label_sources(chunks: list[dict]) -> dict[str, int]:
    """
    Assign numeric citation labels to unique sources.

    This function scans the subset of chunks actually included in the context
    and maps each unique 'source' string to a stable integer label starting at 1.

    Args:
        used_chunks (list[dict]): The chunk records that were included in the context.

    Raises:
        ValueError: If 'used_chunks' is None.

    Returns:
        dict[str, int]: A mapping from source string to numeric label, e.g.:
            {"https://example.com": 1, "testPDF1p.pdf": 2}
    """
    #See if chunks were given for source labeling
    if chunks is None:
        raise ValueError("'used_chunks' is None")
    #Create label dict and add the source for each chunk to the labels
    labels: dict[str,int] = {}
    label_num = 1
    for chunk in chunks:
        source = chunk.get("source")
        if (source not in labels):
            labels[source] = label_num
            label_num += 1
    return labels

def render_answer_with_citations(answer_text: str, used_chunks: list[dict]) -> str:
    """
    Append a Sources section with numeric citations to an answer.

    This function adds a compact 'Sources' list to the end of the model's answer,
    enumerating each unique source that contributed context.

    Args:
        answer_text (str): The raw answer text produced by the language model.
        used_chunks (list[dict]): The chunk records used to build the context.

    Raises:
        ValueError: If 'answer_text' is None or 'used_chunks' is None.

    Returns:
        str: The final answer string with a trailing 'Sources:' section, e.g.:
            Sources:
            [1] https://example.com (chunk 0)
            [2] data/testPDF1p.pdf (chunk 3)
    """
    #Make sure an answer was provided before adding citations
    if answer_text is None or used_chunks is None:
        raise ValueError("'answer_text' or 'used_chunks' is None")
    #Create dict, list and set for matching the text to the right source
    mapping_sources = label_sources(used_chunks)
    lines = [answer_text.rstrip(), "", "Sources:"]
    seen = set()

    #Match text to the right source
    for chunk in used_chunks:
        source = chunk.get("source")
        if (source in seen):
            continue
        seen.add(source)
        lines.append(f"[{mapping_sources[source]}] {source} (chunk {chunk.get('chunk')})")
    return "\n".join(lines)
    
def answer(query: str, *, k: int = 5, col_name: str = "docs", max_context_chars: int = 6000, system_msg: str = DEFAULT_SYSTEM, with_citations: bool = True,) -> dict:
    """
    Generate an answer by retrieving context and querying a language model.

    This function orchestrates a complete RAG flow: it retrieves the most relevant
    chunks, builds a context string, constructs a prompt, calls the language model,
    and optionally appends a Sources section with citations.

    Args:
        query (str): The user's question to be answered.
        k (int, optional): The number of top chunks to retrieve. Defaults to 5.
        col_name (str, optional): The name of the vector collection to query.
            Defaults to "docs".
        max_context_chars (int, optional): Maximum character budget for the context.
            Defaults to 6000.
        system_msg (str, optional): System instruction that governs answer policy.
            Defaults to 'DEFAULT_SYSTEM'.
        with_citations (bool, optional): Whether to append a 'Sources' section
            listing contributing documents. Defaults to True.

    Raises:
        ValueError: If 'query' is None or empty.
        RuntimeError: If the model call fails or produces no answer.

    Returns:
        dict: A dictionary containing:
            answer (str): The final answer string (with citations if enabled).
            prompt (str): The exact prompt used for the model call.
            sources (list[dict]): A compact list of source metadata for the
                chunks that were included in the context, where each item has:
                source (str | None): The document or URL source.
                chunk (int | None): The chunk index within that source.
                score (float): Normalized similarity score for the chunk.
                id (str): The unique identifier of the chunk in the store.
    """
    #Check that a query was given and is not empty
    if not query or not str(query).strip():
        raise ValueError("query is empty")

    #Grab the chunks, context, and create the prompt for the llm call
    chunks = retrieve_chunks(query, k=k, col_name=col_name)
    context, used = build_context(chunks, max_chars=max_context_chars)
    prompt = make_prompt(query, context, system_msg=PERSONA)

    #Make a call to the llm with the made prompt and make sure an answer was given
    raw = call_llm(prompt)
    if not raw or not str(raw).strip():
        raise RuntimeError("Model returned no output")

    #Make the answer more readable for the end user, adding proper citations
    final = render_answer_with_citations(raw, used) if with_citations else str(raw).strip()
    sources = [
        {"source": r.get("source"), "chunk": r.get("chunk"), "score": r.get("score"), "id": r.get("id")}
        for r in used
    ]
    return {"answer": final, "prompt": prompt, "sources": sources}