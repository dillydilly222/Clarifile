from ingest import build_or_get_collection

DEFAULT_SYSTEM = (
    "You must answer strictly from the provided context. "
    "If the answer is not in the context, say you don't know."
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


