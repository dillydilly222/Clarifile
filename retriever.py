from ingest import build_or_get_collection

def retrieve_chunks(query, k=5, col_name="docs"):
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

