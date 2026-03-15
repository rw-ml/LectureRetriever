from sentence_transformers import SentenceTransformer
import numpy as np
from helpers_chunking import MultiCentroidManager

def window_similarity(embeddings, page_number, window_size=2):
    """
    Returns max similarity of page page_number with previous pages
    """

    start = max(0, page_number - window_size)
    sims = embeddings[start:page_number] @ embeddings[page_number]

    return sims.max() if len(sims) > 0 else 1.0


def split_large_text(text, max_chars):
    """
    Returns a tuple: (first_part, remainder)
    first_part fits into max_chars, remainder is the rest (or empty string)
    """
    if len(text) <= max_chars:
        return text, ""

    paragraphs = text.split("\n\n")
    current = ""
    for i, p in enumerate(paragraphs):
        if len(current) + len(p) < max_chars:
            current += (" " if current else "") + p
        else:
            break

    # join current as first part
    first_part = current.strip()
    remainder = " ".join(paragraphs[i:]).strip() if i < len(paragraphs) else ""

    return first_part, remainder










def semantic_chunk_pages(
        document: list[dict],
        embedding_model: str = "intfloat/multilingual-e5-small",
        similarity_threshold: float=0.65,
        sliding_window_size: int=3,
        max_chars:int=2000
    ):
    '''
    Semantic chunking for documents with multiple topics and large pages.
    Each page is a dict: {"source": str, "text": str, "page": int}

    :param document: document as list[dict] where dict has keys: "source", "text", "page"
    :param embedding_model:
        - "intfloat/multilingual-e5-small" --> higher speed,  ~420 MB
        - "BAAI/bge-m3" --> higher quality, slower, ~2.2 GB
    :param similarity_threshold: minimum cosine similarity required
    :param sliding_window_size:
    :param max_chars:
    :return:
    '''
    model = SentenceTransformer(embedding_model)

    texts = [page["text"] for page in document]
    page_embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    manager = MultiCentroidManager(similarity_threshold=similarity_threshold)
    source = document[0]["source"]
    for idx, page in enumerate(document):
        page_text = page["text"]
        page_num = page["page"]

        # Split if page is too large
        remaining_text = page_text
        while remaining_text:
            part, remaining_text = split_large_text(remaining_text, max_chars)

            # finish chunk when either topic changes or max size reached
            page_info = {
                "source": source,
                "page": page_num,
                "text": part
            }

            manager.add_page(page_embeddings[idx], page_info)


    return manager.get_chunks()




