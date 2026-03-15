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
    text = text.strip()
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


    if not first_part:
        first_part = text[:max_chars]
        remainder = text[max_chars:].strip()
    else:
        remainder = " ".join(paragraphs[i:]).strip() if i < len(paragraphs) else ""

    return first_part, remainder







class SemanticChunker:
    def __init__(self,
                 embedding_model: str = "intfloat/multilingual-e5-small",
                 similarity_threshold: float = 0.9,
                 max_chars: int = 2000):
        '''
        Semantic chunking for documents with multiple topics and large pages.
        :param embedding_model:
            - "intfloat/multilingual-e5-small" --> higher speed,  ~420 MB
            - "BAAI/bge-m3" --> higher quality, slower, ~2.2 GB
        :param similarity_threshold: minimum cosine similarity required
        '''
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_chars = max_chars
        self.model = SentenceTransformer(embedding_model)

    def chunk_document(self, document: list[dict]):
        """
        document: list[dict] with keys: 'source', 'text', 'page'
        Returns list of chunks with 'chunk_id', 'source', 'pages', 'text'
        """
        texts = [page["text"] for page in document]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        manager = MultiCentroidManager(similarity_threshold=self.similarity_threshold)

        for idx, page in enumerate(document):
            embedding = embeddings[idx]
            page_text = page["text"]
            page_num = page["page"]
            source = page["source"]

            remaining_text = page_text
            while remaining_text:
                part, remaining_text = split_large_text(remaining_text, self.max_chars)

                page_info = {
                    "source": source,
                    "page": page_num,
                    "text": part
                }
                manager.add_page(embedding, page_info)

        return manager.get_chunks()





