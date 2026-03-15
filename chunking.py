from sentence_transformers import SentenceTransformer
import numpy as np
from helpers_chunking import MultiCentroidManager



from sklearn.cluster import AgglomerativeClustering

def detect_document_type(document, slide_threshold=800):
    """
    Detect if document is slides or script based on avg chars per page
    """
    avg_chars = sum(len(p["text"]) for p in document) / len(document)

    if avg_chars < slide_threshold:
        return "slides"
    else:
        return "script"



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




class SlidesChunker:
    def __init__(self, merge_pages=3, overlap=1, max_chars=2500):
        self.merge_pages = merge_pages
        self.overlap_pages = overlap
        self.max_chars = max_chars
        if overlap >= merge_pages:
            raise ValueError(f"overlap_pages ({overlap}) must be smaller than merge_pages ({merge_pages})")

    def chunk_document(self, document):
        chunks = []
        chunk_id = 0
        source = document[0]["source"]
        step = self.merge_pages - self.overlap_pages
        i = 0
        n = len(document)
        while i < n:
            end = min(i + self.merge_pages, n)
            pages = document[i:end]
            text = " ".join(p["text"] for p in pages)

            # respect max_chars
            if len(text) > self.max_chars:
                text = text[:self.max_chars]
            chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "pages": [p["page"] for p in pages],
                "text": text
            })
            chunk_id += 1
            if end == n:
                break
            i += step
        return chunks




class RollingSemanticChunker:
    def __init__(
        self,
        embedding_model="intfloat/multilingual-e5-small",
        window_size=3,
        deviation_factor=1.0,
        max_chars=2000
    ):
        """
        window_size: how many previous pages to compare with
        deviation_factor: how strong similarity must drop to trigger new chunk
        """
        self.model = SentenceTransformer(embedding_model)
        self.window_size = window_size
        self.deviation_factor = deviation_factor
        self.max_chars = max_chars

    def chunk_document(self, document):
        texts = [p["text"] for p in document]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )
        chunks = []
        current_chunk_pages = []
        current_chunk_text = []
        similarities = []
        source = document[0]["source"]
        chunk_id = 0
        for i, page in enumerate(document):
            page_embedding = embeddings[i]
            page_text = page["text"]
            page_num = page["page"]
            # ---------- similarity calculation ----------
            if i > 0:
                start = max(0, i - self.window_size)
                window_embeds = embeddings[start:i]

                sims = window_embeds @ page_embedding
                sim = sims.mean()

                similarities.append(sim)
                mean = np.mean(similarities)
                std = np.std(similarities)

                topic_change = sim < (mean - self.deviation_factor * std)
            else:
                topic_change = False

            # ---------- start new chunk ----------
            if topic_change and current_chunk_pages:
                chunks.append({
                    "chunk_id": chunk_id,
                    "source": source,
                    "pages": current_chunk_pages,
                    "text": " ".join(current_chunk_text)
                })

                chunk_id += 1
                current_chunk_pages = []
                current_chunk_text = []

            # ---------- handle large pages ----------
            remaining_text = page_text

            while remaining_text:
                part, remaining_text = split_large_text(
                    remaining_text,
                    self.max_chars
                )
                current_chunk_pages.append(page_num)
                current_chunk_text.append(part)

        # finalize last chunk
        if current_chunk_pages:
            chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "pages": current_chunk_pages,
                "text": " ".join(current_chunk_text)
            })
        return chunks


def chunk_document(document):
    doc_type = detect_document_type(document)
    if doc_type == "slides":
        chunker = SlidesChunker(
            merge_pages=3,
            max_chars=2000,
            overlap=1
        )
    else:
        chunker = RollingSemanticChunker(
            window_size=3,
            deviation_factor=1.0,
            max_chars=2000
        )
    return chunker.chunk_document(document)