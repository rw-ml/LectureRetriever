# retriever.py

import numpy as np
from sqlalchemy import select
from sentence_transformers import SentenceTransformer, CrossEncoder

from database.models import Lecture, Document, Chunk


class Retriever:
    def __init__(self, db, embedding_model: str="intfloat/multilingual-e5-small", reranker: str="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        '''
        :param embedding_model: choices -- MUST FIT THE ENCODING MODEL USED IN DB CREATION
            - intfloat/multilingual-e5-small    ~420MB
            - BAAI/bge-m3                       ~2.4GB
        :param reranker: choices:
            MiniLM-L6 → 80–85% quality
            MiniLM-L12 → +5–8%
            BAAI/bge-reranker-base → +10–15%
            BAAI/bge-reranker-large → SOTA
        '''

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(self.embedding_model)
        self.db = db
        # stage 2 reranker (better but slower)
        self.reranker = CrossEncoder(reranker)

    def retrieve(self, query: str, lecture_name: str, top_k: int = 30):
        session = self.db.get_session()
        try:
            # -------- 1 get lecture id --------
            lecture = session.query(Lecture).filter(
                Lecture.name == lecture_name
            ).first()
            if lecture is None:
                raise ValueError(f"Lecture '{lecture_name}' not found in database.")
            lecture_id = lecture.id

            # -------- 2 load chunks from that lecture --------
            stmt = (
                select(Chunk)
                .join(Document)
                .where(
                    Document.lecture_id == lecture_id,
                    Chunk.embedding_model == self.embedding_model
                )
            )
            chunks = session.execute(stmt).scalars().all()
            if not chunks:
                print(f"Warning: No chunks found for {lecture_name}")
                return []

            # -------- 3 embed query --------
            query_embedding = self.model.encode(
                query,
                normalize_embeddings=True
            )

            # -------- 4 compute similarity --------
            chunk_embeddings = np.array(
                [c.get_embedding() for c in chunks]
            )
            similarities = chunk_embeddings @ query_embedding

            # -------- 5 select top-k --------
            top_indices = similarities.argsort()[::-1][:top_k]
            results = []
            for idx in top_indices:
                c = chunks[idx]
                results.append({
                    "chunk_id": c.id,
                    "text": c.text,
                    "pages": c.pages,
                    "document_id": c.document_id,
                    "document_title": c.document.title
                })
            return results
        finally:
            session.close()

    def rerank(self, query: str, candidates: list, top_k: int = 5):
        """
        Stage 2: cross-encoder reranking
        """
        if not candidates:
            return []
        pairs = [
            (query, c["text"])
            for c in candidates
        ]
        scores = self.reranker.predict(pairs, batch_size=32)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            {
                "text": c["text"],
                "pages": c["pages"],
                "document_id": c["document_id"],
                "document_title": c["document_title"],
                'chunk_id': c["chunk_id"],
                "score": float(score)
            }
            for c, score in ranked[:top_k]
        ]