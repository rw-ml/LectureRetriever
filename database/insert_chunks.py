#insert chunks into database

from sentence_transformers import SentenceTransformer
from database.models import Document, Chunk


class DatasetInserter:
    def __init__(self, db_manager):
        """
        Class to handle inserting chunks into the database with embeddings.
        """
        self.db = db_manager
        self.embedding_model = db_manager.embedding_model
        self.model = SentenceTransformer(self.embedding_model)

    def add(self, chunks):
        """
        Insert a list of chunks into the database.

        Each chunk is a dict:
        {
            "chunk_id": int,
            "pages": list[int],
            "text": str,
            "source": str
        }
        """
        db = self.db.get_session()
        try:
            if not chunks:
                return
            # Use the source from the first chunk
            source = chunks[0].get("source", "unknown")

            # ---------- create document entry ----------
            document = Document(source=source)
            db.add(document)
            db.commit()
            db.refresh(document)
            document_id = document.id

            # ---------- batch embed ----------
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True
            )

            # ---------- insert chunks ----------
            for chunk, embedding in zip(chunks, embeddings):
                db_chunk = Chunk(
                    document_id=document_id,
                    pages=",".join(str(p) for p in chunk["pages"]),
                    text=chunk["text"],
                    embedding_model=self.embedding_model,
                    embedding_dimension=len(embedding),
                    embedding=embedding.tolist()
                )
                db_chunk.set_embedding(embedding)
                db.add(db_chunk)

            db.commit()

        finally:
            db.close()