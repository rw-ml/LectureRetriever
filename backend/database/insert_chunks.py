#insert chunks into database

from sentence_transformers import SentenceTransformer
from database.models import Document, Chunk, Lecture
from sqlalchemy.orm import Session


class DatasetInserter:
    def __init__(self, db_manager):
        """
        Class to handle inserting chunks into the database with embeddings.
        """
        self.db = db_manager
        self.embedding_model = db_manager.embedding_model
        self.model = SentenceTransformer(self.embedding_model)

    def _get_or_create_lecture(self, db: Session, lecture_name: str) -> Lecture:
        lecture = db.query(Lecture).filter(Lecture.name == lecture_name).first()
        if not lecture:
            lecture = Lecture(name=lecture_name)
            db.add(lecture)
            db.commit()
            db.refresh(lecture)
        return lecture

    def add(self, chunks, lecture_name: str=None, document_title:str=None):
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

            # ---------- lecture ----------
            lecture_id = None
            if lecture_name:
                lecture = self._get_or_create_lecture(db, lecture_name)
                lecture_id = lecture.id
            else:
                print("Warning Lecture Name not Found")

            # ---------- create document entry ----------
            document = Document(
                source=source,
                lecture_id=self._get_or_create_lecture(db, lecture_name).id,
                title=document_title if document_title else chunks[0].get("title", "Untitled")
            )
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
                    lecture_id=lecture_id,
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