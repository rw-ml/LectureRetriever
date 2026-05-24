# database connection layer

from sqlalchemy import create_engine, text, Column, Text
from sqlalchemy.orm import sessionmaker


from database.base import Base
from database.models import Chunk, Document, Lecture  #import after Base as it uses that too


class DBManager:
    def __init__(self, database_url: str, embedding_model: str = "intfloat/multilingual-e5-small"):
        '''
            :param
            embedding_model -- options:
            - intfloat/multilingual-e5-small    ~420MB
            - BAAI/bge-m3                       ~2.4GB
        '''
        self.database_url = database_url
        self.embedding_model = embedding_model

        # SQLite needs check_same_thread=False
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_session(self):
        return self.SessionLocal()

    def init_db(self):
        """Create tables and pgvector HNSW index if using Postgres."""
        Base.metadata.create_all(bind=self.engine)

    def delete_lecture(self, lecture_name: str):
        session = self.get_session()
        lecture = (
            session.query(Lecture)
            .filter(Lecture.name == lecture_name)
            .first()
        )
        if not lecture:
            session.close()
            return False

        session.delete(lecture)
        session.commit()
        session.close()
        return True

    def delete_document(self, document_name: str, lecture_name: str):
        session = self.get_session()
        document = (
            session.query(Document)
            .join(Lecture)
            .filter(
                Document.title == document_name,
                Lecture.name == lecture_name
            )
            .first()
        )
        if not document:
            session.close()
            return False

        session.delete(document)
        session.commit()
        session.close()
        return True

