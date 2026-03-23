# database connection layer

from sqlalchemy import create_engine, text, Column, Text
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector

from general_tools.db_name_sanitizer import clean_name

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
        self.embedding_model = clean_name(embedding_model)
        self.is_postgres = "postgresql" in database_url

        # SQLite needs check_same_thread=False
        connect_args = {"check_same_thread": False} if "sqlite" in database_url else {}
        self.engine = create_engine(database_url, connect_args=connect_args)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # dynamically patch embedding column type based on backend
        if self.is_postgres:
            Chunk.embedding = Column("embedding", Vector(), nullable=True)
        else:
            Chunk.embedding = Column("embedding", Text, nullable=True)

    def get_session(self):
        return self.SessionLocal()

    def init_db(self):
        """Create tables and pgvector HNSW index if using Postgres."""
        Base.metadata.create_all(bind=self.engine)
        if self.is_postgres:
            with self.engine.connect() as conn:
                # enable pgvector extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                # create HNSW index for this embedding model
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_{self.embedding_model}
                    ON chunks
                    USING hnsw (embedding vector_cosine_ops)
                    WHERE embedding_model = '{self.embedding_model}'
                """))
                conn.commit()

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

