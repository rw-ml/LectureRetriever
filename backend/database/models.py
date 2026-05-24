#defining database structure

from sqlalchemy import UniqueConstraint, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship

import json
import numpy as np

from database.base import Base

class Lecture(Base):
    '''
        Lecture
        ----------------
        id          : Integer primary key
        name        : String (e.g., "Reinforcement Learning")
    '''
    __tablename__ = "lectures"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)

    # relationship to chunks
    chunks = relationship("Chunk", back_populates="lecture", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="lecture", cascade="all, delete-orphan")

class Document(Base):
    '''
        documents
        -------------------
        id
        title
        source
    '''
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("lecture_id", "title", name="uq_document_source_title"),
    )
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    source = Column(String, nullable=False)

    #relationship
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    lecture_id = Column(Integer, ForeignKey("lectures.id"))
    lecture = relationship("Lecture", back_populates="documents")

class Chunk(Base):
    '''
        chunks
        --------------------------------
        id
        document_id
        pages
        text
        embedding
        embedding_model
        embedding_dimension
    '''
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True, index=True)
    pages = Column(String)
    text = Column(Text, nullable=False)

    # embedding metadata --> allow filtering by model
    embedding_model = Column(String)
    embedding_model_version = Column(String, nullable=True)
    embedding_created_at = Column(DateTime, nullable=True)

    embedding_dimension = Column(Integer)
    embedding = Column(Text, nullable=True)
    # relationship
    document_id = Column(Integer, ForeignKey("documents.id"))
    document = relationship("Document", back_populates="chunks")
    lecture_id = Column(Integer, ForeignKey("lectures.id"))
    lecture = relationship("Lecture", back_populates="chunks")

    # helper methods
    def set_embedding(self, vector: list | np.ndarray):
        self.embedding = json.dumps(vector.tolist() if hasattr(vector, "tolist") else vector)

    def get_embedding(self):
        return np.array(json.loads(self.embedding))