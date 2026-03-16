#defining database structure

from sqlalchemy import Column, Integer, String, Text, ForeignKey
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
    chunks = relationship("Chunk", back_populates="lecture")
    documents = relationship("Document", back_populates="lecture")

class Document(Base):
    '''
        documents
        -------------------
        id
        title
        source
    '''
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    source = Column(String)

    #relationship
    chunks = relationship("Chunk", back_populates="document")
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
    text = Column(Text)

    # embedding metadata --> allow filtering by model
    embedding_model = Column(String)
    embedding_dimension = Column(Integer)
    embedding = None  # set in DBManager
    # relationship
    document_id = Column(Integer, ForeignKey("documents.id"))
    document = relationship("Document", back_populates="chunks")
    lecture_id = Column(Integer, ForeignKey("lectures.id"))
    lecture = relationship("Lecture", back_populates="chunks")

    # helper methods
    def set_embedding(self, vector: list | np.ndarray):
        if hasattr(self.embedding, "bind_expression"):  # pgvector Column
            self.embedding = vector
        else:
            self.embedding = json.dumps(vector.tolist() if hasattr(vector, "tolist") else vector)

    def get_embedding(self):
        if hasattr(self.embedding, "bind_expression"):
            return self.embedding
        else:
            return np.array(json.loads(self.embedding))