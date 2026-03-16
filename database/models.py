#defining database structure

from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship

import json
import numpy as np

from database.base import Base

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
    document_id = Column(Integer, ForeignKey("documents.id"))
    pages = Column(String)
    text = Column(Text)

    # embedding metadata --> allow filtering by model
    embedding_model = Column(String)
    embedding_dimension = Column(Integer)
    embedding = None  # set in DBManager
    # relationship
    document = relationship("Document", back_populates="chunks")

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