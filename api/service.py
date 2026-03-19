from typing import Optional
from pdf_preprocessing.clean_text import clean_text_file
from pdf_preprocessing.pdf_loader import handle_upload
from chunking.chunking import RollingSemanticChunker, chunk_document
from database.insert_chunks import DatasetInserter
from database.db import DBManager

from response_generation.retriever import Retriever
from response_generation.rag import RAGPipeline
from response_generation.llm import get_generator, get_generator_old


class IngestionService:
    def __init__(self, db_manager: DBManager, embedding_model: str="intfloat/multilingual-e5-small"):
        self.db_manager = db_manager
        # preprocessing
        self.chunker = RollingSemanticChunker()
        # saving
        self.dataset_inserter = DatasetInserter(self.db_manager)

    def add_slide_set(self, pdf_file, lecture_name, document_title: Optional[str]=None, store_file:bool=False):
        txt_dict = handle_upload(pdf_file, not store_file)
        cleaned_txt_dict = clean_text_file(txt_dict)
        chunks = self.chunker.chunk_document(cleaned_txt_dict)
        self.dataset_inserter.add(
            chunks,
            lecture_name=lecture_name,
            document_title=document_title if document_title else pdf_file.filename
        )

class QAService:
    def __init__(
            self,
            db_manager: DBManager,
            embedding_model: str="intfloat/multilingual-e5-small",
            generator_model: str="Qwen/Qwen3.5-2B",
            reranker_model: str="intfloat/multilingual-e5-small"
    ):

        self.db_manager = db_manager
        self.generator = get_generator_old(generator_model)
        retriever = Retriever(
            self.db_manager,
            embedding_model=embedding_model,
            reranker=reranker_model
        )
        self.rag = RAGPipeline(retriever, self.generator)

    def generate_response(self, question: str, lecture_name: str):
        answer = self.rag.ask(
            question,
            lecture_name=lecture_name
        )
        return answer

class AppService:
    def __init__(
            self,
            embedding_model: str="intfloat/multilingual-e5-small",
            generator_model: str="Qwen/Qwen3.5-2B",
            reranker_model: str="cross-encoder/ms-marco-MiniLM-L-6-v2",
            db_path: str = "data/rag_db.sqlite"
    ):
        # -- db insertion -----------------------------
        sqlite_url = f"sqlite:///{db_path}"
        self.db_manager = DBManager(sqlite_url, embedding_model=embedding_model)
        # create tables (will create rag_db.sqlite automatically)
        self.db_manager.init_db()

        self.ingester = IngestionService(self.db_manager, embedding_model=embedding_model)
        self.response_generator = QAService(self.db_manager,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            generator_model=generator_model
        )
    def add_slide_set(self, pdf_file, lecture_name, document_title: Optional[str]=None, store_file:bool=False):
        self.ingester.add_slide_set(pdf_file, lecture_name, document_title, store_file)

    def generate_response(self, question: str, lecture_name: str):
        return self.response_generator.generate_response(question, lecture_name)

    def list_lectures(self):
        session = self.db_manager.get_session()
        try:
            from database.models import Lecture
            lectures = session.query(Lecture).all()
            return [l.name for l in lectures]
        finally:
            session.close()
