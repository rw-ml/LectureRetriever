from typing import Optional, Generator
from huggingface_hub import snapshot_download

from pdf_preprocessing.clean_text import clean_text_file
from pdf_preprocessing.pdf_loader import handle_upload
from chunking.chunking import RollingSemanticChunker
from database.insert_chunks import DatasetInserter
from database.db import DBManager

from response_generation.retriever import Retriever
from response_generation.rag import RAGPipeline
from response_generation.llm import VLLMClient #get_generator, get_generator_old
from api.vllm_manager import VLLMManager

class IngestionService:
    def __init__(self, db_manager: DBManager):
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
            reranker_model: str="intfloat/multilingual-e5-small",
            quantization=None,
            max_tokens=512,
            temperature=0.0,
            max_model_len=4096
    ):

        self.db_manager = db_manager
        snapshot_download(
            repo_id="Qwen/Qwen3.5-2B"
        )

        print("vLLM Model Startup. This can take a few minutes...")
        self.vllm_manager = VLLMManager(
            model_name=generator_model,
            port=30001,
            gpu_memory_utilization=0.7,
            max_model_len=max_model_len,
            quantization=quantization,
        )
        self.vllm_manager.start()

        self.vLLM_client = VLLMClient(
            base_url=self.vllm_manager.get_url(),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # self.generator = # get_generator_old(generator_model)
        retriever = Retriever(
            self.db_manager,
            embedding_model=embedding_model,
            reranker=reranker_model
        )
        self.rag = RAGPipeline(retriever, self.vLLM_client)
    def generate_response(self, question: str, lecture_name: str) -> Generator[str, None, None]:
        answer = self.rag.ask_stream(
            question,
            lecture_name=lecture_name
        )
        return answer

    def shutdown(self):
        self.vllm_manager.stop()


class AppService:
    def __init__(
            self,
            embedding_model: str="intfloat/multilingual-e5-small",
            generator_model: str="Qwen/Qwen3.5-2B",
            reranker_model: str="cross-encoder/ms-marco-MiniLM-L-6-v2",
            db_path: str = "app/data/rag_db.sqlite"
    ):
        # -- db insertion -----------------------------
        sqlite_url = f"sqlite:////{db_path}"
        self.db_manager = DBManager(sqlite_url, embedding_model=embedding_model)
        # create tables (will create rag_db.sqlite automatically)
        self.db_manager.init_db()

        self.ingester = IngestionService(self.db_manager)
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

    def list_documents_in_lecture(self, lecture_name: str):
        session = self.db_manager.get_session()
        try:
            from database.models import Document, Lecture

            documents = (
                session.query(Document)
                .join(Lecture)
                .filter(Lecture.name == lecture_name)
                .all()
            )

            return [{"id": d.id, "title": d.title} for d in documents]
        finally:
            session.close()

    def delete_lecture(self, lecture_name: str):
        return self.db_manager.delete_lecture(lecture_name)

    def delete_document(self, document_name: str, lecture_name: str):
        return self.db_manager.delete_document(document_name,lecture_name)

    def shutdown(self):
        self.response_generator.shutdown()