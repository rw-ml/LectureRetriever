from response_generation.retriever import Retriever
from response_generation.rag import RAGPipeline
from response_generation.llm import get_generator, get_generator_old
from database.db import DBManager



db = DBManager("sqlite:///../rag_db.sqlite")
db.init_db()

gen = get_generator_old("Qwen/Qwen3.5-2B")
retriever = Retriever(
    db,
    embedding_model="intfloat/multilingual-e5-small",
    reranker="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

rag = RAGPipeline(retriever, gen)

answer = rag.ask(
    "Was sind Solid Principles?",
    lecture_name="SE2"
)

print(answer)