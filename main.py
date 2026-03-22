from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse

from contextlib import asynccontextmanager
from pydantic import BaseModel

from api.service import AppService

# ---------- LIFESPAN ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = AppService(
        embedding_model="intfloat/multilingual-e5-small",
        generator_model="Qwen/Qwen3.5-2B",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    yield

# ---------- APP ----------
app = FastAPI(
    title="Lecture RAG API",
    lifespan=lifespan
)

# ---------- HELPER ----------
def get_service(request: Request) -> AppService:
    return request.app.state.service

@app.post("/upload")
def upload_lecture(
    request: Request,
    file: UploadFile = File(...),
    lecture_name: str = Form(...),
    document_title: str = Form(None)
):
    service = get_service(request)
    service.add_slide_set(
        pdf_file=file,
        lecture_name=lecture_name,
        document_title=document_title
    )
    if not document_title:
        document_title = file.filename
    return {"message": f"File {document_title} (part of the Lecture: '{lecture_name}') uploaded successfully"}

class AskRequest(BaseModel):
    question: str
    lecture_name: str

@app.get("/lectures")
def list_lectures(request: Request):
    service = get_service(request)
    return service.list_lectures()

@app.post("/ask_stream")
def ask_stream(req: AskRequest, request: Request):
    service = get_service(request)

    return StreamingResponse(
        service.generate_response(req.question, req.lecture_name),
        media_type="text/plain"
    )