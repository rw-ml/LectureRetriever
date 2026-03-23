from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import StreamingResponse
import docker
from contextlib import asynccontextmanager
from pydantic import BaseModel

from api.service import AppService

# ---------- LIFESPAN ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    #startup
    app_service = AppService(
        embedding_model="intfloat/multilingual-e5-small",
        generator_model="Qwen/Qwen3.5-2B",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    app.state.service = app_service
    yield
    #shutdown
    app_service.shutdown()


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

@app.delete("/lectures/{lecture_name}")
def delete_lecture(lecture_name: str, request: Request):
    service = get_service(request)
    success = service.delete_lecture(lecture_name)
    if not success:
        raise HTTPException(status_code=404, detail="Lecture not found")

    return {"status": "lecture deleted"}


@app.get("/lectures/{lecture_name}/documents")
def list_documents(lecture_name: str, request: Request):
    service = get_service(request)
    return service.list_documents_in_lecture(lecture_name)

@app.delete("/lectures/{lecture_name}/documents/{document_name}")
def delete_document(
    lecture_name: str,
    document_name: str,
    request: Request
):
    service = get_service(request)
    success = service.delete_document(document_name, lecture_name)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"status": "document deleted"}



@app.post("/ask_stream")
def ask_stream(req: AskRequest, request: Request):
    service = get_service(request)

    return StreamingResponse(
        service.generate_response(req.question, req.lecture_name),
        media_type="text/plain"
    )