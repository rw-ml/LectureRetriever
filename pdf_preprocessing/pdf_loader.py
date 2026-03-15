import pdfplumber
import typing
from pathlib import Path

def load_pdf(file_path: str):
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append({
                    "source": Path(file_path).name,
                    "page": i + 1,
                    "text": text
                })
    return pages

def load_all_pdfs(directory: str):
    documents = []
    for file_path in Path(directory).glob("*.pdf"):
        pages = load_pdf(file_path)
        documents.extend(pages)
    return documents

def load_multiple_pdfs(file_paths: list[str]):
    documents = []
    for file_path in file_paths:
        pages = load_pdf(file_path)
        documents.extend(pages)
    return documents