import pdfplumber
import typing
from pathlib import Path
from fastapi import UploadFile
import os

def load_pdf(file_path: str):
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            words = page.extract_words()
            #fix some ordering issues
            words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
            text = " ".join(w["text"] for w in words_sorted)
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

import tempfile
import shutil

def handle_upload(file: UploadFile, delete_file: bool = False):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    pages = load_pdf(tmp_path)
    if delete_file:
        try:
            os.remove(tmp_path)
        except Exception as e:
            print(f"Warning: could not delete temp file {tmp_path}: {e}")
    return pages