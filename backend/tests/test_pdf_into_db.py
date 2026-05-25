from backend.pdf_preprocessing.clean_text import clean_text_file
from backend.pdf_preprocessing.pdf_loader import load_pdf
from backend.chunking.chunking import RollingSemanticChunker, chunk_document
from backend.database.insert_chunks import DatasetInserter
from backend.database.db import DBManager

example_pdf = r"C:\Users\robin\Downloads\SWT2_Lecture.pdf"

#pdf to dict(page,topic and text)
txt_dict = load_pdf(example_pdf)
#cleaning multispaces and multi new lines
cleaned_txt_dict = clean_text_file(txt_dict)

print("Pages loaded:", len(cleaned_txt_dict))

n=0
for key in cleaned_txt_dict[n]:
    print(key+": ", cleaned_txt_dict[n][key])

#chunking
chunker = RollingSemanticChunker()
chunks = chunker.chunk_document(cleaned_txt_dict)

print("RollingSemantic Chunks:", len(chunks))

for c in chunks:
    print(c["chunk_id"], c["pages"], len(c["text"]), c["text"][:30])


chunks = chunk_document(cleaned_txt_dict)
print("Chunks created:", len(chunks))

for c in chunks:
    print("------")
    print("Chunk:", c["chunk_id"])
    print("Pages:", c["pages"])
    print("Preview:", len(c["text"]), c["text"][:100])

#DB_Manager
db_file = "../rag_db.sqlite"
sqlite_url = f"sqlite:///{db_file}"
# initialize DB manager
db_manager = DBManager(sqlite_url, embedding_model="intfloat/multilingual-e5-small")

# create tables (will create rag_db.sqlite automatically)
db_manager.init_db()
dataset_inserter = DatasetInserter(db_manager)
dataset_inserter.add(
    chunks,
    lecture_name="SE2",
    document_title="WholeLecture"
)