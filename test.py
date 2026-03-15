from pdf_preprocessing.clean_text import clean_text_file
from pdf_preprocessing.pdf_loader import load_pdf, load_multiple_pdfs, load_all_pdfs
from chunking import SemanticChunker

example_pdf = r"C:\Users\robin\Downloads\02_CleanCode_WA.pdf"

#pdf to dict(page,topic and text)
txt_dict = load_pdf(example_pdf)
#cleaning multispaces and multi new lines
cleaned_txt_dict = clean_text_file(txt_dict)

print("Pages loaded:", len(cleaned_txt_dict))

n=50
for key in cleaned_txt_dict[n]:
    print(key+": ", cleaned_txt_dict[n][key])

#chunking
chunker = SemanticChunker(
    embedding_model="intfloat/multilingual-e5-small",
    similarity_threshold=0.97,
    max_chars=2000
)
chunks = chunker.chunk_document(cleaned_txt_dict)

print(f"Generated {len(chunks)} chunks\n")

# first chunk
for key, val in chunks[0].items():
    print(f"{key}: {val}")