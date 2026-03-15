from pdf_preprocessing.clean_text import clean_text_file
from pdf_preprocessing.pdf_loader import load_pdf, load_multiple_pdfs, load_all_pdfs
from chunking import RollingSemanticChunker, chunk_document

example_pdf = r"C:\Users\robin\Downloads\02_CleanCode_WA.pdf"

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