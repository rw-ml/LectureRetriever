from pdf_preprocessing.clean_text import clean_text_file
from pdf_preprocessing.pdf_loader import load_pdf, load_multiple_pdfs, load_all_pdfs

example_pdf = r"C:\Users\robin\Downloads\02_CleanCode_WA.pdf"

txt_dict = load_pdf(example_pdf)
cleaned_txt_dict = clean_text_file(txt_dict)

print("Pages loaded:", len(cleaned_txt_dict))

n=50
for key in cleaned_txt_dict[n]:
    print(key+": ", cleaned_txt_dict[n][key])