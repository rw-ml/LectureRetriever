import re
import typing

def clean_text(text: str):
    text = re.sub(r"\s+", " ", text) #remove multi space
    text = re.sub(r"\n+", "\n", text) #remove multi newline
    return text.strip()

def clean_text_file(document: list[dict]):
    '''
        in place removal of excess whitespaces and new lines
    '''
    for page in document:
        page["text"] = clean_text(page["text"])

    return document