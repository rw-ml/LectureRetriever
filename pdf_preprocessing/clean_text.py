import re
import typing

def clean_text(text: str):
    text = re.sub(r"\s+", " ", text) #remove multi space
    text = re.sub(r"\n+", "\n", text) #remove multi newline
    text = re.sub(r"\b\d+\b", "", text) #remove isolated numbers (as created for images)
    return text.strip()


CONTINUATION_PATTERN = re.compile(
    r'^((step|part|phase|example|stage|case)\s*(\d+|[ivxlcdm]+)[:\-]?|\d+[\.\)]|\(?continued\)?)',
    re.IGNORECASE
)

def propagate_titles(pages):
    last_heading = None
    for page in pages:
        lines = page["text"].split("\n")
        if not lines:
            continue
        first_line = lines[0].strip()
        # detect heading
        if len(first_line) < 80 and not CONTINUATION_PATTERN.match(first_line):
            last_heading = first_line
        # detect continuation
        elif CONTINUATION_PATTERN.match(first_line) and last_heading:
            page["text"] = last_heading + "\n" + page["text"]
    return pages

def clean_text_file(document: list[dict]):
    '''
        in place removal of excess whitespaces and new lines
    '''
    for page in document:
        page["text"] = clean_text(page["text"])
    #repeat titles for slidesets
    document = propagate_titles(document)
    return document