from collections import defaultdict

def build_sources(results, lecture_name):
    """
    Build citation list from retrieved chunks.
    """
    doc_pages = defaultdict(set)
    for r in results:
        title = r["document_title"]
        pages = r["pages"]
        if isinstance(pages, str):
            pages = [int(p) for p in pages.split(",")]

        for p in pages:
            doc_pages[title].add(p)

    lines = [f"Sources\n\nFrom: {lecture_name}"]
    for doc, pages in doc_pages.items():
        page_list = sorted(pages)
        pages_str = ",".join(str(p) for p in page_list)
        lines.append(f"-- from {doc} - pages {pages_str}")

    return "\n".join(lines)