"""
Utility functions for RAG
"""


def extract_metadata(chunk: str) -> dict:
    subject = "Unknown"
    for s in [
        "Physics",
        "Chemistry",
        "Mathematics",
        "Computer",
        "Computational",
        "Compute",
    ]:
        if s.lower() in chunk.lower():
            subject = s
            break
    page_number = None
    if "Page:" in chunk:
        import re

        match = re.search(r"Page:\s*(\d+)", chunk)
        if match:
            page_number = int(match.group(1))
    return {"subject": subject, "page_number": page_number}
