"""Dokümanları küçük ve örtüşen metin parçalarına ayırır."""

import re


def _find_chunk_end(text: str, start: int, chunk_size: int, overlap: int) -> int:
    maximum_end = min(start + chunk_size, len(text))
    if maximum_end == len(text):
        return maximum_end
    section = text[start:maximum_end]
    minimum_length = max(chunk_size // 2, overlap + 1)
    matches = [
        match.end()
        for match in re.finditer(r"[.!?](?:[\"'”’\)\]]*)\s+", section)
        if match.end() >= minimum_length
    ]
    if matches:
        return start + matches[-1]
    last_space = section.rfind(" ", minimum_length)
    return start + last_space + 1 if last_space != -1 else maximum_end


def split_documents_into_chunks(documents: list[dict], chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """Dokümanları kaynak bilgilerini koruyarak chunklara böler."""
    if chunk_size <= 0:
        raise ValueError("chunk_size sıfırdan büyük olmalıdır.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap, 0 ile chunk_size arasında olmalıdır.")

    chunks: list[dict] = []
    chunk_number = 1
    for document in documents:
        text = document.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        start = 0
        while start < len(text):
            end = _find_chunk_end(text, start, chunk_size, overlap)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_number}",
                    "file_name": document.get("file_name"),
                    "file_path": document.get("file_path"),
                    "page_number": document.get("page_number"),
                    "text": chunk_text,
                })
                chunk_number += 1
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks
