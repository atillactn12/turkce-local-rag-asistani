"""TXT, Markdown ve PDF dokümanlarından metin çıkarır."""

from pathlib import Path

from pypdf import PdfReader


def _load_text_file(file_path: Path) -> list[dict]:
    try:
        text = file_path.read_text(encoding="utf-8").strip()
    except Exception as error:
        print(f"'{file_path}' okunamadı: {error}")
        return []
    if not text:
        return []
    return [{"file_name": file_path.name, "file_path": str(file_path), "page_number": None, "text": text}]


def _load_pdf_file(file_path: Path) -> list[dict]:
    documents: list[dict] = []
    try:
        pages = list(PdfReader(file_path).pages)
    except Exception as error:
        print(f"'{file_path}' PDF dosyası açılamadı: {error}")
        return documents

    for page_number, page in enumerate(pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as error:
            print(f"'{file_path}' dosyasının {page_number}. sayfası okunamadı: {error}")
            continue
        if text:
            documents.append({"file_name": file_path.name, "file_path": str(file_path), "page_number": page_number, "text": text})
    return documents


def load_documents_from_folder(folder_path: str) -> list[dict]:
    """Klasördeki desteklenen, boş olmayan dokümanları okur."""
    folder = Path(folder_path)
    documents: list[dict] = []
    if not folder.exists() or not folder.is_dir():
        print(f"Doküman klasörü bulunamadı: '{folder}'")
        return documents
    try:
        files = sorted(folder.iterdir())
    except Exception as error:
        print(f"'{folder}' klasörü okunamadı: {error}")
        return documents

    for file_path in files:
        if not file_path.is_file():
            continue
        extension = file_path.suffix.lower()
        if extension in {".txt", ".md"}:
            documents.extend(_load_text_file(file_path))
        elif extension == ".pdf":
            documents.extend(_load_pdf_file(file_path))
    return documents
