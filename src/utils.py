"""Dosya ve arayüz işlemlerinde kullanılan yardımcı fonksiyonlar."""

from pathlib import Path


def ensure_directory(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file, target_folder: str) -> str:
    ensure_directory(target_folder)
    target = Path(target_folder) / Path(uploaded_file.name).name
    target.write_bytes(uploaded_file.getbuffer())
    return str(target)


def shorten_text(text: str, max_length: int = 350) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= max_length else clean[: max_length - 3].rstrip() + "..."


def format_page_number(page_number) -> str:
    return str(page_number) if page_number is not None else "Belirtilmemiş"
