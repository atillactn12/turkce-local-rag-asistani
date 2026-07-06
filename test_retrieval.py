"""LLM indirmeden loader, chunker ve retriever akışını test eder."""

from pathlib import Path

from src.chunker import split_documents_into_chunks
from src.document_loader import load_documents_from_folder
from src.retriever import SimpleRetriever


QUESTIONS = [
    (
        "Akıllı Kampüs Enerji Yönetim Sistemi'nin temel amacı nedir?",
        ("temel amacı", "ana hedef", "elektrik tüketimini"),
    ),
    (
        "Sistemin hedef kullanıcıları kimlerdir?",
        ("hedef kullanıcılar", "kampüs enerji yöneticileri"),
    ),
    (
        "Anomali tespiti ne anlama gelir?",
        ("anomali tespiti", "beklenen enerji tüketiminden"),
    ),
]


def main() -> None:
    documents_folder = Path(__file__).resolve().parent / "data" / "documents"
    database_path = Path(__file__).resolve().parent / "data" / "index" / "rag_index.sqlite"
    documents = load_documents_from_folder(str(documents_folder))
    if not documents:
        raise RuntimeError("Retrieval testi için doküman bulunamadı.")
    chunks = split_documents_into_chunks(documents)
    if not chunks:
        raise RuntimeError("Retrieval testi için chunk oluşturulamadı.")

    retriever = SimpleRetriever(db_path=str(database_path))
    retriever.build_index(chunks)
    if not database_path.exists():
        raise RuntimeError("SQLite index dosyası oluşturulamadı.")
    if not retriever.sqlite_ready:
        raise RuntimeError("SQLite vektör indeksi hazır değil.")

    print(f"Okunan doküman/sayfa: {len(documents)}")
    print(f"Oluşturulan chunk: {len(chunks)}")
    print(f"SQLite index: {database_path}")

    for question, expected_terms in QUESTIONS:
        print(f"\nSoru: {question}")
        results = retriever.search(question, top_k=3)
        if not results:
            raise RuntimeError(f"Soru için kaynak bulunamadı: {question}")
        combined_text = " ".join(result["text"] for result in results).casefold()
        if not any(term.casefold() in combined_text for term in expected_terms):
            raise RuntimeError(f"Soru için ilgili içerik bulunamadı: {question}")
        for result in results:
            print(
                f"- {result['file_name']} | Sayfa: {result['page_number']} | "
                f"Skor: {result['score']:.4f} | {result['chunk_id']}"
            )

    print(f"\nArama backend'i: {retriever.last_search_backend}")
    retriever.close()


if __name__ == "__main__":
    main()
