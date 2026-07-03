"""LLM indirmeden loader, chunker ve retriever akışını test eder."""

from pathlib import Path

from src.chunker import split_documents_into_chunks
from src.document_loader import load_documents_from_folder
from src.retriever import SimpleRetriever


QUESTIONS = [
    "Bu projenin amacı nedir?",
    "RAG bu projede nasıl kullanılıyor?",
    "Finalde ne teslim edilecek?",
]


def main() -> None:
    documents_folder = Path(__file__).resolve().parent / "data" / "documents"
    documents = load_documents_from_folder(str(documents_folder))
    chunks = split_documents_into_chunks(documents)
    if not chunks:
        raise RuntimeError("Retrieval testi için chunk oluşturulamadı.")

    retriever = SimpleRetriever()
    retriever.build_index(chunks)
    print(f"Okunan doküman/sayfa: {len(documents)}")
    print(f"Oluşturulan chunk: {len(chunks)}")

    for question in QUESTIONS:
        print(f"\nSoru: {question}")
        results = retriever.search(question, top_k=3)
        if not results:
            raise RuntimeError(f"Soru için kaynak bulunamadı: {question}")
        for result in results:
            print(
                f"- {result['file_name']} | Sayfa: {result['page_number']} | "
                f"Skor: {result['score']:.4f} | {result['chunk_id']}"
            )


if __name__ == "__main__":
    main()
