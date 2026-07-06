"""LLM indirmeden loader, chunker ve retriever akışını test eder."""

import os
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

    # Test hiçbir Foundry modeli yüklememeli veya indirmemelidir. Kullanıcının
    # terminalinde alias tanımlı olsa bile bu süreçte geçici olarak devre dışıdır.
    previous_embedding_alias = os.environ.pop(
        "FOUNDRY_EMBEDDING_MODEL_ALIAS", None
    )
    retriever = None
    hybrid_retriever = None
    try:
        retriever = SimpleRetriever(db_path=str(database_path))
        if retriever.retrieval_mode != "tfidf":
            raise RuntimeError("Varsayılan arama yöntemi TF-IDF olmalıdır.")
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
            combined_text = " ".join(
                result["text"] for result in results
            ).casefold()
            if not any(term.casefold() in combined_text for term in expected_terms):
                raise RuntimeError(f"Soru için ilgili içerik bulunamadı: {question}")
            for result in results:
                print(
                    f"- {result['file_name']} | Sayfa: {result['page_number']} | "
                    f"Skor: {result['score']:.4f} | {result['chunk_id']}"
                )

        print(f"\nVarsayılan arama backend'i: {retriever.last_search_backend}")

        hybrid_retriever = SimpleRetriever(
            db_path=str(database_path), retrieval_mode="hybrid"
        )
        hybrid_retriever.build_index(chunks)
        hybrid_results = hybrid_retriever.search(QUESTIONS[0][0], top_k=3)
        if not hybrid_results:
            raise RuntimeError("Hybrid mod TF-IDF yedeğiyle sonuç döndürmedi.")
        if hybrid_retriever.foundry_embedding_ready:
            raise RuntimeError("Testte Foundry embedding modeli etkin olmamalıdır.")
        if not hybrid_retriever.fell_back_to_tfidf:
            raise RuntimeError("Hybrid mod TF-IDF yedeğine dönmedi.")
        print(
            "Hybrid arama: embedding kullanılamadığında "
            f"{hybrid_retriever.last_search_backend} ile çalıştı."
        )
    finally:
        if retriever is not None:
            retriever.close()
        if hybrid_retriever is not None:
            hybrid_retriever.close()
        if previous_embedding_alias is not None:
            os.environ["FOUNDRY_EMBEDDING_MODEL_ALIAS"] = previous_embedding_alias


if __name__ == "__main__":
    main()
