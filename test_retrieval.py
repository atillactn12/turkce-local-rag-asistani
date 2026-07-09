"""LLM indirmeden loader, chunker ve retriever akışını test eder."""

from pathlib import Path

from src.chunker import split_documents_into_chunks
from src.document_loader import load_documents_from_folder
from src.rag_pipeline import NOT_FOUND_MESSAGE, answer_question
from src.retriever import SimpleRetriever
from src.vector_store import SQLiteVectorStore


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
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"


class NoInferenceLLM:
    """Kapsam dışı sorunun modele ulaşmadığını doğrulayan küçük test nesnesi."""

    model_alias = "test-no-inference"

    def generate_answer(self, _system_prompt: str, _user_prompt: str) -> str:
        raise AssertionError("Kapsam dışı soru LLM'e gönderilmemelidir.")


def main() -> None:
    documents_folder = Path(__file__).resolve().parent / "data" / "documents"
    database_path = Path(__file__).resolve().parent / "data" / "index" / "rag_index.sqlite"
    documents = load_documents_from_folder(str(documents_folder))
    if not documents:
        raise RuntimeError("Retrieval testi için doküman bulunamadı.")
    chunks = split_documents_into_chunks(documents)
    if not chunks:
        raise RuntimeError("Retrieval testi için chunk oluşturulamadı.")

    retriever = None
    foundry_retriever = None
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

        out_of_scope = answer_question(
            "Projenin bütçesi kaç TL'dir?",
            retriever,
            NoInferenceLLM(),
        )
        if out_of_scope.get("answer") != NOT_FOUND_MESSAGE:
            raise RuntimeError("Kapsam dışı soru beklenen cevabı döndürmedi.")
        print("Kapsam dışı soru: güvenli şekilde dokümanda bulunamadı.")

        foundry_retriever = SimpleRetriever(
            db_path=str(database_path),
            retrieval_mode="foundry_embedding",
            embedding_model_alias=EMBEDDING_MODEL_ALIAS,
        )
        foundry_retriever.build_index(chunks)
        foundry_results = foundry_retriever.search(QUESTIONS[0][0], top_k=3)
        if not foundry_results:
            raise RuntimeError("Foundry Embedding modu sonuç döndürmedi.")
        if not foundry_retriever.embedding_available:
            if not foundry_retriever.fell_back_to_tfidf:
                raise RuntimeError("Foundry modu TF-IDF yedeğine dönmedi.")
            print(
                "Foundry Embedding: kullanılamadı; TF-IDF yedeği çalıştı. "
                f"Durum: {foundry_retriever.embedding_status_message}"
            )
        else:
            if foundry_retriever.active_backend != "foundry_embedding":
                raise RuntimeError("Foundry embedding backend'i etkinleşmedi.")
            print("Foundry Embedding: gerçek embedding araması çalıştı.")

        hybrid_retriever = SimpleRetriever(
            db_path=str(database_path),
            retrieval_mode="hybrid",
            embedding_model_alias=EMBEDDING_MODEL_ALIAS,
        )
        hybrid_retriever.build_index(chunks)
        hybrid_results = hybrid_retriever.search(QUESTIONS[0][0], top_k=3)
        if not hybrid_results:
            raise RuntimeError("Hybrid mod TF-IDF yedeğiyle sonuç döndürmedi.")
        if hybrid_retriever.embedding_available:
            if hybrid_retriever.active_backend != "hybrid":
                raise RuntimeError("Hybrid backend etkinleşmedi.")
            print("Hybrid arama: TF-IDF ve Foundry embedding birlikte çalıştı.")
        else:
            if not hybrid_retriever.fell_back_to_tfidf:
                raise RuntimeError("Hybrid mod TF-IDF yedeğine dönmedi.")
            print(
                "Hybrid arama: embedding kullanılamadığında "
                f"{hybrid_retriever.last_search_backend} ile çalıştı."
            )

        vector_store = SQLiteVectorStore(str(database_path))
        try:
            vector_store.initialize()
            providers = {
                provider: len(vector_store.get_embeddings(provider))
                for provider in ("tfidf_fallback", "foundry_embedding")
            }
        finally:
            vector_store.close()
        print("SQLite sağlayıcı sayıları:")
        for provider, count in providers.items():
            if count:
                print(f"- {provider}: {count}")
        if providers["tfidf_fallback"] != len(chunks):
            raise RuntimeError("SQLite TF-IDF vektör sayısı chunk sayısıyla eşleşmiyor.")
        if hybrid_retriever.embedding_available and providers["foundry_embedding"] != len(chunks):
            raise RuntimeError("SQLite Foundry vektör sayısı chunk sayısıyla eşleşmiyor.")
    finally:
        if retriever is not None:
            retriever.close()
        if foundry_retriever is not None:
            foundry_retriever.close()
        if hybrid_retriever is not None:
            hybrid_retriever.close()


if __name__ == "__main__":
    main()
