"""SQLite vektör araması ve TF-IDF fallback kullanan basit retriever."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.embedding_client import LocalEmbeddingClient
from src.vector_store import SQLiteVectorStore


META_QUESTION_PHRASES = (
    "rag testi için önerilen sorular",
    "bu dokümanı rag uygulamasında test etmek için",
    "dokümanda bulunmayan bilgiyi test etmek için",
    "önerilen sorular",
    "bu soruların doğru cevabı",
)


def is_meta_question_chunk(text: str) -> bool:
    """RAG test sorularını içeren meta chunkları tanır."""
    normalized = " ".join(str(text).casefold().split())
    if any(phrase in normalized for phrase in META_QUESTION_PHRASES):
        return True

    # Tek başına başlık olarak kullanılan SORULAR ifadesini de meta kabul et.
    return any(line.strip().casefold() == "sorular" for line in str(text).splitlines())


class SimpleRetriever:
    """Önce SQLite'taki vektörleri, gerekirse bellek içi TF-IDF'i arar."""

    def __init__(
        self,
        use_sqlite: bool = True,
        db_path: str = "data/index/rag_index.sqlite",
    ) -> None:
        self.use_sqlite = use_sqlite
        self.db_path = db_path
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.chunks: list[dict] = []
        self.document_vectors = None
        self.vector_store: SQLiteVectorStore | None = None
        self.embedding_client: LocalEmbeddingClient | None = None
        self.sqlite_ready = False
        self.last_search_backend = "tfidf"
        self._index_is_built = False

    @staticmethod
    def _searchable_texts(chunks: list[dict]) -> list[str]:
        """Dosya adı ile chunk metnini aynı arama metninde birleştirir."""
        return [f"{chunk.get('file_name', '')} {chunk['text']}" for chunk in chunks]

    def build_index(self, chunks: list[dict]) -> None:
        """TF-IDF indeksini ve mümkünse yerel SQLite vektör indeksini kurar."""
        self.chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk.get("text"), str) and chunk["text"].strip()
        ]
        self.document_vectors = None
        self.sqlite_ready = False
        self.last_search_backend = "tfidf"
        self._index_is_built = True
        if not self.chunks:
            return

        texts = self._searchable_texts(self.chunks)

        # Bellek içi TF-IDF her zaman hazır tutulur. SQLite kullanılamazsa
        # uygulama otomatik olarak bu indeksten arama yapmaya devam eder.
        try:
            self.document_vectors = self.vectorizer.fit_transform(texts)
        except ValueError:
            self.chunks = []
            return

        if not self.use_sqlite:
            return

        try:
            self.vector_store = SQLiteVectorStore(self.db_path)
            self.vector_store.initialize()
            self.vector_store.clear()
            self.vector_store.add_chunks(self.chunks)

            self.embedding_client = LocalEmbeddingClient()
            vectors = self.embedding_client.embed_texts(texts)
            for chunk, vector in zip(self.chunks, vectors):
                self.vector_store.save_embedding(
                    str(chunk.get("chunk_id")),
                    vector,
                    self.embedding_client.provider,
                )
            self.sqlite_ready = True
        except Exception as error:
            # SQLite katmanı ek bir özelliktir; hata halinde çalışan TF-IDF
            # indeksini kaybetmeyiz ve kullanıcı arama yapmaya devam eder.
            self.sqlite_ready = False
            print(f"SQLite vektör indeksi hazırlanamadı; TF-IDF kullanılacak: {error}")

    @staticmethod
    def _rank_results(chunks: list[dict], scores, top_k: int) -> list[dict]:
        """Skorları sıralar ve dışarıya verilen ortak sonuç biçimini oluşturur."""
        adjusted_scores = [float(score) for score in scores]

        # Test soruları gerçek açıklayıcı içerik değildir. Tamamen silmek yerine
        # skorlarını düşürerek asıl doküman bölümlerinin önce gelmesini sağlarız.
        for index, chunk in enumerate(chunks):
            if is_meta_question_chunk(chunk.get("text", "")):
                adjusted_scores[index] *= 0.10

        indexes = sorted(
            (index for index in range(len(chunks)) if adjusted_scores[index] > 0),
            key=lambda index: adjusted_scores[index],
            reverse=True,
        )[:top_k]
        return [
            {
                "chunk_id": chunks[index].get("chunk_id"),
                "file_name": chunks[index].get("file_name"),
                "page_number": chunks[index].get("page_number"),
                "text": chunks[index].get("text", ""),
                "score": adjusted_scores[index],
            }
            for index in indexes
        ]

    def _search_sqlite(self, query: str, top_k: int) -> list[dict]:
        """SQLite'ta saklanan vektörler üzerinde cosine similarity uygular."""
        if not self.vector_store or not self.embedding_client:
            return []

        stored_chunks = self.vector_store.get_all_chunks()
        stored_embeddings = self.vector_store.get_embeddings(
            self.embedding_client.provider
        )
        chunk_by_id = {str(chunk.get("chunk_id")): chunk for chunk in stored_chunks}

        # Chunk ve embeddingleri chunk_id üzerinden eşleştiririz; böylece SQLite
        # satırlarının geliş sırasına bağımlı kalmayız.
        matched_chunks: list[dict] = []
        vectors: list[list[float]] = []
        for item in stored_embeddings:
            chunk = chunk_by_id.get(str(item.get("chunk_id")))
            vector = item.get("vector")
            if chunk is not None and isinstance(vector, list) and vector:
                matched_chunks.append(chunk)
                vectors.append(vector)

        if not matched_chunks:
            return []

        query_vector = self.embedding_client.embed_query(query)
        similarities = cosine_similarity([query_vector], vectors).flatten()
        return self._rank_results(matched_chunks, similarities, top_k)

    def _search_tfidf(self, query: str, top_k: int) -> list[dict]:
        """Mevcut bellek içi TF-IDF indeksinde arama yapar."""
        if self.document_vectors is None:
            return []
        similarities = cosine_similarity(
            self.vectorizer.transform([query]), self.document_vectors
        ).flatten()
        return self._rank_results(self.chunks, similarities, top_k)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """En alakalı, sıfırdan yüksek skorlu chunkları döndürür."""
        if not self._index_is_built:
            raise RuntimeError("Arama indeksi oluşturulmadı. Önce build_index çağrılmalıdır.")
        if not self.chunks or not query.strip() or top_k <= 0:
            return []

        if self.sqlite_ready:
            try:
                results = self._search_sqlite(query, top_k)
                if results:
                    self.last_search_backend = "sqlite_vector"
                    return results
            except Exception as error:
                print(f"SQLite vektör araması başarısız; TF-IDF kullanılacak: {error}")

        self.last_search_backend = "tfidf"
        return self._search_tfidf(query, top_k)

    def close(self) -> None:
        """Açık SQLite bağlantısını güvenli biçimde kapatır."""
        if self.vector_store is not None:
            self.vector_store.close()
