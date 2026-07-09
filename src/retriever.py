"""SQLite, TF-IDF ve isteğe bağlı hibrit arama bileşeni."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.embedding_client import LocalEmbeddingClient
from src.vector_store import SQLiteVectorStore


META_QUESTION_PHRASES = (
    "rag testi için önerilen sorular",
    "bu dokümanı rag uygulamasında test etmek için",
    "dokümanda bulunmayan bilgiyi test etmek için",
    "dokümanda bulunmayan bilgi",
    "cevabı bu dokümanda verilmemiştir",
    "cevabı verilmemiştir",
    "bu tür sorularda",
    "uydurma cevap vermemesi",
    "bulunamadı demesi beklenir",
    "test etmek için sorular",
    "önerilen sorular",
    "bu soruların doğru cevabı",
)
RETRIEVAL_MODES = {"tfidf", "foundry_embedding", "hybrid"}


def is_meta_question_chunk(text: str) -> bool:
    """RAG test sorularını içeren meta chunkları tanır."""
    normalized = " ".join(str(text).casefold().split())
    if any(phrase in normalized for phrase in META_QUESTION_PHRASES):
        return True

    # Tek başına başlık olarak kullanılan SORULAR ifadesini de meta kabul et.
    return any(line.strip().casefold() == "sorular" for line in str(text).splitlines())


class SimpleRetriever:
    """TF-IDF'i daima hazır tutar, isteğe bağlı Foundry sonuçlarıyla birleştirir."""

    def __init__(
        self,
        use_sqlite: bool = True,
        db_path: str = "data/index/rag_index.sqlite",
        retrieval_mode: str = "tfidf",
        embedding_model_alias: str = "qwen3-embedding-0.6b",
    ) -> None:
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"Desteklenmeyen arama yöntemi: {retrieval_mode}")

        self.use_sqlite = use_sqlite
        self.db_path = db_path
        self.retrieval_mode = retrieval_mode
        self.embedding_model_alias = embedding_model_alias.strip()
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
        self.foundry_embedding_client: LocalEmbeddingClient | None = None
        self.sqlite_ready = False
        self.foundry_embedding_ready = False
        self.foundry_embedding_error = ""
        self.embedding_available = False
        self.embedding_status_message = "Embedding henüz hazırlanmadı."
        self.fell_back_to_tfidf = False
        self.last_search_backend = "tfidf"
        self.active_backend = "tfidf"
        self._index_is_built = False

    @staticmethod
    def _searchable_texts(chunks: list[dict]) -> list[str]:
        """Dosya adı ile chunk metnini aynı arama metninde birleştirir."""
        return [f"{chunk.get('file_name', '')} {chunk['text']}" for chunk in chunks]

    def build_index(self, chunks: list[dict]) -> None:
        """TF-IDF indeksini, SQLite kayıtlarını ve varsa Foundry vektörlerini kurar."""
        self.chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk.get("text"), str) and chunk["text"].strip()
        ]
        self.document_vectors = None
        self.sqlite_ready = False
        self.foundry_embedding_ready = False
        self.foundry_embedding_error = ""
        self.embedding_available = False
        self.embedding_status_message = "Embedding henüz hazırlanmadı."
        self.fell_back_to_tfidf = False
        self.last_search_backend = "tfidf"
        self.active_backend = "tfidf"
        self._index_is_built = True
        if not self.chunks:
            return

        texts = self._searchable_texts(self.chunks)

        # Bellek içi TF-IDF her zaman hazır tutulur. Diğer katmanlar başarısız
        # olsa bile uygulama bu indeksle arama yapmaya devam eder.
        try:
            self.document_vectors = self.vectorizer.fit_transform(texts)
        except ValueError:
            self.chunks = []
            return

        if not self.use_sqlite:
            if self.retrieval_mode != "tfidf":
                self.foundry_embedding_error = (
                    "Foundry embedding araması için SQLite etkin olmalıdır."
                )
            return

        try:
            self.vector_store = SQLiteVectorStore(self.db_path)
            self.vector_store.initialize()
            self.vector_store.clear()
            self.vector_store.add_chunks(self.chunks)

            self.embedding_client = LocalEmbeddingClient("tfidf_fallback")
            vectors = self.embedding_client.embed_texts(texts)
            provider = self.embedding_client.get_provider_name()
            for chunk, vector in zip(self.chunks, vectors):
                self.vector_store.save_embedding(
                    str(chunk.get("chunk_id")), vector, provider
                )
            self.sqlite_ready = True
        except Exception as error:
            # SQLite ek bir katmandır; hata halinde bellek içi TF-IDF korunur.
            self.sqlite_ready = False
            print(f"SQLite vektör indeksi hazırlanamadı; TF-IDF kullanılacak: {error}")

        if self.retrieval_mode in {"foundry_embedding", "hybrid"}:
            self._build_foundry_index(texts)

    def _build_foundry_index(self, texts: list[str]) -> None:
        """Önceden indirilmiş Foundry embedding modeli varsa vektörlerini kaydeder."""
        if not self.sqlite_ready or self.vector_store is None:
            self.foundry_embedding_error = (
                "SQLite hazır olmadığı için Foundry embedding indeksi kurulamadı."
            )
            self.embedding_status_message = self.foundry_embedding_error
            return

        client = LocalEmbeddingClient(
            "foundry_embedding",
            embedding_model_alias=self.embedding_model_alias,
        )
        self.foundry_embedding_client = client
        if not client.is_available():
            self.foundry_embedding_error = client.last_error
            self.embedding_status_message = client.get_status_message()
            return

        try:
            vectors = client.embed_texts(texts)
            if len(vectors) != len(self.chunks):
                raise RuntimeError("Foundry embedding sayısı chunk sayısıyla eşleşmiyor.")
            provider = client.get_provider_name()
            for chunk, vector in zip(self.chunks, vectors):
                self.vector_store.save_embedding(
                    str(chunk.get("chunk_id")), vector, provider
                )
            self.foundry_embedding_ready = True
            self.embedding_available = True
            self.foundry_embedding_error = ""
            self.embedding_status_message = client.get_status_message()
            self.active_backend = self.retrieval_mode
        except Exception as error:
            self.foundry_embedding_ready = False
            self.embedding_available = False
            self.foundry_embedding_error = f"Foundry embedding indeksi kurulamadı: {error}"
            self.embedding_status_message = self.foundry_embedding_error

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

    def _search_sqlite_with_client(
        self,
        query: str,
        top_k: int,
        client: LocalEmbeddingClient,
    ) -> list[dict]:
        """SQLite'taki belirli sağlayıcı vektörlerinde kosinüs araması yapar."""
        if not self.vector_store:
            return []

        stored_chunks = self.vector_store.get_all_chunks()
        stored_embeddings = self.vector_store.get_embeddings(
            client.get_provider_name()
        )
        chunk_by_id = {str(chunk.get("chunk_id")): chunk for chunk in stored_chunks}
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

        query_vector = client.embed_query(query)
        similarities = cosine_similarity([query_vector], vectors).flatten()
        return self._rank_results(matched_chunks, similarities, top_k)

    def _search_tfidf_memory(self, query: str, top_k: int) -> list[dict]:
        """Bellek içi TF-IDF indeksinde arama yapar."""
        if self.document_vectors is None:
            return []
        similarities = cosine_similarity(
            self.vectorizer.transform([query]), self.document_vectors
        ).flatten()
        return self._rank_results(self.chunks, similarities, top_k)

    def _search_tfidf(self, query: str, top_k: int) -> tuple[list[dict], str]:
        """Önce SQLite TF-IDF vektörlerini, sonra bellek içi indeksi arar."""
        if self.sqlite_ready and self.embedding_client is not None:
            try:
                results = self._search_sqlite_with_client(
                    query, top_k, self.embedding_client
                )
                if results:
                    return results, "sqlite_vector"
            except Exception as error:
                print(f"SQLite vektör araması başarısız; TF-IDF kullanılacak: {error}")
        return self._search_tfidf_memory(query, top_k), "tfidf"

    def _search_foundry(self, query: str, top_k: int) -> list[dict]:
        """Hazır Foundry embedding vektörlerinde arama yapar."""
        if not self.foundry_embedding_ready or self.foundry_embedding_client is None:
            return []
        try:
            return self._search_sqlite_with_client(
                query, top_k, self.foundry_embedding_client
            )
        except Exception as error:
            self.foundry_embedding_error = f"Foundry embedding araması başarısız: {error}"
            self.embedding_available = False
            self.embedding_status_message = self.foundry_embedding_error
            return []

    @staticmethod
    def _normalized_scores(results: list[dict]) -> dict[str, float]:
        """Bir sonuç listesindeki pozitif skorları 0-1 aralığına getirir."""
        positive_scores = [float(item.get("score", 0.0)) for item in results if float(item.get("score", 0.0)) > 0]
        if not positive_scores:
            return {}
        maximum = max(positive_scores)
        if maximum <= 0:
            return {}
        return {
            str(item.get("chunk_id")): float(item.get("score", 0.0)) / maximum
            for item in results
            if float(item.get("score", 0.0)) > 0
        }

    def _merge_hybrid_results(
        self,
        tfidf_results: list[dict],
        embedding_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """TF-IDF ve embedding sonuçlarını ağırlık ve ortak-sonuç artışıyla birleştirir."""
        if not embedding_results:
            return tfidf_results[:top_k]

        tfidf_scores = self._normalized_scores(tfidf_results)
        embedding_scores = self._normalized_scores(embedding_results)
        result_by_id: dict[str, dict] = {}
        for item in tfidf_results + embedding_results:
            result_by_id.setdefault(str(item.get("chunk_id")), dict(item))

        merged: list[dict] = []
        for chunk_id, item in result_by_id.items():
            tfidf_score = tfidf_scores.get(chunk_id, 0.0)
            embedding_score = embedding_scores.get(chunk_id, 0.0)
            combined_score = 0.55 * embedding_score + 0.45 * tfidf_score
            if tfidf_score > 0 and embedding_score > 0:
                combined_score += 0.10
            if combined_score <= 0:
                continue
            item["score"] = float(combined_score)
            merged.append(item)

        merged.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return merged[:top_k]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Seçili yöntemi kullanır; Foundry başarısızsa TF-IDF'e döner."""
        if not self._index_is_built:
            raise RuntimeError("Arama indeksi oluşturulmadı. Önce build_index çağrılmalıdır.")
        if not self.chunks or not query.strip() or top_k <= 0:
            return []

        self.fell_back_to_tfidf = False
        tfidf_results, tfidf_backend = self._search_tfidf(query, top_k)

        if self.retrieval_mode == "tfidf":
            self.last_search_backend = tfidf_backend
            self.active_backend = "tfidf"
            return tfidf_results

        embedding_results = self._search_foundry(query, top_k)
        if not embedding_results:
            self.fell_back_to_tfidf = True
            self.last_search_backend = "tfidf_fallback"
            self.active_backend = "tfidf"
            return tfidf_results

        if self.retrieval_mode == "foundry_embedding":
            self.last_search_backend = "foundry_embedding"
            self.active_backend = "foundry_embedding"
            return embedding_results

        self.last_search_backend = "hybrid"
        self.active_backend = "hybrid"
        return self._merge_hybrid_results(
            tfidf_results, embedding_results, top_k
        )

    def close(self) -> None:
        """Açık SQLite bağlantısını ve isteğe bağlı Foundry modelini kapatır."""
        if self.vector_store is not None:
            self.vector_store.close()
        if self.foundry_embedding_client is not None:
            self.foundry_embedding_client.close()
