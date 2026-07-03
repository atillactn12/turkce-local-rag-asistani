"""TF-IDF ve cosine similarity tabanlı basit retrieval bileşeni."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleRetriever:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), sublinear_tf=True)
        self.chunks: list[dict] = []
        self.document_vectors = None
        self._index_is_built = False

    def build_index(self, chunks: list[dict]) -> None:
        """Boş metinleri atlayıp TF-IDF indeksini oluşturur."""
        self.chunks = [chunk for chunk in chunks if isinstance(chunk.get("text"), str) and chunk["text"].strip()]
        self.document_vectors = None
        self._index_is_built = True
        if not self.chunks:
            return
        texts = [f"{chunk.get('file_name', '')} {chunk['text']}" for chunk in self.chunks]
        try:
            self.document_vectors = self.vectorizer.fit_transform(texts)
        except ValueError:
            self.chunks = []

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Sorguyla en alakalı, sıfırdan yüksek skorlu chunkları döndürür."""
        if not self._index_is_built:
            raise RuntimeError("Arama indeksi oluşturulmadı. Önce build_index çağrılmalıdır.")
        if not self.chunks or self.document_vectors is None or not query.strip() or top_k <= 0:
            return []
        similarities = cosine_similarity(self.vectorizer.transform([query]), self.document_vectors).flatten()
        indexes = sorted(
            (index for index in range(len(self.chunks)) if similarities[index] > 0),
            key=lambda index: similarities[index],
            reverse=True,
        )[:top_k]
        return [
            {
                "chunk_id": self.chunks[index].get("chunk_id"),
                "file_name": self.chunks[index].get("file_name"),
                "page_number": self.chunks[index].get("page_number"),
                "text": self.chunks[index].get("text", ""),
                "score": float(similarities[index]),
            }
            for index in indexes
        ]
