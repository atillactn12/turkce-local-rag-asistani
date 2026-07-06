"""TF-IDF ve cosine similarity tabanlı basit retrieval bileşeni."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

        # Test soruları gerçek içerik değildir. Tamamen silmek yerine skorlarını
        # düşürerek açıklayıcı bölümlerin önce gelmesini sağlarız.
        adjusted_scores = similarities.copy()
        for index, chunk in enumerate(self.chunks):
            if is_meta_question_chunk(chunk.get("text", "")):
                adjusted_scores[index] *= 0.10

        indexes = sorted(
            (index for index in range(len(self.chunks)) if adjusted_scores[index] > 0),
            key=lambda index: adjusted_scores[index],
            reverse=True,
        )[:top_k]
        return [
            {
                "chunk_id": self.chunks[index].get("chunk_id"),
                "file_name": self.chunks[index].get("file_name"),
                "page_number": self.chunks[index].get("page_number"),
                "text": self.chunks[index].get("text", ""),
                "score": float(adjusted_scores[index]),
            }
            for index in indexes
        ]
