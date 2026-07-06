"""İndirme gerektirmeyen, deterministik yerel embedding istemcisi."""

from sklearn.feature_extraction.text import TfidfVectorizer


class LocalEmbeddingClient:
    """MVP için TF-IDF tabanlı yerel dense vektörler üretir.

    Foundry Local embedding modelleri SDK ve cihaz kataloğuna bağlı olduğundan
    otomatik model indirilmez. Güvenilir varsayılan sağlayıcı TF-IDF'dir.
    """

    def __init__(self, provider: str = "tfidf_fallback") -> None:
        self.provider = provider
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            max_features=2048,
            sublinear_tf=True,
        )
        self._is_fitted = False

    def fit(self, texts: list[str]) -> None:
        if self.provider != "tfidf_fallback":
            raise RuntimeError(f"Embedding sağlayıcısı kullanılamıyor: {self.provider}")
        if not texts:
            raise ValueError("Embedding oluşturmak için metin bulunamadı.")
        self.vectorizer.fit(texts)
        self._is_fitted = True

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self._is_fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, query: str) -> list[float]:
        if not self._is_fitted:
            raise RuntimeError("Önce doküman embeddingleri oluşturulmalıdır.")
        return self.vectorizer.transform([query]).toarray()[0].tolist()

    def is_available(self) -> bool:
        return self.provider == "tfidf_fallback"
