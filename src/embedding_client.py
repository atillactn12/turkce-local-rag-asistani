"""TF-IDF ve isteğe bağlı Foundry Local embedding istemcisi."""

import os

from sklearn.feature_extraction.text import TfidfVectorizer


class LocalEmbeddingClient:
    """Yerel metin vektörleri üretir.

    Güvenli varsayılan sağlayıcı TF-IDF'dir. Foundry sağlayıcısı yalnızca
    ``FOUNDRY_EMBEDDING_MODEL_ALIAS`` ayarlanmışsa ve belirtilen model daha
    önce yerel önbelleğe indirilmişse kullanılır. Bu sınıf model indirmez.
    """

    SUPPORTED_PROVIDERS = {"tfidf_fallback", "foundry_embedding", "hybrid"}

    def __init__(
        self,
        provider: str = "tfidf_fallback",
        model_alias: str | None = None,
    ) -> None:
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Desteklenmeyen embedding sağlayıcısı: {provider}")

        self.provider = provider
        self.model_alias = (
            model_alias
            if model_alias is not None
            else os.getenv("FOUNDRY_EMBEDDING_MODEL_ALIAS", "")
        ).strip()
        self.last_error = ""
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            max_features=2048,
            sublinear_tf=True,
        )
        self._is_fitted = False
        self._foundry_model = None
        self._foundry_client = None
        self._loaded_here = False
        self._availability_checked = False

    def get_provider_name(self) -> str:
        """SQLite kayıtlarında kullanılacak sabit sağlayıcı adını döndürür."""
        if self.provider == "tfidf_fallback":
            return "tfidf_fallback"
        return "foundry_embedding"

    def _initialize_foundry(self) -> bool:
        """Yalnızca önbellekteki bir embedding modeline güvenle bağlanır."""
        if self._foundry_client is not None:
            return True
        if self._availability_checked:
            return False
        self._availability_checked = True

        if not self.model_alias:
            self.last_error = (
                "FOUNDRY_EMBEDDING_MODEL_ALIAS ayarlı değil; TF-IDF kullanılacak."
            )
            return False

        try:
            from foundry_local_sdk import FoundryLocalManager

            manager = FoundryLocalManager.instance
            if manager is None:
                self.last_error = (
                    "Foundry Local yöneticisi henüz başlatılmadı; mevcut LLM "
                    "yaşam döngüsünü korumak için TF-IDF kullanılacak."
                )
                return False

            # Güvenlik: katalogdaki modeli indirmek yerine yalnızca daha önce
            # indirilmiş yerel modeller arasında arama yaparız.
            cached_models = manager.catalog.get_cached_models()
            model = next(
                (
                    item
                    for item in cached_models
                    if item.alias == self.model_alias or item.id == self.model_alias
                ),
                None,
            )
            if model is None:
                self.last_error = (
                    "Foundry embedding modeli yerel önbellekte bulunamadı; "
                    "otomatik indirme yapılmadı."
                )
                return False

            info = getattr(model, "info", None)
            metadata = " ".join(
                str(value or "")
                for value in (
                    getattr(info, "task", ""),
                    getattr(info, "model_type", ""),
                    getattr(info, "capabilities", ""),
                )
            ).casefold()
            if "embed" not in metadata:
                self.last_error = (
                    "Seçilen Foundry modeli embedding modeli olarak tanımlanmamış."
                )
                return False

            if not model.is_loaded:
                model.load()
                self._loaded_here = True
            self._foundry_model = model
            self._foundry_client = model.get_embedding_client()
            self.last_error = ""
            return True
        except Exception as error:
            self.last_error = f"Foundry embedding hazırlanamadı: {error}"
            self._foundry_client = None
            return False

    @staticmethod
    def _vectors_from_response(response) -> list[list[float]]:
        """OpenAI uyumlu Foundry cevabından sıralı vektörleri çıkarır."""
        data = list(getattr(response, "data", []) or [])
        data.sort(key=lambda item: int(getattr(item, "index", 0)))
        vectors = [
            list(getattr(item, "embedding", []) or [])
            for item in data
        ]
        if not vectors or any(not vector for vector in vectors):
            raise RuntimeError("Foundry embedding cevabında vektör bulunamadı.")
        return vectors

    def fit(self, texts: list[str]) -> None:
        """TF-IDF sözlüğünü kurar; Foundry sağlayıcısında ek eğitim gerekmez."""
        if not texts:
            raise ValueError("Embedding oluşturmak için metin bulunamadı.")
        if self.provider == "tfidf_fallback":
            self.vectorizer.fit(texts)
            self._is_fitted = True
            return
        if not self._initialize_foundry():
            raise RuntimeError(self.last_error or "Foundry embedding kullanılamıyor.")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Doküman metinlerini seçilen sağlayıcıyla vektörleştirir."""
        if not texts:
            return []
        if self.provider == "tfidf_fallback":
            if not self._is_fitted:
                self.fit(texts)
            return self.vectorizer.transform(texts).toarray().tolist()

        if not self._initialize_foundry():
            raise RuntimeError(self.last_error or "Foundry embedding kullanılamıyor.")
        response = self._foundry_client.generate_embeddings(texts)
        return self._vectors_from_response(response)

    def embed_query(self, query: str) -> list[float]:
        """Tek bir sorgu için dokümanlarla aynı türde vektör üretir."""
        if not query.strip():
            raise ValueError("Embedding sorgusu boş olamaz.")
        if self.provider == "tfidf_fallback":
            if not self._is_fitted:
                raise RuntimeError("Önce doküman embeddingleri oluşturulmalıdır.")
            return self.vectorizer.transform([query]).toarray()[0].tolist()

        if not self._initialize_foundry():
            raise RuntimeError(self.last_error or "Foundry embedding kullanılamıyor.")
        response = self._foundry_client.generate_embedding(query)
        return self._vectors_from_response(response)[0]

    def is_available(self) -> bool:
        """Sağlayıcının mevcut ortamda güvenle kullanılabilir olup olmadığını bildirir."""
        if self.provider == "tfidf_fallback":
            return True
        return self._initialize_foundry()

    def close(self) -> None:
        """Bu istemcinin yüklediği Foundry modelini güvenle serbest bırakır."""
        try:
            if self._loaded_here and self._foundry_model is not None:
                self._foundry_model.unload()
        except Exception:
            pass
        finally:
            self._foundry_client = None
            self._foundry_model = None
            self._loaded_here = False
