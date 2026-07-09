"""Microsoft Foundry Local SDK için küçük ve güvenli istemci wrapper'ı."""

import os
import re


DEFAULT_MODEL_ALIAS = os.getenv("FOUNDRY_MODEL_ALIAS", "qwen2.5-0.5b").strip() or "qwen2.5-0.5b"


def clean_model_answer(text: str) -> str:
    """Chat template işaretlerini ve gereksiz boşlukları temizler."""
    if not isinstance(text, str):
        return ""
    cleaned = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    cleaned = re.sub(r"(?im)^\s*(system|user|assistant)\s*:?[\s]*$", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def is_repetitive_answer(text: str) -> bool:
    """Belirgin kelime, cümle veya kelime grubu tekrarlarını tespit eder."""
    cleaned = clean_model_answer(text).casefold()
    words = re.findall(r"\w+", cleaned)
    if len(words) < 8:
        return False
    for index in range(len(words) - 2):
        if len(set(words[index : index + 3])) == 1:
            return True
    for size in range(4, 11):
        groups = [tuple(words[index : index + size]) for index in range(len(words) - size + 1)]
        if len(groups) != len(set(groups)):
            return True
    sentences = [item.strip() for item in re.split(r"[.!?]+", cleaned) if item.strip()]
    return len(sentences) != len(set(sentences))


class FoundryLLMClient:
    """Foundry Local model yaşam döngüsünü ve cevap üretimini yönetir."""

    def __init__(self, model_alias: str | None = None) -> None:
        self.model_alias = (model_alias or DEFAULT_MODEL_ALIAS).strip()
        self.manager = None
        self.model = None
        self.client = None
        self.last_error = ""
        self.last_answer_had_artifacts = False

    def initialize(self) -> None:
        if self.client is not None:
            return
        try:
            from foundry_local_sdk import Configuration, FoundryLocalManager

            # Embedding bileşeni SDK yöneticisini daha önce başlatmış olabilir.
            # Singleton mevcutsa onu yeniden kullanmak LLM yaşam döngüsünü korur.
            if FoundryLocalManager.instance is None:
                FoundryLocalManager.initialize(
                    Configuration(app_name="turkce_local_rag_asistani")
                )
            self.manager = FoundryLocalManager.instance
            self.manager.download_and_register_eps()
            self.model = self.manager.catalog.get_model(self.model_alias)
            if self.model is None:
                raise ValueError(f"Model bulunamadı: {self.model_alias}")
            self.model.download()
            self.model.load()
            self.client = self.model.get_chat_client()
            try:
                self.client.settings.temperature = 0.1
                self.client.settings.max_tokens = 500
            except Exception:
                pass
        except Exception as error:
            self.client = None
            self.last_error = f"Foundry Local cevap üretirken hata oluştu: {error}"

    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        if self.client is None:
            self.initialize()
        if self.client is None:
            return self.last_error or "Foundry Local cevap üretirken hata oluştu: İstemci hazırlanamadı."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            try:
                response = self.client.complete_chat(messages)
            except Exception:
                if hasattr(self.client, "settings"):
                    self.client.settings.temperature = None
                    self.client.settings.max_tokens = None
                response = self.client.complete_chat(messages)
            if not response.choices or not response.choices[0].message.content:
                return ""
            raw_answer = str(response.choices[0].message.content)
            self.last_answer_had_artifacts = bool(
                "<|im_start|>" in raw_answer
                or "<|im_end|>" in raw_answer
                or re.search(r"(?im)^\s*(system|user|assistant)\s*:?[\s]*$", raw_answer)
            )
            self.last_error = ""
            return clean_model_answer(raw_answer)
        except Exception as error:
            self.last_error = f"Foundry Local cevap üretirken hata oluştu: {error}"
            return self.last_error

    def unload(self) -> None:
        try:
            if self.model is not None and getattr(self.model, "is_loaded", False):
                self.model.unload()
        except Exception:
            pass
        finally:
            self.client = None
