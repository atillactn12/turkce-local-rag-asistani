"""Retrieval, Foundry Local üretimi ve güvenli fallback akışı."""

import re

from src.foundry_client import clean_model_answer, is_repetitive_answer


NOT_FOUND_MESSAGE = "Bu bilgi yüklenen dokümanlarda bulunamadı."


def _shorten(text: str, max_length: int = 220) -> str:
    clean = " ".join(text.split())
    sentence = next((item for item in re.split(r"(?<=[.!?])\s+", clean) if len(item) >= 20), clean)
    return sentence if len(sentence) <= max_length else sentence[: max_length - 3].rsplit(" ", 1)[0] + "..."


def create_extractive_fallback_answer(question: str, chunks: list[dict]) -> str:
    """Retrieved chunklardan en fazla beş kısa, tamamen kaynaklı madde üretir."""
    del question
    items: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        item = _shorten(str(chunk.get("text", "")))
        if item and item.casefold() not in seen:
            seen.add(item.casefold())
            items.append(item)
        if len(items) == 5:
            break
    if not items:
        return NOT_FOUND_MESSAGE
    bullets = "\n".join(f"- {item}" for item in items)
    return (
        "Yüklenen dokümanlarda bu soruyla ilişkili bulunan bilgiler şunlardır:\n\n"
        f"{bullets}\n\n"
        "Bu cevap, model üretimi başarısız olduğunda doğrudan doküman parçalarından oluşturulmuştur."
    )


def _model_answer_is_bad(answer: str) -> bool:
    if not isinstance(answer, str) or len(answer.strip()) < 20:
        return True
    lowered = answer.casefold()
    bad_parts = (
        "foundry local cevap üretirken hata oluştu", "<|im_start|>", "<|im_end|>",
        "<think>", "\nsystem\n", "\nuser\n", "\nassistant\n",
    )
    if any(part in lowered for part in bad_parts):
        return True
    if re.search(r"(?im)^\s*(system|user|assistant)\s*:?[\s]*$", answer):
        return True
    return is_repetitive_answer(answer)


def _result(answer: str, sources: list[dict], used_fallback: bool, llm_client) -> dict:
    return {
        "answer": answer,
        "sources": sources,
        "used_fallback": used_fallback,
        "model_alias": getattr(llm_client, "model_alias", "Bilinmiyor"),
    }


def answer_question(question: str, retriever, llm_client, top_k: int = 5, minimum_score: float = 0.03) -> dict:
    """Soruyla ilgili bağlamı bulur ve kaynaklı, güvenli cevap döndürür."""
    if not isinstance(question, str) or not question.strip():
        return _result("Lütfen bir soru yazın.", [], False, llm_client)
    try:
        results = retriever.search(question.strip(), top_k)
    except Exception as error:
        return _result(f"Dokümanlar aranırken hata oluştu: {error}", [], False, llm_client)
    if not results or float(results[0].get("score", 0.0)) < minimum_score:
        return _result(NOT_FOUND_MESSAGE, [], False, llm_client)
    chunks = [chunk for chunk in results if float(chunk.get("score", 0.0)) >= minimum_score]
    if not chunks:
        return _result(NOT_FOUND_MESSAGE, [], False, llm_client)

    context_parts: list[str] = []
    sources: list[dict] = []
    for order, chunk in enumerate(chunks, start=1):
        page = chunk.get("page_number")
        page_text = page if page is not None else "Belirtilmemiş"
        text = str(chunk.get("text", "")).strip()
        context_parts.append(
            f"[KAYNAK {order}]\nDosya: {chunk.get('file_name')}\nSayfa: {page_text}\nİçerik:\n{text}"
        )
        preview = " ".join(text.split())
        sources.append({
            "chunk_id": chunk.get("chunk_id"), "file_name": chunk.get("file_name"),
            "page_number": page, "score": float(chunk.get("score", 0.0)),
            "preview": preview[:177].rstrip() + "..." if len(preview) > 180 else preview,
        })

    system_prompt = f"""Sen Türkçe cevap veren bir RAG doküman asistanısın.
Sadece verilen BAĞLAM içindeki bilgilere dayanarak cevap ver.
BAĞLAM dışında bilgi ekleme. Tahmin yapma. Uydurma bilgi üretme.
Eğer cevap bağlamda açıkça yoksa sadece şunu yaz:
"{NOT_FOUND_MESSAGE}"
Cevabı kısa, net ve öğrenci dostu Türkçe ile ver.
Cevabın sonunda kullandığın kaynakları dosya adı ve sayfa numarasıyla belirt."""
    user_prompt = f"Kullanıcı sorusu:\n{question.strip()}\n\nBAĞLAM:\n" + "\n\n".join(context_parts)
    try:
        raw_answer = llm_client.generate_answer(system_prompt, user_prompt)
        answer_bad_before_cleaning = _model_answer_is_bad(raw_answer)
        answer = clean_model_answer(raw_answer)
    except Exception:
        answer_bad_before_cleaning = True
        answer = ""
    used_fallback = (
        answer_bad_before_cleaning
        or getattr(llm_client, "last_answer_had_artifacts", False)
        or _model_answer_is_bad(answer)
    )
    if used_fallback:
        answer = create_extractive_fallback_answer(question, chunks)
    return _result(answer, sources, used_fallback, llm_client)
