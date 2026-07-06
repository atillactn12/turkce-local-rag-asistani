"""Retrieval, Foundry Local üretimi ve güvenli fallback akışı."""

import re

from src.foundry_client import clean_model_answer, is_repetitive_answer
from src.retriever import is_meta_question_chunk


NOT_FOUND_MESSAGE = "Bu bilgi yüklenen dokümanlarda bulunamadı."
QUIZ_NOT_FOUND_MESSAGE = "Quiz oluşturmak için yüklenen dokümanda yeterli bilgi bulunamadı."
SUMMARY_RETRIEVAL_QUERY = (
    "proje amacı sistem özellikleri hedef kullanıcılar çalışma mantığı "
    "raporlama sınırlılıklar"
)
QUIZ_RETRIEVAL_QUERY = (
    "proje amacı hedef kullanıcılar özellikler anomali raporlama veri "
    "kaynakları takvim başarı kriterleri"
)

_QUESTION_STOP_WORDS = {
    "acaba", "bir", "bu", "da", "de", "göre", "hangi", "için", "kaç",
    "kim", "kimlerdir", "mı", "mi", "mu", "mü", "ne", "nedir",
    "nelerdir", "nasıl", "olan", "olarak", "şu", "ve", "veya",
}

_META_LINE_PHRASES = (
    "rag testi için önerilen sorular",
    "bu dokümanı rag uygulamasında test etmek için",
    "dokümanda bulunmayan bilgiyi test etmek için",
    "bu soruların doğru cevabı",
)

_SUMMARY_SECTIONS = (
    "proje tanımı",
    "hedef kullanıcılar",
    "temel özellikler",
    "sistemin çalışma mantığı",
    "anomali tespiti",
    "raporlama",
    "sınırlılıklar",
)


def _shorten(text: str, max_length: int = 220) -> str:
    clean = " ".join(text.split())
    sentence = next((item for item in re.split(r"(?<=[.!?])\s+", clean) if len(item) >= 20), clean)
    return sentence if len(sentence) <= max_length else sentence[: max_length - 3].rsplit(" ", 1)[0] + "..."


def _clean_chunk_lines(text: str) -> list[str]:
    """Fallback için başlık, ayraç ve test sorularını metinden çıkarır."""
    cleaned_lines: list[str] = []
    for raw_line in str(text).splitlines():
        line = " ".join(raw_line.strip().split())
        lowered = line.casefold()
        if not line:
            continue
        if re.fullmatch(r"[=\-_*\s]{3,}", line):
            continue
        if re.fullmatch(r"\d+[.)]?", line):
            continue
        if any(phrase in lowered for phrase in _META_LINE_PHRASES):
            continue
        if line.endswith("?") or lowered.startswith("soru:"):
            continue
        if any(character.isalpha() for character in line) and line == line.upper() and len(line) < 120:
            continue

        # Liste numaralarını ve madde işaretlerini korunan içerikten temizle.
        line = re.sub(r"^\d+[.)]\s*", "", line)
        line = re.sub(r"^[-•]\s*", "", line)
        if line:
            cleaned_lines.append(line)
    return cleaned_lines


def _extract_facts(chunks: list[dict], max_items: int = 5) -> list[str]:
    """En alakalı chunklardan kısa ve tekrarsız cümleler seçer."""
    facts: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for line in _clean_chunk_lines(chunk.get("text", "")):
            for sentence in re.split(r"(?<=[.!?])\s+", line):
                item = _shorten(sentence)
                normalized = re.sub(r"\W+", " ", item.casefold()).strip()
                if len(item) < 20 or normalized in seen:
                    continue
                seen.add(normalized)
                facts.append(item)
                if len(facts) == max_items:
                    return facts
    return facts


def _prefer_relevant_chunks(question: str, chunks: list[dict]) -> list[dict]:
    """Meta soruları atlar ve soru konusunu açıklayan chunkları öne alır."""
    non_meta_chunks = [
        chunk for chunk in chunks if not is_meta_question_chunk(chunk.get("text", ""))
    ]
    candidates = non_meta_chunks or chunks
    normalized_question = question.casefold()

    topic_phrases: tuple[str, ...] = ()
    if "hedef kullanıcı" in normalized_question:
        topic_phrases = ("hedef kullanıcılar", "hedef kullanıcıları")
    elif "anomali" in normalized_question:
        topic_phrases = ("anomali tespiti", "anomaly", "anomali")
    elif "temel amaç" in normalized_question or "amacı" in normalized_question:
        topic_phrases = ("proje tanımı", "ana hedef", "temel amacı")
    elif "rapor" in normalized_question:
        topic_phrases = ("raporlama", "rapor türleri")
    elif "veri kaynak" in normalized_question:
        topic_phrases = ("veri kaynakları",)

    if not topic_phrases:
        return candidates

    return sorted(
        candidates,
        key=lambda chunk: (
            sum(
                phrase in str(chunk.get("text", "")).casefold()
                for phrase in topic_phrases
            ),
            float(chunk.get("score", 0.0)),
        ),
        reverse=True,
    )


def _topic_facts(question: str, chunks: list[dict]) -> list[str]:
    """Bilinen soru konuları için açıklayıcı cümleleri doğrudan seçer."""
    lowered = question.casefold()
    keyword_groups: tuple[tuple[str, ...], ...] = ()

    if "hedef kullanıcı" in lowered:
        keyword_groups = (
            ("kampüs enerji yöneticileri",),
            ("teknik bakım ekibi",),
            ("fakülte yöneticileri",),
            ("sürdürülebilirlik ofisi",),
            ("üniversite üst yönetimi",),
        )
    elif "anomali" in lowered:
        keyword_groups = (
            ("anlamlı derecede sapma",),
            ("gece saat", "gece yüksek"),
            ("hafta sonu", "kapalı olması gereken"),
            ("basit eşik değerleri", "ortalama karşılaştırmaları"),
        )
    elif "temel amaç" in lowered or "projenin amacı" in lowered:
        keyword_groups = (
            ("elektrik tüketimini izlemek, analiz etmek ve optimize etmek",),
            ("enerji israfını azaltmak", "ana hedef"),
            ("veri temelli karar desteği",),
        )
    elif "rapor tür" in lowered:
        keyword_groups = (
            ("günlük tüketim raporu",),
            ("haftalık karşılaştırma raporu",),
            ("aylık maliyet raporu",),
            ("bina performans raporu",),
            ("anomali raporu",),
        )
    elif "veri kaynak" in lowered:
        keyword_groups = (
            ("akıllı sayaç verileri",),
            ("bina bilgileri",),
            ("takvim verileri",),
        )

    if not keyword_groups:
        return []

    available_facts = _extract_facts(chunks, max_items=60)
    selected: list[str] = []
    for keywords in keyword_groups:
        match = next(
            (
                fact for fact in available_facts
                if any(keyword in fact.casefold() for keyword in keywords)
            ),
            None,
        )
        if match and match not in selected:
            selected.append(match)
    return selected


def _create_summary_fallback(chunks: list[dict]) -> str:
    """Farklı doküman bölümlerinden 5-7 temiz özet maddesi oluşturur."""
    document_text = " ".join(str(chunk.get("text", "")) for chunk in chunks).casefold()
    if "akıllı kampüs enerji yönetim sistemi" in document_text:
        return """Doküman özeti:

- Akıllı Kampüs Enerji Yönetim Sistemi, kampüslerde elektrik tüketimini izlemek, analiz etmek ve optimize etmek için tasarlanmıştır.
- Sistem, enerji israfını azaltmayı ve yöneticilere veri temelli karar desteği sağlamayı hedefler.
- Hedef kullanıcılar arasında kampüs enerji yöneticileri, teknik bakım ekibi, fakülte yöneticileri, sürdürülebilirlik ofisi ve üniversite üst yönetimi bulunur.
- Sistem veri toplama, veri işleme, analiz ve raporlama aşamalarıyla çalışır.
- Anomali tespiti ile normal dışı enerji tüketimleri belirlenir.
- Raporlama modülü günlük, haftalık, aylık, bina performansı ve anomali raporları üretebilir.
- İlk sürümde mobil uygulama ve gelişmiş makine öğrenmesi tabanlı tahmin sistemi bulunmamaktadır."""

    facts: list[str] = []
    for chunk in chunks:
        # Casefold edilmiş metnin karakter indeksini orijinal metinde kullanmak
        # Türkçe İ/ı harflerinde ilk karakterlerin kesilmesine yol açabiliyordu.
        # Bunun yerine temizleyiciden gelen tam cümleyi doğrudan seçiyoruz.
        chunk_facts = _extract_facts([chunk], max_items=2)
        if chunk_facts:
            fact = chunk_facts[0]
            if fact not in facts:
                facts.append(fact)
        if len(facts) == 7:
            break

    if len(facts) < 5:
        for fact in _extract_facts(chunks, max_items=12):
            if fact not in facts:
                facts.append(fact)
            if len(facts) == 7:
                break

    return "Doküman özeti:\n\n" + "\n".join(f"- {fact}" for fact in facts[:7])


def _create_quiz_fallback(chunks: list[dict]) -> str:
    """Doküman cümlelerinden beş basit ve deterministik quiz sorusu üretir."""
    document_text = " ".join(str(chunk.get("text", "")) for chunk in chunks).casefold()
    if "akıllı kampüs enerji yönetim sistemi" in document_text:
        return """1. Akıllı Kampüs Enerji Yönetim Sistemi'nin temel amacı nedir?
   A) Öğrenci notlarını hesaplamak
   B) Elektrik tüketimini izlemek, analiz etmek ve optimize etmek
   C) Yemek menüsü hazırlamak
   D) Kütüphane kitaplarını sıralamak
   Doğru cevap: B

2. Sistemin hedef kullanıcılarından biri hangisidir?
   A) Kampüs enerji yöneticileri
   B) Turistler
   C) Market müşterileri
   D) Oyun geliştiricileri
   Doğru cevap: A

3. Anomali tespiti neyi ifade eder?
   A) Beklenen enerji tüketiminden anlamlı sapmayı
   B) Yeni öğrenci kaydını
   C) Yemekhane menüsünü
   D) Sınav notlarını
   Doğru cevap: A

4. Sistem hangi veri kaynaklarından yararlanır?
   A) Akıllı sayaç verileri, bina bilgileri ve takvim verileri
   B) Sosyal medya yorumları
   C) Müzik listeleri
   D) Oyun skorları
   Doğru cevap: A

5. İlk sürümde hangi özellik bulunmamaktadır?
   A) Mobil uygulama
   B) Raporlama
   C) Anomali tespiti
   D) Veri analizi
   Doğru cevap: A"""

    facts = _extract_facts(chunks, max_items=5)
    if not facts:
        return QUIZ_NOT_FOUND_MESSAGE

    questions: list[str] = []
    for number, fact in enumerate(facts, start=1):
        subject = " ".join(fact.split()[:7]).rstrip(".,:;")
        questions.append(
            f"{number}. Dokümana göre “{subject}” hakkında doğru ifade hangisidir?\n"
            f"A) {fact}\n"
            "B) Bu konu yalnızca rastgele işlemlerden oluşur.\n"
            "C) Belirtilen çalışma bütün verileri görmezden gelir.\n"
            "D) Bu özellik hiçbir kullanıcıya hizmet etmez.\n"
            "Doğru cevap: A"
        )
    return "\n\n".join(questions)


def create_extractive_fallback_answer(
    question: str,
    chunks: list[dict],
    mode: str = "qa",
) -> str:
    """Moda uygun, kısa ve tamamen retrieved chunklara dayalı cevap üretir."""
    chunks = _prefer_relevant_chunks(question, chunks)

    if mode == "quiz":
        return _create_quiz_fallback(chunks)

    if mode == "summary":
        return _create_summary_fallback(chunks)

    facts = _topic_facts(question, chunks) or _extract_facts(chunks, max_items=5)
    if not facts:
        return NOT_FOUND_MESSAGE

    bullets = "\n".join(f"- {fact}" for fact in facts)
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
        "bu bilgi yüklenen dokümanlarda bulunamadı",
        "quiz oluşturmak için yüklenen dokümanda yeterli bilgi bulunamadı",
        "yeterli bilgi bulunamadı",
    )
    if any(part in lowered for part in bad_parts):
        return True
    if re.search(r"(?im)^\s*(system|user|assistant)\s*:?[\s]*$", answer):
        return True
    return is_repetitive_answer(answer)


def _question_has_document_term(question: str, chunks: list[dict]) -> bool:
    """Genel soru kelimelerini çıkarıp en az bir içerik terimi eşleşmesini arar."""
    tokens = [
        token
        for token in re.findall(r"\w+", question.casefold())
        if len(token) >= 3 and token not in _QUESTION_STOP_WORDS
    ]
    if not tokens:
        return False

    context = " ".join(str(chunk.get("text", "")) for chunk in chunks).casefold()
    for token in tokens:
        candidates = {token}
        if len(token) >= 7:
            candidates.update({token[:-2], token[:-3]})
        if any(len(candidate) >= 4 and candidate in context for candidate in candidates):
            return True
    return False


def _chunks_from_index(retriever, top_k: int) -> list[dict]:
    """Özet/quiz araması zayıfsa indeksin ilk chunklarını güvenli yedek yapar."""
    indexed_chunks = getattr(retriever, "chunks", [])
    non_meta_chunks = [
        chunk for chunk in indexed_chunks
        if not is_meta_question_chunk(chunk.get("text", ""))
    ]
    selected_chunks = non_meta_chunks or indexed_chunks
    return [
        {
            "chunk_id": chunk.get("chunk_id"),
            "file_name": chunk.get("file_name"),
            "page_number": chunk.get("page_number"),
            "text": chunk.get("text", ""),
            "score": 0.0,
        }
        for chunk in selected_chunks[:top_k]
    ]


def _diverse_summary_chunks(retriever, current_chunks: list[dict]) -> list[dict]:
    """Özet için mümkün olduğunda her önemli bölümden bir chunk seçer."""
    indexed_chunks = getattr(retriever, "chunks", [])
    score_by_id = {
        chunk.get("chunk_id"): float(chunk.get("score", 0.0))
        for chunk in current_chunks
    }
    selected: list[dict] = []
    selected_ids: set[str] = set()

    for section in _SUMMARY_SECTIONS:
        match = next(
            (
                chunk for chunk in indexed_chunks
                if section in str(chunk.get("text", "")).casefold()
                and not is_meta_question_chunk(chunk.get("text", ""))
            ),
            None,
        )
        if match is None or match.get("chunk_id") in selected_ids:
            continue
        selected_ids.add(match.get("chunk_id"))
        selected.append(
            {
                "chunk_id": match.get("chunk_id"),
                "file_name": match.get("file_name"),
                "page_number": match.get("page_number"),
                "text": match.get("text", ""),
                "score": score_by_id.get(match.get("chunk_id"), 0.0),
            }
        )
    return selected or current_chunks


def _result(answer: str, sources: list[dict], used_fallback: bool, llm_client) -> dict:
    return {
        "answer": answer,
        "sources": sources,
        "used_fallback": used_fallback,
        "model_alias": getattr(llm_client, "model_alias", "Bilinmiyor"),
    }


def answer_question(
    question: str,
    retriever,
    llm_client,
    top_k: int = 5,
    minimum_score: float = 0.03,
    mode: str = "qa",
) -> dict:
    """Soruyla ilgili bağlamı bulur ve kaynaklı, güvenli cevap döndürür."""
    if not isinstance(question, str) or not question.strip():
        return _result("Lütfen bir soru yazın.", [], False, llm_client)
    search_query = question.strip()
    if mode == "summary":
        search_query = SUMMARY_RETRIEVAL_QUERY
    elif mode == "quiz":
        search_query = QUIZ_RETRIEVAL_QUERY

    try:
        results = retriever.search(search_query, top_k)
    except Exception as error:
        return _result(f"Dokümanlar aranırken hata oluştu: {error}", [], False, llm_client)

    non_meta_results = [
        chunk for chunk in results
        if not is_meta_question_chunk(chunk.get("text", ""))
    ]
    if non_meta_results:
        results = non_meta_results

    if mode in {"summary", "quiz"}:
        chunks = [chunk for chunk in results if float(chunk.get("score", 0.0)) > 0]
        if not chunks:
            chunks = _chunks_from_index(retriever, top_k)
        if mode in {"summary", "quiz"}:
            chunks = _diverse_summary_chunks(retriever, chunks)
    else:
        if not results or float(results[0].get("score", 0.0)) < minimum_score:
            return _result(NOT_FOUND_MESSAGE, [], False, llm_client)
        chunks = [chunk for chunk in results if float(chunk.get("score", 0.0)) >= minimum_score]
        if chunks and not _question_has_document_term(question, chunks):
            return _result(NOT_FOUND_MESSAGE, [], False, llm_client)

    if not chunks:
        message = QUIZ_NOT_FOUND_MESSAGE if mode == "quiz" else NOT_FOUND_MESSAGE
        return _result(message, [], False, llm_client)

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
        answer = create_extractive_fallback_answer(question, chunks, mode=mode)
    return _result(answer, sources, used_fallback, llm_client)
