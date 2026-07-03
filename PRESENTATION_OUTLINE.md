# 5 Dakikalık Final Sunumu

## Slayt 1 — 📚 Türkçe Local RAG Doküman Asistanı

- Yerel dokümanlardan Türkçe cevap üretir.
- Python, Streamlit ve Foundry Local kullanır.
- Geliştirici: Atilla Çetin

**Konuşma notu:** Projenin adını ve temel amacını 20 saniyede tanıt.

## Slayt 2 — Problem

- Uzun dokümanlarda doğru bilgiyi bulmak zaman alır.
- Genel modeller belgeye dayanmadan bilgi uydurabilir.
- Özel dokümanların dış servislere gönderilmesi istenmeyebilir.

**Konuşma notu:** Bir öğrencinin uzun ders dokümanında bilgi aramasını örnek ver.

## Slayt 3 — Çözüm

- PDF, TXT ve Markdown yükleme
- Türkçe soru-cevap, özet ve quiz modları
- Cevap altında açık kaynak gösterimi

**Konuşma notu:** Uygulamanın yalnızca bulunan bağlamı kullandığını ve fallback içerdiğini vurgula.

## Slayt 4 — RAG Mimarisi

- Doküman → Loader → Chunker
- TF-IDF Retriever → ilgili chunklar
- Foundry Local LLM → Cevap + Kaynaklar

**Konuşma notu:** Retrieve, Augment ve Generate adımlarını basitçe açıkla.

## Slayt 5 — Kullanılan Teknolojiler

- Python ve Streamlit
- pypdf ve scikit-learn
- Foundry Local SDK ve qwen2.5-0.5b

**Konuşma notu:** Her teknolojinin projedeki görevini bir cümleyle anlat.

## Slayt 6 — Demo

- Demo dokümanını işle.
- “Bu projenin amacı nedir?” sorusunu sor.
- Cevap, fallback bilgisi ve kaynak expanderlarını göster.

**Konuşma notu:** Model önceden indirilmiş olmalı; ayrıca Özet ve Quiz modlarını kısaca göster.

## Slayt 7 — Sonuç ve Gelecek Geliştirmeler

- Kaynaklı, yerel ve çalışan bir RAG MVP'si oluşturuldu.
- Çok dilli embedding ve OCR eklenebilir.
- Chat geçmişi ve çoklu koleksiyonlar geliştirilebilir.

**Konuşma notu:** Çalışan özellikleri özetle ve TF-IDF/model limitlerini dürüstçe belirt.
