# 🎤 5 Dakikalık Final Sunumu

## Slayt 1 — 📚 Türkçe Local RAG Doküman Asistanı (30 sn)

| Konu | İçerik |
|------|--------|
| **Ne?** | Yerel dokümanlardan Türkçe soru-cevap, özet ve quiz üreten RAG uygulaması. |
| **Nasıl?** | Python + Streamlit + SQLite + TF-IDF + Foundry Local |
| **Geliştirici** | Atilla Çetin |

**Konuşma notu:** Projenin yerel çalıştığını, dokümanların dış servise gönderilmediğini ve Türkçe cevap üretmeye odaklandığını vurgula.

---

## Slayt 2 — ❓ Problem (30 sn)

- Uzun dokümanlarda doğru bilgiyi hızlı bulmak zordur.
- Genel sohbet modelleri belgeye dayanmadan bilgi uydurabilir.
- Özel dokümanların dış servislere gönderilmesi güvenlik riski oluşturabilir.
- Demo sırasında sistemin stabil ve açıklanabilir olması gerekir.

**Konuşma notu:** “Bu proje, cevabı yalnızca yüklenen dokümanlardan bulmaya çalışan yerel bir RAG asistanıdır.”

---

## Slayt 3 — ✅ Çözüm ve Özellikler (40 sn)

- PDF, TXT ve Markdown yükleme
- Türkçe soru-cevap
- Doküman özeti
- Quiz üretme
- Kaynak gösterimi: dosya, sayfa, skor ve chunk önizlemesi
- Güvenli fallback cevap sistemi
- Kapsam dışı sorularda “Bu bilgi yüklenen dokümanlarda bulunamadı.” cevabı

**Konuşma notu:** Cevapların altında kaynak gösteriminin demo için en güçlü kanıt olduğunu söyle.

---

## Slayt 4 — 🏗 Mimari (1 dk)

```text
Document Upload
      ↓
Loader
      ↓
Chunker
      ↓
SQLite Local Index
      ↓
TF-IDF / Foundry Embedding / Hybrid Retrieval
      ↓
Foundry Local LLM
      ↓
Answer + Sources
      ↓
Safe Fallback if needed
```

**Konuşma notu:** “Dokümanlar önce küçük parçalara ayrılır. Parçalar ve vektörler SQLite içinde yerel olarak saklanır. Soru geldiğinde retrieval katmanı ilgili parçaları bulur. LLM cevabı zayıfsa fallback doğrudan kaynak chunklardan cevap üretir.”

---

## Slayt 5 — 🔎 Arama Modları (45 sn)

| Mod | Açıklama |
|-----|----------|
| **TF-IDF** | Hızlı ve stabil varsayılan mod. Embedding modeli gerektirmez. |
| **Foundry Embedding** | `qwen3-embedding-0.6b` hazırsa 1024 boyutlu semantik vektörlerle arama yapar. |
| **Hybrid Retrieval** | TF-IDF ve Foundry Embedding sonuçlarını birleştirir. Embedding yoksa TF-IDF yedeğine döner. |

**Konuşma notu:** “Hybrid mod hem kelime eşleşmesini hem anlam benzerliğini kullanır. Bu da demo sırasında daha esnek sonuçlar verir.”

---

## Slayt 6 — 🗃 SQLite Yerel İndeks ve Kaynaklar (35 sn)

- Chunklar SQLite içinde saklanır.
- TF-IDF vektörleri `tfidf_fallback` provider adıyla saklanır.
- Foundry embedding vektörleri `foundry_embedding` provider adıyla saklanır.
- Kaynak gösterimi kullanıcıya cevabın nereden geldiğini gösterir.

**Konuşma notu:** Test komutuyla provider sayıları gösterilebilir:

```bash
sqlite3 data/index/rag_index.sqlite "SELECT provider, COUNT(*) FROM embeddings GROUP BY provider;"
```

---

## Slayt 7 — 🛡 Güvenli Fallback ve Kapsam Dışı Cevap (40 sn)

- Küçük yerel LLM bazen zayıf veya bozuk cevap üretebilir.
- Bu durumda sistem kaynak doküman parçalarından güvenli fallback cevap oluşturur.
- UI bunu kırmızı hata gibi değil, nötr şekilde gösterir:

```text
Yanıt kaynak dokümanlara göre oluşturuldu.
```

- Cevap dokümanda yoksa:

```text
Bu bilgi yüklenen dokümanlarda bulunamadı.
```

**Konuşma notu:** “Bu yaklaşım hallucination riskini azaltmak için eklendi.”

---

## Slayt 8 — 🎬 5 Dakikalık Demo Akışı (1,5 dk)

1. Uygulamayı aç.
2. Sol menüden PDF/TXT/Markdown dokümanı yükle.
3. Arama modunu seç: TF-IDF veya Hybrid.
4. **Dokümanları işle** butonuna bas.
5. Soru-Cevap modunda dokümana özgü bir soru sor.
6. Cevap altındaki kaynak expander’ını aç.
7. Doküman Özeti modunu göster.
8. Quiz Üret modunu göster.
9. Kapsam dışı bir soru sorarak güvenli “bulunamadı” cevabını göster.

**Konuşma notu:** İlk model yüklemesi uzun sürebileceği için demo öncesinde modeli ve embedding indeksini hazırlamak iyi olur.

---

## Slayt 9 — 🛠 Kullanılan Teknolojiler (30 sn)

| Teknoloji | Görevi |
|-----------|--------|
| Python 3.11+ | Ana programlama dili |
| Streamlit | Web arayüzü |
| pypdf | PDF okuma |
| scikit-learn | TF-IDF ve cosine similarity |
| SQLite | Yerel chunk ve vektör indeksi |
| Foundry Local SDK | Yerel LLM ve embedding desteği |
| qwen2.5-0.5b | Varsayılan yerel LLM |
| qwen3-embedding-0.6b | Foundry Embedding modu için embedding modeli |

---

## Slayt 10 — 🏁 Sonuç ve Gelecek Geliştirmeler (30 sn)

**Başarılanlar:**

- Çalışan Streamlit MVP
- SQLite yerel indeks
- TF-IDF retrieval
- Foundry Embedding retrieval
- Hybrid Retrieval
- Kaynak gösterimi
- Güvenli fallback
- Özet ve quiz modları

**Gelecek geliştirmeler:**

- OCR desteği
- Sohbet geçmişi
- Çoklu koleksiyon desteği
- Daha gelişmiş kaynak vurgulama
- Daha güçlü yerel modeller

**Konuşma notu:** “Final hedefi, hızlı, stabil, yerel ve kaynaklı bir Türkçe RAG demosu sunmaktı. Bu hedef karşılandı.”
