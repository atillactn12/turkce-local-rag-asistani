# 🎤 5 Dakikalık Final Sunumu

## Slayt 1 — 📚 Türkçe Local RAG Doküman Asistanı (30 sn)

| Konu | İçerik |
|------|--------|
| **Ne?** | Yerel dokümanlardan Türkçe soru-cevap, özet ve quiz üreten RAG uygulaması. |
| **Nasıl?** | Python + Streamlit + Foundry Local (Microsoft yerel LLM SDK'sı) |
| **Geliştirici** | Atilla Çetin |

**🗣 Konuşma notu:** Projenin adını ve temel amacını 20 saniyede tanıt. "Foundry Local sayesinde tüm işlem bilgisayarınızda döner, dokümanlar dışarı çıkmaz."

---

## Slayt 2 — ❓ Problem (30 sn)

- Uzun dokümanlarda doğru bilgiyi bulmak **zaman alır**.
- Genel sohbet modelleri (ChatGPT vb.) belgeye dayanmadan **bilgi uydurabilir (hallucination)**.
- Özel/kişisel dokümanların **dış servislere gönderilmesi güvenlik riski** oluşturur.

**🗣 Konuşma notu:** Bir öğrencinin uzun ders dokümanında bilgi aramasını örnek ver. "Şimdi uygulamanın bu sorunu nasıl çözdüğüne bakalım."

---

## Slayt 3 — ✅ Çözüm (30 sn)

| Özellik | Açıklama |
|---------|----------|
| 📂 Yükleme | PDF, TXT ve Markdown dosyalarını yükle |
| ❓ Soru-Cevap | Dokümana dayalı Türkçe cevaplar |
| 📝 Doküman Özeti | Otomatik madde madde özet |
| 📋 Quiz Üret | 5 soruluk çoktan seçmeli test |
| 📄 Kaynak gösterimi | Cevabın hangi doküman/sayfadan geldiğini göster |

**🗣 Konuşma notu:** "Uygulama yalnızca dokümanda bulduğu bağlamı kullanır — eğer model başarısız olursa fallback devreye girer."

---

## Slayt 4 — 🏗 RAG Mimarisi (1 dk)

```text
Doküman (PDF/TXT/MD)
       ↓
   [Böl: Chunker]
       ↓
   [Bul: TF-IDF Retriever]
       ↓
   [Üret: Foundry Local LLM  ← başarısızsa → Fallback]
       ↓
   Cevap + Kaynak Listesi
```

**🗣 Konuşma notu:** "Retrieve (dokümanı parçala ve ilgili kısmı bul), Augment (bulunan parçayı modele bağlam olarak ver), Generate (model bağlama dayanarak cevap üretir). Eğer model kötü cevap verirse doğrudan doküman parçalarından maddeler oluştururuz."

---

## Slayt 5 — 🛠 Kullanılan Teknolojiler (30 sn)

| Teknoloji | Projedeki Görevi |
|-----------|------------------|
| **Python 3.11+** | Ana programlama dili |
| **Streamlit** | Web arayüzü (sadece `app.py`) |
| **pypdf** | PDF dosyalarından metin çıkarma |
| **scikit-learn** | TF-IDF vektörleştirme + cosine similarity |
| **Foundry Local SDK** | LLM'yi yerelde indirme, yükleme, çalıştırma |
| **qwen2.5-0.5b** | Varsayılan yerel model (küçük ve hızlı) |

**🗣 Konuşma notu:** "Her teknolojinin tek bir sorumluluğu var. Toplamda 7 Python dosyası — küçük ve anlaşılır."

---

## Slayt 6 — 🎬 Demo (1,5 dk)

1. Sol menüden `demo_proje_bilgisi.txt` belgesi zaten hazır.
2. **📄 Dokümanları işle** butonuna tıklayın.
3. **Soru-Cevap** modunda "Bu projenin amacı nedir?" yazıp cevabı alın.
4. Cevap altındaki **kaynak expander**'ını açın — chunk_id, dosya, sayfa, skor ve önizlemeyi gösterin.
5. **Doküman Özeti** moduna geçip özet alın.
6. **Quiz Üret** moduna geçip 5 soruluk test oluşturun.

**🗣 Konuşma notu:** "Model önceden indirilmiş olmalı. İlk çalıştırma biraz uzun sürebilir. Fallback mesajını görmek için model alias'ı yanlış yazıp deneyebilirsiniz."

---

## Slayt 7 — 🏁 Sonuç ve Gelecek Geliştirmeler (30 sn)

**Başarılanlar:**
- ✅ Kaynaklı, yerel ve çalışan bir RAG MVP'si
- ✅ Türkçe dil desteği
- ✅ Güvenli fallback mekanizması
- ✅ Model gerektirmeyen test (`test_retrieval.py`)

**Gelecek:**
- 🔮 Embedding tabanlı vector search (daha iyi anlam)
- 🔮 OCR desteği (taranmış PDF'ler)
- 🔮 Chat geçmişi
- 🔮 Çoklu koleksiyon desteği

**🗣 Konuşma notu:** "Şu an TF-IDF kullanıyoruz — basit ama çalışıyor. Limitlerini biliyoruz ve fallback ile güvence altına alıyoruz. Gelecekte embedding'lerle çok daha iyi sonuç alınabilir."

---

> 💡 **Sunum İpuçları:**
> - Sunumu 5 dakikada bitirmek için her slaytta **ana mesajı** verin, detaya girme  ün.
> - Demo sırasında **önce soruyu yazın**, sonra butona basın — boş beklemeyin.
> - Fallback'i göstermek isterseniz model alias'ı `yok-boyle-bir-model` yapıp deneyin.
> - Kod göstermeyin — sadece **çalışan uygulamayı** gösterin.
