# 📚 Türkçe Local RAG Doküman Asistanı

**Turkish Local RAG Document Assistant** — Foundry Local LLM ile yerel dokümanlarınızdan Türkçe, kaynaklı ve bağlama dayalı cevaplar üreten Streamlit RAG uygulaması.

---

## 🎯 Projenin Amacı

Kullanıcının PDF, TXT veya Markdown belgelerini kendi bilgisayarında işlemesini, belgeler hakkında soru sormasını, özet almasını ve quiz üretmesini sağlar. **Tüm işlem yerel cihazda yapılır — dokümanlar dışarı gönderilmez.**

## 🔁 RAG Nedir?

| Adım | Açıklama |
|------|----------|
| **Retrieve** | Soruyla ilgili doküman parçalarını bulur. |
| **Augment** | Bulunan parçaları model bağlamına ekler. |
| **Generate** | Model bağlama dayanarak cevap üretir. |

## 🖥 Foundry Local Nedir?

Foundry Local, Microsoft SDK'sidir. Desteklenen yapay zekâ modellerini (LLM) yerel cihazda indirmek, yüklemek ve çalıştırmak için kullanılır; çalışma sırasında internete gerek yoktur.

## ✨ Özellikler

- PDF, TXT ve Markdown yükleme
- Örtüşmeli (overlapping) chunking ile doküman bölme
- TF-IDF + cosine similarity ile alakalı parçaları bulma
- Foundry Local LLM ile Türkçe cevap üretimi
- **Dosya, sayfa, skor ve önizlemeli kaynak gösterimi**
- **Doküman Özeti** ve **Quiz Üret** modları
- Ayarlanabilir `top_k` ve minimum benzerlik eşiği
- **Güvenli extractive fallback sistemi** (model başarısız olursa doğrudan belgeden cevap)
- `FOUNDRY_MODEL_ALIAS` ortam değişkeni ile model seçimi

## 🛠 Kullanılan Teknolojiler

| Teknoloji | Görevi |
|-----------|--------|
| Python 3.11+ | Ana dil |
| Streamlit | Web arayüzü |
| pypdf | PDF okuma |
| scikit-learn | TF-IDF vektörleştirme ve cosine similarity |
| Foundry Local SDK | Yerel LLM yönetimi |

## 🏗 Sistem Mimarisi

```text
Doküman (PDF/TXT/MD)
       ↓
   [Loader] ──> Metin çıkar
       ↓
   [Chunker] ──> Örtüşmeli parçalara böl
       ↓
   [Retriever (TF-IDF)] ──> İlgili chunkları bul
       ↓
   [Foundry Local LLM] ──> Cevap üret (başarısızsa → Fallback)
       ↓
   Cevap + Kaynaklar
```

## 📁 Klasör Yapısı

```text
turkce-local-rag-asistani/
├── app.py                     # Streamlit arayüzü
├── requirements.txt           # Bağımlılıklar
├── README.md                  # Bu dosya
├── PRESENTATION_OUTLINE.md    # Sunum taslağı
├── test_retrieval.py          # Model gerektirmeyen test
├── data/
│   ├── documents/             # Dokümanlar buraya yüklenir
│   │   └── (kullanıcının aktif yüklemeleri)
│   ├── backup/                # İndekse alınmayan örnek/yedek belgeler
│   └── index/                 # İndeks dosyaları (ileri)
└── src/
    ├── __init__.py
    ├── document_loader.py     # Dosya okuma
    ├── chunker.py             # Metin bölme
    ├── retriever.py           # TF-IDF arama
    ├── foundry_client.py      # Foundry Local SDK yönetimi
    ├── rag_pipeline.py        # RAG akışı + fallback
    └── utils.py               # Yardımcı fonksiyonlar
```

## 📦 Kurulum

Makinenizde **Python 3.11+** yüklü olduğundan emin olun.

```bash
# Sanal ortam oluştur
python3.11 -m venv .venv

# Sanal ortamı aktifleştir
source .venv/bin/activate

# pip'i güncelle
pip install --upgrade pip

# Bağımlılıkları yükle
pip install -r requirements.txt
```

İsteğe bağlı — kullanılacak modeli seçin (varsayılan: `qwen2.5-0.5b`):

```bash
export FOUNDRY_MODEL_ALIAS=qwen2.5-0.5b   # Küçük ve hızlı
# export FOUNDRY_MODEL_ALIAS=Phi-3.5-mini-instruct  # Daha güçlü
```

## 🚀 Çalıştırma

```bash
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` adresinde açılır.

## 📝 Örnek Kullanım

1. Uygulamayı açın.
2. Sol menüden kendi PDF, TXT veya Markdown dokümanınızı yükleyin.
3. **📄 Dokümanları işle** butonuna basın.
4. Çalışma modunu seçin:
   - **Soru-Cevap**: Belge hakkında soru sorun.
   - **Doküman Özeti**: Otomatik özet alın.
   - **Quiz Üret**: 5 soruluk çoktan seçmeli quiz oluşturun.
5. Cevabı ve kaynakları inceleyin.

## ❓ Örnek Sorular (Soru-Cevap modu)

- Bu projenin amacı nedir?
- RAG bu projede nasıl kullanılıyor?
- Foundry Local neden kullanılıyor?
- Finalde ne teslim edilecek?

## 🎬 Demo Senaryosu

1. Sol menüden sunumda kullanılacak tek bir PDF, TXT veya Markdown belgesi yükleyin.
2. **Dokümanları işle** butonuna basın. Uygulama önceki aktif belgeleri temizler ve yalnızca seçilen dosyayı indeksler.
3. Belgeye özgü bir soru sorun; yanıtı ve kaynakları gösterin.
4. **Doküman Özeti** ve **Quiz Üret** modlarını deneyin.
5. Demo sonunda isterseniz **Yüklenen dokümanları temizle** butonuyla aktif belgeleri kaldırın.

## 🛡 Fallback Sistemi

Uygulama öncelikle **Foundry Local LLM** kullanarak cevap üretir. Model cevabı aşağıdaki durumlardan birini taşıyorsa, güvenli **extractive fallback** devreye girer:

- Boş veya çok kısa (<20 karakter)
- Hata mesajı içeriyor
- Chat template artıkları (`<|im_start|>`, `<|im_end|>`) barındırıyor
- Tekrar eden kelime/cümle desenleri
- Sistem kullanıcı rolü etiketleri (`system:`, `user:`, `assistant:`)

Fallback aktif olduğunda, bulunan kaynak chunklardan doğrudan kısa maddeler oluşturulur. **Arayüz fallback kullanıldığını açıkça belirtir:** "Model yanıtı yeterli olmadığı için güvenli doküman tabanlı fallback kullanıldı."

## 🧪 Retrieval Testi

Model indirmeden yalnızca belge yükleme, chunking ve retrieval akışını test eder:

```bash
python test_retrieval.py
```

## ⚠️ Limitasyonlar

| Limitasyon | Açıklama |
|------------|----------|
| TF-IDF | Diller arası anlam benzerliğini her zaman yakalayamaz (embedding tabanlı değil) |
| Model kalitesi | Seçilen yerel modele bağlıdır |
| Büyük PDF'ler | İşlem süresi uzayabilir |
| Taranmış PDF'ler | Metin çıkarma kalitesi düşük olabilir |
| `qwen2.5-0.5b` | Küçük ve hızlıdır ancak her zaman kaliteli cevap üretemeyebilir (fallback bunun için var) |

## 🔮 Gelecek Geliştirmeler

- [ ] Embedding tabanlı vector search (daha iyi anlam benzerliği)
- [ ] Chat geçmişi
- [ ] OCR desteği (taranmış PDF'ler için)
- [ ] Gelişmiş kaynak vurgulama
- [ ] Çoklu koleksiyon desteği
- [ ] İsteğe bağlı daha güçlü yerel modeller

## 👨‍💻 Geliştirici

**Atilla Çetin** — [GitHub](https://github.com/atillactn12/turkce-local-rag-asistani)
