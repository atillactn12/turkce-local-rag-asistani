# 📚 Türkçe Local RAG Doküman Asistanı

Türkçe Local RAG Doküman Asistanı; PDF, TXT ve Markdown dokümanlarından Türkçe soru-cevap, özet ve quiz üretebilen yerel bir Streamlit uygulamasıdır. Proje; SQLite tabanlı yerel indeks, TF-IDF retrieval, isteğe bağlı Foundry Embedding, Hybrid retrieval ve Foundry Local LLM bileşenlerini birlikte kullanır.

Bu proje Microsoft Summer School / Foundry Local RAG projesi kapsamında geliştirilmiştir.

---

## 🎯 Projenin Amacı

Amaç, kullanıcının yerel dokümanlarını kendi bilgisayarında işlemesini ve bu dokümanlara dayalı Türkçe cevaplar üretmesini sağlamaktır. Uygulama, dokümanları küçük parçalara ayırır, ilgili parçaları bulur, kaynakları gösterir ve gerekirse güvenli doküman tabanlı fallback cevabı üretir.

## ✨ Ana Özellikler

- PDF, TXT ve Markdown dosyası yükleme
- Türkçe soru-cevap
- Doküman özeti modu
- Quiz üretme modu
- Kaynak gösterimi: dosya adı, sayfa numarası, skor ve chunk önizlemesi
- SQLite tabanlı yerel indeks
- TF-IDF retrieval
- Foundry Embedding retrieval
- Hybrid retrieval: TF-IDF + Foundry Embedding
- Güvenli doküman tabanlı fallback cevap sistemi
- Kapsam dışı soru kontrolü

## 🔎 Arama Modları

Uygulamada üç arama modu bulunur.

### 1. TF-IDF

TF-IDF hızlı ve stabil varsayılan arama modudur. Kelime ve vektör benzerliği mantığıyla soruya en yakın doküman parçalarını bulur. Embedding modeli gerektirmez, bu nedenle demo ve temel kullanım için güvenli seçenektir.

### 2. Foundry Embedding

Foundry Embedding modu, `qwen3-embedding-0.6b` modeli hazır olduğunda semantik arama yapar. Bu mod 1024 boyutlu embedding vektörleri üretir ve vektörleri SQLite içinde `foundry_embedding` provider adıyla saklar.

Bu modun çalışması için embedding modelinin Foundry Local üzerinde yerel olarak mevcut olması gerekir. Başka bir makinede model daha önce indirilmemişse ayrıca hazırlanması gerekebilir.

### 3. Hybrid

Hybrid mod, TF-IDF sonuçları ile Foundry Embedding sonuçlarını birleştirir. Böylece hem kelime eşleşmesi hem de anlam benzerliği birlikte kullanılır. Embedding kullanılamazsa sistem güvenli şekilde TF-IDF yedeğine döner.

## 🗃 SQLite Yerel İndeks

Doküman parçaları ve vektör kayıtları yerel SQLite dosyasında saklanır:

```text
data/index/rag_index.sqlite
```

Dokümanlar işlendiğinde SQLite indeks otomatik oluşturulur. Aynı indeks içinde farklı vektör sağlayıcıları saklanabilir:

```text
tfidf_fallback
foundry_embedding
```

SQLite bağlantıları kısa ömürlü açılıp kapatılacak şekilde tasarlanmıştır. Bu yaklaşım Streamlit rerun/thread davranışında SQLite bağlantı hatalarını azaltır.

## 🧠 Foundry Local Kullanımı

Varsayılan yerel LLM modeli:

```text
qwen2.5-0.5b
```

Embedding modu için kullanılan model alias:

```text
qwen3-embedding-0.6b
```

`qwen2.5-0.5b` küçük ve hızlı bir yerel modeldir. Küçük modeller bazı durumlarda zayıf cevap üretebilir; bu yüzden uygulamada güvenli fallback sistemi vardır.

Önemli not: Bu proje, `qwen3-embedding-0.6b` modelinin her makinede otomatik olarak hazır olduğunu iddia etmez. Foundry Embedding veya Hybrid modunun semantik kısmı için modelin ilgili cihazda Foundry Local tarafından erişilebilir olması gerekir.

## 🛡 Güvenli Fallback Sistemi

Uygulama önce Foundry Local LLM ile cevap üretmeyi dener. Model cevabı boş, hatalı, çok kısa, tekrarlı veya chat template artığı içeriyorsa güvenli doküman tabanlı fallback devreye girer.

Fallback cevabı, bulunan doküman parçalarından doğrudan oluşturulur. Bu yaklaşım hallucination riskini azaltır.

Arayüzde fallback durumu kırmızı hata gibi gösterilmez. Nötr bilgi mesajı kullanılır:

```text
Yanıt kaynak dokümanlara göre oluşturuldu.
```

## 🚫 Kapsam Dışı Soru Davranışı

Cevap yüklenen dokümanlarda yoksa sistem şu cevabı döndürmelidir:

```text
Bu bilgi yüklenen dokümanlarda bulunamadı.
```

Örneğin dokümanda bütçe, IP adresi, resmi web adresi veya proje yöneticisi gibi bilgiler yoksa sistem bu bilgileri uydurmaz. Negatif test bölümlerindeki “bu soru dokümanda yoktur” tarzı satırlar gerçek cevap olarak kabul edilmez.

## 🏗 Proje Mimarisi

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

## 📁 Klasör Yapısı

```text
turkce-local-rag-asistani/
├── app.py
├── requirements.txt
├── README.md
├── PRESENTATION_OUTLINE.md
├── test_retrieval.py
├── data/
│   ├── documents/
│   └── index/
└── src/
    ├── document_loader.py
    ├── chunker.py
    ├── retriever.py
    ├── embedding_client.py
    ├── vector_store.py
    ├── foundry_client.py
    ├── rag_pipeline.py
    └── utils.py
```

## 🛠 Kullanılan Teknolojiler

| Teknoloji | Görevi |
|---|---|
| Python 3.11+ | Ana programlama dili |
| Streamlit | Web arayüzü |
| pypdf | PDF metni okuma |
| scikit-learn | TF-IDF ve cosine similarity |
| SQLite | Yerel chunk ve vektör indeksi |
| Foundry Local SDK | Yerel LLM ve embedding entegrasyonu |

## 📦 Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Çalıştırma

```bash
streamlit run app.py
```

Uygulama açıldıktan sonra:

1. Sol menüden PDF, TXT veya Markdown dosyası yükleyin.
2. **Dokümanları işle** butonuna basın.
3. Arama modunu seçin: TF-IDF, Foundry Embedding veya Hybrid.
4. Soru-Cevap, Doküman Özeti veya Quiz Üret modunu kullanın.
5. Cevapla birlikte gösterilen kaynakları inceleyin.

## 🧪 Test Komutları

```bash
.venv/bin/python -m compileall app.py src test_retrieval.py
.venv/bin/python test_retrieval.py
sqlite3 data/index/rag_index.sqlite "SELECT provider, COUNT(*) FROM embeddings GROUP BY provider;"
```

Hybrid veya Foundry Embedding ile indeks oluşturulduktan sonra beklenen provider çıktısı şu yapıda olur:

```text
tfidf_fallback|...
foundry_embedding|...
```

Sayılar yüklenen dokümana ve oluşan chunk sayısına göre değişebilir.

## ⚠️ Limitasyonlar

- Foundry Embedding modu için `qwen3-embedding-0.6b` modelinin yerel Foundry Local ortamında hazır olması gerekir.
- Embedding modeli yoksa sistem TF-IDF fallback ile çalışmaya devam eder.
- Küçük yerel LLM modelleri bazı cevaplarda yetersiz kalabilir; bu durumda güvenli doküman tabanlı fallback kullanılır.
- SQLite indeks yerel olarak oluşturulur; yeni doküman yüklediğinizde indeksin yeniden oluşturulması gerekir.
- Taranmış PDF dosyalarında metin çıkarma kalitesi düşük olabilir.

## 🔮 Gelecek Geliştirmeler

- OCR desteği
- Çoklu doküman koleksiyonları
- Sohbet geçmişi
- Daha gelişmiş kaynak vurgulama
- Daha güçlü yerel LLM seçenekleri
- Foundry Local embedding model seçeneklerinin genişletilmesi

## 👨‍💻 Geliştirici

**Atilla Çetin**  
GitHub: https://github.com/atillactn12/turkce-local-rag-asistani
