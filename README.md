# Türkçe Local RAG Doküman Asistanı

**Turkish Local RAG Document Assistant**

Foundry Local LLM kullanarak yerel dokümanlardan Türkçe, kaynaklı ve bağlama dayalı cevaplar üreten Streamlit RAG uygulaması.

## Projenin Amacı

Kullanıcının PDF, TXT veya Markdown belgelerini kendi bilgisayarında işlemesini; belgeler hakkında soru sormasını, özet almasını ve quiz üretmesini sağlar.

## RAG Nedir?

1. **Retrieve:** Soruyla ilgili doküman parçalarını bulur.
2. **Augment:** Bulunan parçaları model bağlamına ekler.
3. **Generate:** Model bağlama dayanarak cevap üretir.

## Foundry Local Nedir?

Foundry Local, desteklenen yapay zekâ modellerini yerel cihazda indirmek, yüklemek ve çalıştırmak için kullanılan Microsoft SDK'sidir.

## Özellikler

- PDF, TXT ve Markdown yükleme
- Doküman okuma ve örtüşmeli chunking
- TF-IDF ve cosine similarity tabanlı retrieval
- Foundry Local LLM ile Türkçe cevap
- Dosya, sayfa, skor ve önizlemeli kaynak gösterimi
- Doküman Özeti ve Quiz Üret modları
- Ayarlanabilir `top_k` ve minimum skor
- Güvenli extractive fallback sistemi
- `FOUNDRY_MODEL_ALIAS` ile model seçimi

## Kullanılan Teknolojiler

- Python 3.11+
- Streamlit
- pypdf
- scikit-learn
- Foundry Local SDK

## Sistem Mimarisi

```text
Doküman
→ Loader
→ Chunker
→ Retriever
→ Foundry Local LLM
→ Cevap + Kaynaklar
```

## Klasör Yapısı

```text
turkce-local-rag-asistani/
├── app.py
├── requirements.txt
├── README.md
├── PRESENTATION_OUTLINE.md
├── test_retrieval.py
├── data/
│   ├── documents/demo_proje_bilgisi.txt
│   └── index/
└── src/
    ├── __init__.py
    ├── document_loader.py
    ├── chunker.py
    ├── retriever.py
    ├── foundry_client.py
    ├── rag_pipeline.py
    └── utils.py
```

## Kurulum

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

İsteğe bağlı model seçimi:

```bash
export FOUNDRY_MODEL_ALIAS=qwen2.5-0.5b
```

## Çalıştırma

```bash
streamlit run app.py
```

## Örnek Kullanım

1. Uygulamayı açın.
2. Bir doküman yükleyin veya hazır demo belgesini kullanın.
3. **Dokümanları işle** butonuna basın.
4. Çalışma modunu seçin ve cevabı kaynaklarıyla inceleyin.

## Örnek Sorular

- Bu projenin amacı nedir?
- RAG bu projede nasıl kullanılıyor?
- Foundry Local neden kullanılıyor?
- Finalde ne teslim edilecek?

## Demo Senaryosu

`data/documents/demo_proje_bilgisi.txt` belgesini işleyin. “Bu projenin amacı nedir?” sorusunu sorun; ardından Özet ve Quiz modlarını ve kaynak expanderlarını gösterin.

## Fallback Sistemi

Uygulama öncelikle Foundry Local LLM kullanır. Model cevabı başarısız, boş, 20 karakterden kısa, hatalı, tekrar eden veya chat template artığı içeren bir çıktıysa retrieved kaynak chunklardan güvenli maddeler oluşturur. Arayüz fallback kullanıldığını açıkça belirtir.

## Retrieval Testi

Bu test model indirmeden çalışır:

```bash
python test_retrieval.py
```

## Limitasyonlar

- İlk sürüm TF-IDF kullanır; diller arası anlam benzerliğini her zaman yakalayamaz.
- Model kalitesi seçilen yerel modele bağlıdır.
- Büyük PDF dosyalarında işlem süresi artabilir.
- Taranmış PDF'lerde metin çıkarma kalitesi düşük olabilir.
- `qwen2.5-0.5b` küçük ve hızlıdır ancak her zaman kaliteli cevap üretemeyebilir; bu nedenle fallback vardır.

## Gelecek Geliştirmeler

- Embedding tabanlı vector search
- Chat geçmişi
- OCR desteği
- Daha iyi kaynak gösterimi
- Çoklu koleksiyon desteği
- İsteğe bağlı daha güçlü yerel modeller

## Geliştirici

**Atilla Çetin**
