# 📚 Türkçe Local RAG Doküman Asistanı

Foundry Local büyük dil modeliyle yerel dokümanlarınızdan Türkçe, kaynaklı ve bağlama dayalı cevaplar üreten Streamlit tabanlı RAG uygulaması.

---

## 🎯 Projenin Amacı

Kullanıcının PDF, TXT veya Markdown belgelerini kendi bilgisayarında işlemesini, belgeler hakkında soru sormasını, özet almasını ve quiz üretmesini sağlar. **Tüm işlem yerel cihazda yapılır — dokümanlar dışarı gönderilmez.**

## 🔁 RAG Nedir?

| Adım | Açıklama |
|------|----------|
| **Bilgiyi bulma** | Soruyla ilgili doküman parçalarını bulur. |
| **Bağlamı zenginleştirme** | Bulunan parçaları model bağlamına ekler. |
| **Cevap üretme** | Modelin yalnızca sağlanan bağlama dayanarak cevap üretmesini sağlar. |

## 🖥 Foundry Local Nedir?

Foundry Local, Microsoft tarafından sunulan bir yerel yapay zekâ geliştirme kitidir. Desteklenen büyük dil modellerini yerel cihazda indirmek, yüklemek ve çalıştırmak için kullanılır; model hazırlandıktan sonra cevap üretimi yerel cihazda gerçekleşir.

## ✨ Özellikler

- PDF, TXT ve Markdown yükleme
- Dokümanları örtüşmeli metin parçalarına ayırma
- Metin parçalarını ve vektörleri yerel SQLite dizininde saklama
- TF-IDF vektörleri ve kosinüs benzerliği ile alakalı parçaları bulma
- İsteğe bağlı, deneysel TF-IDF + Foundry embedding hibrit araması
- Foundry Local LLM ile Türkçe cevap üretimi
- **Dosya, sayfa, skor ve önizlemeli kaynak gösterimi**
- **Doküman Özeti** ve **Quiz Üret** modları
- Ayarlanabilir `top_k` ve minimum benzerlik eşiği
- **Güvenli doküman tabanlı yedek cevap sistemi** (model başarısız olursa doğrudan belgeden cevap)
- `FOUNDRY_MODEL_ALIAS` ortam değişkeni ile model seçimi

## 🛠 Kullanılan Teknolojiler

| Teknoloji | Görevi |
|-----------|--------|
| Python 3.11+ | Ana dil |
| Streamlit | Tarayıcı tabanlı kullanıcı arayüzü |
| pypdf | PDF okuma |
| scikit-learn | TF-IDF vektörleştirme ve kosinüs benzerliği |
| SQLite | Metin parçalarını ve yerel arama vektörlerini kalıcı olarak saklama |
| Foundry Local SDK | Yerel büyük dil modeli ve deneysel embedding desteği |

## 🏗 Sistem Mimarisi

```text
Doküman (PDF/TXT/MD)
       ↓
   [Doküman Yükleyici] ──> Metni çıkar
       ↓
   [Metin Bölücü] ──> Örtüşmeli parçalara böl
       ↓
   [SQLite Yerel Dizin] ──> Parçaları ve vektörleri sakla
       ↓
   [TF-IDF / Deneysel Hibrit Arama] ──> İlgili parçaları bul
       ↓
   [Foundry Local LLM] ──> Yalnızca bulunan bağlamdan cevap üret
       ↓                         ↓ başarısızsa
   Cevap + Kaynaklar <── [Güvenli Yedek Cevap]
```

## 🗃 SQLite ve Vektör Dizini

Doküman parçaları ve yerel arama vektörleri `data/index/rag_index.sqlite`
dosyasında saklanır. Arama bileşeni önce SQLite'taki vektörler üzerinde kosinüs
benzerliği araması yapar. SQLite veya vektör araması kullanılamazsa mevcut bellek
içi TF-IDF araması otomatik olarak güvenli yedek görevi görür.

Bu MVP sürümünde güvenilirlik ve demo stabilitesi için TF-IDF tabanlı yerel vektörler kullanılmıştır. Mimari, ileride Foundry Local embedding modeliyle değiştirilebilecek şekilde tasarlanmıştır.

Foundry Local büyük dil modeli yalnızca cevap üretiminde kullanılır. Proje, Foundry embedding
modelinin her ortamda bulunduğunu iddia etmez. Gelecekte SDK ve cihaz desteği
uygun olduğunda TF-IDF vektörleri özel bir Foundry Local embedding modeliyle
değiştirilebilir.

## 🔎 Arama Yöntemleri

Arayüzde üç arama yöntemi seçilebilir:

1. **TF-IDF:** Hızlı, kelime bazlı ve güvenli varsayılan yöntemdir.
2. **Foundry Embedding:** `qwen3-embedding-0.6b` gibi uyumlu bir embedding
   modeli önceden Foundry Local önbelleğinde hazırsa anlam bazlı arama yapar.
3. **Hybrid:** TF-IDF ve embedding sonuçlarını birleştirir. Her iki yöntemde de
   bulunan parçalara ek ağırlık vererek kelime ve anlam eşleşmesini birlikte
   kullanır.

Varsayılan arama yöntemi TF-IDF’tir. Foundry Embedding ve Hybrid modları, ilgili embedding modeli Foundry Local üzerinde hazır olduğunda semantik arama için kullanılabilir. Embedding kullanılamazsa sistem otomatik olarak TF-IDF yedeğine döner.

Embedding model alias değeri gelişmiş ayarlardan seçilebilir. Varsayılan değer
`qwen3-embedding-0.6b` şeklindedir. Uygulama embedding modeli indirmez; alias boşsa,
model önbellekte değilse, SDK uygun değilse veya embedding çağrısı hata verirse
çalışmaya TF-IDF ile devam eder.

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
│   └── index/                 # SQLite yerel dizini
└── src/
    ├── __init__.py
    ├── document_loader.py     # Dosya okuma
    ├── chunker.py             # Metin bölme
    ├── embedding_client.py    # İndirmesiz yerel vektör üretimi
    ├── vector_store.py        # SQLite chunk / vektör deposu
    ├── retriever.py           # SQLite vektör araması + TF-IDF yedeği
    ├── foundry_client.py      # Foundry Local SDK yönetimi
    ├── rag_pipeline.py        # RAG akışı + güvenli yedek cevap
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

## 🎬 Gösterim Senaryosu

1. Sol menüden sunumda kullanılacak tek bir PDF, TXT veya Markdown belgesi yükleyin.
2. **Dokümanları işle** butonuna basın. Uygulama önceki aktif belgeleri temizler ve yalnızca seçilen dosyayı indeksler.
3. Belgeye özgü bir soru sorun; yanıtı ve kaynakları gösterin.
4. **Doküman Özeti** ve **Quiz Üret** modlarını deneyin.
5. Gösterim sonunda isterseniz **Yüklenen dokümanları temizle** butonuyla aktif belgeleri kaldırın.

## 🛡 Güvenli Yedek Cevap Sistemi

Uygulama öncelikle **Foundry Local büyük dil modelini** kullanarak cevap üretir. Model cevabı aşağıdaki durumlardan birini taşıyorsa güvenli, doküman tabanlı yedek cevap devreye girer:

- Boş veya çok kısa (<20 karakter)
- Hata mesajı içeriyor
- Sohbet şablonu artıkları (`<|im_start|>`, `<|im_end|>`) barındırıyor
- Tekrar eden kelime/cümle desenleri
- Sistem kullanıcı rolü etiketleri (`system:`, `user:`, `assistant:`)

Yedek cevap aktif olduğunda bulunan kaynak parçalarından doğrudan kısa maddeler oluşturulur. Arayüz, bu yöntemin kullanıldığını açıkça belirtir.

## 🧪 Testler

Önce Python dosyalarının derlenebilir olduğunu kontrol edin, ardından model
indirmeden belge yükleme, metin bölme, SQLite dizini ve ilgili parçaları bulma
akışını test edin:

```bash
python -m compileall app.py src test_retrieval.py
python test_retrieval.py
```

## ⚠️ Limitasyonlar

| Limitasyon | Açıklama |
|------------|----------|
| TF-IDF vektörleri | Kelime eşleşmesine dayanır; özel bir semantik embedding modeli kadar güçlü değildir |
| Deneysel embedding | Yalnızca uyumlu ve önceden indirilmiş Foundry Local embedding modeliyle etkinleşir |
| Model kalitesi | Seçilen yerel modele bağlıdır |
| Büyük PDF'ler | İşlem süresi uzayabilir |
| Taranmış PDF'ler | Metin çıkarma kalitesi düşük olabilir |
| `qwen2.5-0.5b` | Küçük ve hızlıdır ancak her zaman kaliteli cevap üretemeyebilir (güvenli yedek cevap bunun için vardır) |

## 🔮 Gelecek Geliştirmeler

- [ ] Daha geniş Foundry Local embedding model ve cihaz desteği
- [ ] Sohbet geçmişi
- [ ] OCR desteği (taranmış PDF'ler için)
- [ ] Gelişmiş kaynak vurgulama
- [ ] Çoklu koleksiyon desteği
- [ ] İsteğe bağlı daha güçlü yerel modeller

## 👨‍💻 Geliştirici

**Atilla Çetin** — [GitHub](https://github.com/atillactn12/turkce-local-rag-asistani)
