"""Türkçe Local RAG Doküman Asistanı Streamlit arayüzü."""

from pathlib import Path

import streamlit as st

from src.chunker import split_documents_into_chunks
from src.document_loader import load_documents_from_folder
from src.foundry_client import DEFAULT_MODEL_ALIAS, FoundryLLMClient
from src.rag_pipeline import NOT_FOUND_MESSAGE, answer_question
from src.retriever import SimpleRetriever
from src.utils import format_page_number, save_uploaded_file
from src.vector_store import SQLiteVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_FOLDER = PROJECT_ROOT / "data" / "documents"
INDEX_DB_PATH = PROJECT_ROOT / "data" / "index" / "rag_index.sqlite"
SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".pdf"}
EXAMPLE_QUESTIONS = [
    "Dokümanın amacı ve kapsamı nedir?",
    "Ana konuları ve önemli noktaları listele.",
    "Bu dokümana göre yapılması gerekenleri çıkar.",
    "Dokümanda cevabı olan 3 örnek soru öner.",
]


def initialize_session_state() -> None:
    """Streamlit yeniden çalıştığında korunacak değerleri hazırlar."""
    defaults = {
        "retriever": None,
        "chunks": [],
        "documents": [],
        "last_result": None,
        "question_text": "",
        "model_alias": DEFAULT_MODEL_ALIAS,
        "llm_client": None,
        "result_title": "Cevap",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_llm_client(model_alias: str) -> FoundryLLMClient:
    """Seçilen alias için session içinde tek bir istemci tutar."""
    current = st.session_state.llm_client
    if current is None or current.model_alias != model_alias:
        if current is not None:
            current.unload()
        st.session_state.llm_client = FoundryLLMClient(model_alias)
    return st.session_state.llm_client


def clear_uploaded_documents() -> None:
    """Aktif klasördeki yalnızca desteklenen kullanıcı dokümanlarını temizler."""
    DOCUMENTS_FOLDER.mkdir(parents=True, exist_ok=True)
    for file_path in DOCUMENTS_FOLDER.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS:
            file_path.unlink()


def clear_local_index() -> None:
    """SQLite içindeki aktif chunk ve vektör kayıtlarını güvenle temizler."""
    vector_store = SQLiteVectorStore(str(INDEX_DB_PATH))
    try:
        vector_store.initialize()
        vector_store.clear()
    finally:
        vector_store.close()


def reset_document_state() -> None:
    """Doküman ve cevapla ilgili session state değerlerini sıfırlar."""
    current_retriever = st.session_state.retriever
    if current_retriever is not None and hasattr(current_retriever, "close"):
        current_retriever.close()
    st.session_state.documents = []
    st.session_state.chunks = []
    st.session_state.retriever = None
    st.session_state.last_result = None


def process_documents() -> None:
    """Yerel dokümanları yükler, chunklara böler ve indeksler."""
    documents = load_documents_from_folder(str(DOCUMENTS_FOLDER))
    if not documents:
        raise ValueError("İşlenebilir, boş olmayan doküman bulunamadı.")
    chunks = split_documents_into_chunks(documents)
    if not chunks:
        raise ValueError("Dokümanlardan chunk oluşturulamadı.")
    current_retriever = st.session_state.retriever
    if current_retriever is not None and hasattr(current_retriever, "close"):
        current_retriever.close()
    retriever = SimpleRetriever(db_path=str(INDEX_DB_PATH))
    retriever.build_index(chunks)
    st.session_state.documents = documents
    st.session_state.chunks = chunks
    st.session_state.retriever = retriever
    st.session_state.last_result = None


def show_sources(sources: list[dict]) -> None:
    """Kaynakları profesyonel expander bileşenlerinde gösterir."""
    st.subheader("📑 Kullanılan Kaynaklar")
    if not sources:
        st.info("Kaynak bulunamadı.")
        return
    for order, source in enumerate(sources, start=1):
        page = format_page_number(source.get("page_number"))
        file_name = source.get("file_name", "Bilinmeyen dosya")
        score = float(source.get("score", 0.0))
        with st.expander(f"📄 Kaynak {order} — {file_name} — Sayfa {page} (skor: {score:.4f})"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Chunk ID**\n\n{source.get('chunk_id', '-')}")
                st.markdown(f"**Dosya adı**\n\n{file_name}")
            with col2:
                st.markdown(f"**Sayfa numarası**\n\n{page}")
                st.markdown(f"**Benzerlik skoru**\n\n{score:.4f}")
            st.markdown("---")
            st.markdown("**Chunk önizlemesi:**")
            st.markdown(f"> {source.get('preview', '')}")


def automatic_question(mode: str) -> str:
    """Özet ve quiz modları için kullanıcı talimatını oluşturur."""
    if mode == "Doküman Özeti":
        return "Bu dokümanı Türkçe olarak kısa, anlaşılır ve maddeler halinde özetle."
    if mode == "Quiz Üret":
        return (
            "Bu dokümana göre 5 tane Türkçe çoktan seçmeli soru hazırla. "
            "Her soruda 4 seçenek olsun ve doğru cevabı belirt."
        )
    return st.session_state.question_text.strip()


def main() -> None:
    st.set_page_config(page_title="Türkçe Local RAG Asistanı", page_icon="📚", layout="wide")
    initialize_session_state()

    st.title("📚 Türkçe Local RAG Doküman Asistanı")
    st.write(
        "Foundry Local LLM ile yerel dokümanlarınızdan Türkçe, kaynaklı ve "
        "bağlama dayalı cevaplar üretin."
    )

    if st.session_state.retriever is None:
        st.info("📌 Başlamak için sol menüden doküman yükleyin ve **Dokümanları işle** butonuna tıklayın.")

    with st.sidebar:
        st.header("📂 Dokümanlar")
        uploaded_files = st.file_uploader(
            "PDF, TXT veya Markdown dosyası seçin",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if st.button("📄 Dokümanları işle", use_container_width=True):
            selected_files = uploaded_files or []
            if not selected_files:
                st.warning("Lütfen önce yüklenecek dosyaları seçin.")
            else:
                try:
                    # Her işlem yalnızca o anda seçilen dosyaları indeksler.
                    clear_uploaded_documents()
                    for uploaded_file in selected_files:
                        save_uploaded_file(uploaded_file, str(DOCUMENTS_FOLDER))
                    with st.spinner("Dokümanlar okunuyor ve indeksleniyor..."):
                        process_documents()
                    st.success("✅ Doküman indeksi hazırlandı.")
                except ValueError as error:
                    st.warning(f"⚠️ {error}")
                except Exception as error:
                    st.error(f"❌ Dokümanlar işlenemedi: {error}")

        if st.button("Yüklenen dokümanları temizle", use_container_width=True):
            try:
                clear_uploaded_documents()
                reset_document_state()
                clear_local_index()
                st.success("Yüklenen dokümanlar ve aktif indeks temizlendi.")
            except Exception as error:
                st.error(f"Dokümanlar temizlenemedi: {error}")

        if st.session_state.chunks:
            file_count = len({item.get("file_path") for item in st.session_state.documents})
            st.metric("Yüklenen dosya", file_count)
            st.metric("Okunan doküman / sayfa", len(st.session_state.documents))
            st.metric("Oluşturulan chunk", len(st.session_state.chunks))
        else:
            st.info("Henüz hazırlanmış bir indeks yok.")

        st.divider()
        st.header("⚙️ Gelişmiş Ayarlar")
        top_k = st.slider("Gösterilecek kaynak sayısı (top_k)", 1, 8, 5)
        minimum_score = st.slider("Minimum benzerlik skoru (eşik)", 0.00, 0.20, 0.03, 0.01)

        st.divider()
        st.header("🤖 Model Ayarı")
        model_alias = st.text_input("Model alias", key="model_alias")
        st.caption("Daha büyük bir model yazarsanız ilk kullanımda indirme süresi uzayabilir. Varsayılan: qwen2.5-0.5b")

        st.divider()
        st.header("ℹ️ Proje Bilgisi")
        st.caption("Bu uygulama; PDF, TXT ve Markdown dosyalarınızı kullanarak RAG (Retrieval-Augmented Generation) yöntemiyle sorularınızı cevaplar, özet çıkarır ve quiz oluşturur.")
        st.caption("Yerel index: SQLite + TF-IDF fallback")
        if (
            st.session_state.retriever is not None
            and getattr(st.session_state.retriever, "sqlite_ready", False)
        ):
            st.success("SQLite index hazır.")
        elif st.session_state.retriever is not None:
            st.caption("SQLite kullanılamadı; TF-IDF fallback hazır.")

    mode = st.radio(
        "Çalışma modu seçin",
        ["Soru-Cevap", "Doküman Özeti", "Quiz Üret"],
        horizontal=True,
    )

    if mode == "Soru-Cevap":
        st.subheader("Örnek sorular")
        columns = st.columns(2)
        for index, example in enumerate(EXAMPLE_QUESTIONS):
            if columns[index % 2].button(example, key=f"example_{index}", use_container_width=True):
                st.session_state.question_text = example
        st.text_area("Türkçe sorunuzu yazın", key="question_text", height=110)
        action_label, result_title = "Cevap üret", "Cevap"
    elif mode == "Doküman Özeti":
        st.info("İşlenen dokümanlar Türkçe ve maddeler halinde özetlenecek.")
        action_label, result_title = "Özet oluştur", "Doküman Özeti"
    else:
        st.info("Doküman içeriğine göre 5 Türkçe çoktan seçmeli soru hazırlanacak.")
        action_label, result_title = "Quiz oluştur", "Quiz"

    if st.button(action_label, type="primary", use_container_width=True):
        if st.session_state.retriever is None:
            st.error("Önce dokümanları işleyin.")
        elif not model_alias.strip():
            st.error("Model alias boş olamaz.")
        else:
            question = automatic_question(mode)
            if not question:
                st.error("Lütfen bir soru yazın.")
            else:
                try:
                    llm_client = get_llm_client(model_alias.strip())
                    mode_top_k = min(8, max(top_k, 6)) if mode != "Soru-Cevap" else top_k
                    pipeline_mode = {
                        "Soru-Cevap": "qa",
                        "Doküman Özeti": "summary",
                        "Quiz Üret": "quiz",
                    }[mode]
                    with st.spinner("Model yükleniyor, ilk çalıştırma uzun sürebilir..."):
                        result = answer_question(
                            question, st.session_state.retriever, llm_client,
                            top_k=mode_top_k, minimum_score=minimum_score,
                            mode=pipeline_mode,
                        )
                    st.session_state.last_result = result
                    st.session_state.result_title = result_title
                except Exception as error:
                    st.error(f"İşlem tamamlanamadı: {error}")

    result = st.session_state.last_result
    if result:
        st.divider()
        st.subheader(f"📝 {st.session_state.result_title}")
        # Display the answer
        st.markdown(result.get("answer", ""))

        # Fallback / Model source badge — clearly visible
        if result.get("used_fallback"):
            st.error("🛡️ **Fallback:** Model yanıtı yeterli olmadığı için güvenli doküman tabanlı fallback kullanıldı.")
        elif result.get("sources") and result.get("answer") != NOT_FOUND_MESSAGE:
            st.success("✅ Yanıt Foundry Local LLM ile üretildi.")
        else:
            st.info("ℹ️ İlgili kaynak bulunamadığı için model yanıtı kullanılmadı.")
        st.caption(f"Kullanılan model: `{result.get('model_alias', 'Bilinmiyor')}`")
        st.divider()
        show_sources(result.get("sources", []))


if __name__ == "__main__":
    main()
