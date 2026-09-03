import streamlit as st
from rag import answer_query, get_chunk_count
import config

st.set_page_config(page_title=f" {config.SUBJECT_NAME} Asistanı", page_icon="📚")
st.title(config.APP_TITLE)
st.markdown(f"{config.SUBJECT_NAME} notlarıyla çalışır – tamamen çevrimdışı.*")

with st.sidebar:
    st.header("🔍 Alınan Metin Parçaları")
    show_chunks = st.checkbox("Alınan parçaları göster", value=True)
    if st.button("Sohbeti Temizle"):
        st.session_state.messages = []
    st.caption(f"📄 Toplam {get_chunk_count()} adet bilgi parçası yüklendi.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(f" {config.SUBJECT_NAME} sorusu sor..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Aranıyor ve cevap oluşturuluyor..."):
            # Tek bir retrieval çağrısı: hem cevap hem de gösterilecek parçalar buradan gelir
            response, results = answer_query(prompt, top_k=config.DISPLAY_TOP_K)
        st.markdown(response)

        if show_chunks and results:
            with st.sidebar:
                st.subheader("📖 En İlgili Parçalar")
                for i, r in enumerate(results):
                    label = f"Parça {i+1} — {r['source']}"
                    if r.get("page"):
                        label += f" (s. {r['page']})"
                    label += f" — Skor: {r['score']:.3f}"
                    with st.expander(label):
                        text = r["text"]
                        st.write(text[:600] + "..." if len(text) > 600 else text)

    st.session_state.messages.append({"role": "assistant", "content": response})
