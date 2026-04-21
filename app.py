import streamlit as st
from parser import extract_text, chunk_text
from vector_store import VectorStore
from ai import answer_question

st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Analyzer (RAG v2)")

uploaded_files = st.file_uploader(
    "Nahraj dokumenty",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True
)

question = st.text_input("Polož otázku k dokumentům")

# persistent store v session
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

if st.button("📥 Indexovat dokumenty"):

    if not uploaded_files:
        st.warning("Nahraj soubory")
        st.stop()

    store = st.session_state.vector_store
    store.index.reset()
    store.items = []

    with st.spinner("Indexuji dokumenty..."):

        for file in uploaded_files:
            text = extract_text(file)
            chunks = chunk_text(text)

            store.add_documents(chunks, file.name)

    st.success("Dokumenty indexovány")

if st.button("🤖 Zeptat se"):

    if not question:
        st.warning("Zadej otázku")
        st.stop()

    with st.spinner("Hledám odpověď..."):
        result = answer_question(st.session_state.vector_store, question)

    st.subheader("Odpověď")
    st.write(result)
