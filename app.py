import streamlit as st
import tempfile
import os
import re
from pypdf import PdfReader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document  # ← import dari langchain_core
from langchain_community.llms import Ollama

# ============================================
# FUNGSI SPLIT MANUAL
# ============================================
def split_text_manual(text, chunk_size=1000, overlap=200):
    if not text.strip():
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# ============================================
# FUNGSI QA MANUAL (tanpa RetrievalQA)
# ============================================
def ask_question(query, vectorstore, llm, k=3):
    # 1. Cari dokumen relevan
    docs = vectorstore.similarity_search(query, k=k)
    
    # 2. Gabungkan menjadi konteks
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. Buat prompt
    prompt = f"""Anda adalah asisten AI yang membantu pelaut memahami regulasi maritim.
Jawab pertanyaan berikut berdasarkan konteks yang diberikan.
Jika jawaban tidak ditemukan dalam konteks, katakan "Informasi tidak tersedia dalam dokumen".

Konteks:
{context}

Pertanyaan: {query}

Jawaban:"""
    
    # 4. Kirim ke LLM
    answer = llm.invoke(prompt)
    
    return answer, docs

# ============================================
# JUDUL APLIKASI
# ============================================
st.set_page_config(page_title="Marine's QA", layout="wide")
st.title("⚓ Marine's Question Answer")
st.caption("Upload e-book maritim (SOLAS, MARPOL, COLREG) dan tanyakan apapun!")

with st.sidebar:
    st.header("⚙️ Pengaturan Model")
    model_choice = st.selectbox(
        "Pilih Model AI:",
        ["qwen2.5:3b", "llama3", "llama3.1", "llama3.2"],
        index=0
    )
    st.divider()
    st.markdown("""
    **📌 Tips:**
    - Model yang lebih besar = jawaban lebih akurat tapi lebih lambat
    - Pastikan Ollama berjalan di background
    """)
    if st.button("🗑️ Reset Database"):
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        st.success("Database berhasil direset!")

# ============================================
# UPLOAD & PROSES PDF
# ============================================
uploaded_file = st.file_uploader("📤 Upload file PDF", type="pdf")

if "processed" not in st.session_state:
    st.session_state.processed = False

if uploaded_file is not None:
    st.info(f"✅ File berhasil diupload: **{uploaded_file.name}**")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name
    
    if not st.session_state.processed:
        with st.spinner("📖 Memproses dokumen..."):
            reader = PdfReader(temp_path)
            all_text = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    all_text.append({"text": text, "page": page_num + 1})
            
            if not all_text:
                st.error("❌ Tidak ada teks yang bisa diekstrak.")
                st.stop()
            
            st.write(f"📄 Jumlah halaman: **{len(all_text)}**")
            
            documents = []
            for item in all_text:
                chunks_text = split_text_manual(item["text"], chunk_size=1000, overlap=200)
                for chunk in chunks_text:
                    doc = Document(
                        page_content=chunk,
                        metadata={"page": item["page"]}
                    )
                    documents.append(doc)
            
            st.write(f"📦 Jumlah chunks: **{len(documents)}**")
            
            embeddings = OllamaEmbeddings(model=model_choice)
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=embeddings
            )
            st.session_state.vectorstore = vectorstore
            
            # Simpan model untuk digunakan nanti
            st.session_state.llm = Ollama(
                model=model_choice,
                temperature=0.2,
                num_predict=512
            )
            
            st.session_state.processed = True
            os.unlink(temp_path)
            
            st.success("✅ Dokumen siap! Silakan tanyakan sesuatu.")
            st.balloons()
    
    # ============================================
    # TANYA JAWAB
    # ============================================
    st.divider()
    st.subheader("💬 Tanya Jawab")
    
    user_question = st.text_input("✍️ Tulis pertanyaan Anda:", placeholder="Contoh: Apa itu SOLAS?")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🚀 Tanya", type="primary", use_container_width=True)
    
    if ask_button and user_question:
        if "vectorstore" not in st.session_state:
            st.warning("⚠️ Silakan upload PDF terlebih dahulu.")
        else:
            with st.spinner("🤔 Mencari jawaban..."):
                try:
                    answer, docs = ask_question(
                        user_question,
                        st.session_state.vectorstore,
                        st.session_state.llm,
                        k=3
                    )
                    
                    st.subheader("📝 Jawaban:")
                    st.markdown(f"**{answer}**")
                    
                    st.subheader("📖 Referensi:")
                    pages_shown = set()
                    for idx, doc in enumerate(docs, 1):
                        page_num = doc.metadata.get('page', 'Tidak diketahui')
                        if page_num not in pages_shown:
                            pages_shown.add(page_num)
                            with st.expander(f"📄 Halaman {page_num} - Sumber {idx}"):
                                st.text(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Pastikan Ollama berjalan dan model tersedia.")

# ============================================
# PESAN AWAL
# ============================================
else:
    st.info("📌 Silakan upload file PDF untuk memulai.")
    st.markdown("""
    **📚 Contoh file:** SOLAS, MARPOL, COLREG
    """)

with st.sidebar:
    st.divider()
    st.header("📊 Status")
    if uploaded_file is not None:
        st.success(f"✅ File: {uploaded_file.name}")
        if st.session_state.processed:
            st.success("✅ Database siap")
        else:
            st.info("⏳ Proses indexing...")
    else:
        st.warning("⏳ Belum ada file")
    st.caption("⚡ Dibuat dengan Streamlit + Ollama + FAISS")