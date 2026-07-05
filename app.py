import streamlit as st
import tempfile
import os
import re
from pypdf import PdfReader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document

# ============================================
# FUNGSI SPLIT MANUAL (Tanpa LangChain)
# ============================================
def split_text_manual(text, chunk_size=1000, overlap=200):
    """Split teks menjadi potongan-potongan dengan ukuran tertentu."""
    if not text.strip():
        return []
    
    # Gunakan regex untuk split berdasarkan kalimat atau paragraf
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Jika menambahkan kalimat ini melebihi chunk_size, simpan chunk saat ini
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Overlap: ambil sebagian dari akhir chunk sebelumnya
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Tambahkan sisa teks
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


# ============================================
# JUDUL APLIKASI
# ============================================
st.set_page_config(page_title="Marine's QA", layout="wide")
st.title("⚓ Marine's Question Answer")
st.caption("Upload e-book maritim (SOLAS, MARPOL, COLREG) dan tanyakan apapun!")

# ============================================
# SIDEBAR - Pengaturan Model
# ============================================
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
    - Model harus sudah di-pull (`ollama pull nama_model`)
    """)
    
    if st.button("🗑️ Reset Database"):
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        if "qa_chain" in st.session_state:
            del st.session_state.qa_chain
        st.success("Database berhasil direset!")

# ============================================
# 1. UPLOAD FILE PDF
# ============================================
uploaded_file = st.file_uploader("📤 Upload file PDF", type="pdf")

if "processed" not in st.session_state:
    st.session_state.processed = False

# ============================================
# 2. PROSES PDF & BUILD VECTOR DATABASE (FAISS)
# ============================================
if uploaded_file is not None:
    st.info(f"✅ File berhasil diupload: **{uploaded_file.name}**")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name
    
    if not st.session_state.processed:
        with st.spinner("📖 Memproses dokumen... (ini mungkin butuh waktu 1-2 menit untuk file besar)"):
            
            # ---- A. EKSTRAKSI TEKS DARI PDF ----
            reader = PdfReader(temp_path)
            all_text = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    all_text.append({
                        "text": text,
                        "page": page_num + 1
                    })
            
            if not all_text:
                st.error("❌ Tidak ada teks yang bisa diekstrak dari PDF ini. Pastikan PDF bukan hasil scan/gambar.")
                st.stop()
            
            st.write(f"📄 Jumlah halaman dengan teks: **{len(all_text)}**")
            
            # ---- B. BUAT DOCUMENTS UNTUK LANGCHAIN ----
            documents = []
            for item in all_text:
                # Split teks per halaman menjadi chunks
                chunks_text = split_text_manual(item["text"], chunk_size=1000, overlap=200)
                for chunk in chunks_text:
                    doc = Document(
                        page_content=chunk,
                        metadata={"page": item["page"]}
                    )
                    documents.append(doc)
            
            st.write(f"📦 Jumlah potongan teks (chunks): **{len(documents)}**")
            
            # ---- C. BUAT EMBEDDINGS & VECTOR DATABASE (FAISS) ----
            embeddings = OllamaEmbeddings(model=model_choice)
            
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=embeddings
            )
            
            st.session_state.vectorstore = vectorstore
            
            # ---- D. BUAT QA CHAIN ----
            llm = Ollama(
                model=model_choice,
                temperature=0.2,
                num_predict=512
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            st.session_state.qa_chain = qa_chain
            st.session_state.processed = True
            
            os.unlink(temp_path)
            
            st.success("✅ Dokumen siap! Silakan tanyakan sesuatu.")
            st.balloons()
    
    # ============================================
    # 3. FITUR TANYA JAWAB
    # ============================================
    st.divider()
    st.subheader("💬 Tanya Jawab dengan AI")
    
    user_question = st.text_input("✍️ Tulis pertanyaan Anda di sini:", placeholder="Contoh: Apa itu SOLAS?")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🚀 Tanya", type="primary", use_container_width=True)
    
    if ask_button or (user_question and st.session_state.get("auto_ask", False)):
        if not user_question:
            st.warning("⚠️ Silakan tulis pertanyaan terlebih dahulu.")
        elif "qa_chain" not in st.session_state:
            st.warning("⚠️ Silakan upload dan proses file PDF terlebih dahulu.")
        else:
            with st.spinner("🤔 Mencari jawaban..."):
                try:
                    result = st.session_state.qa_chain({"query": user_question})
                    answer = result['result']
                    source_docs = result['source_documents']
                    
                    st.subheader("📝 Jawaban:")
                    st.markdown(f"**{answer}**")
                    
                    st.subheader("📖 Referensi (dari halaman):")
                    
                    pages_shown = set()
                    for idx, doc in enumerate(source_docs, 1):
                        page_num = doc.metadata.get('page', 'Tidak diketahui')
                        if page_num not in pages_shown:
                            pages_shown.add(page_num)
                            with st.expander(f"📄 Halaman {page_num} - Cuplikan {idx}"):
                                text_preview = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                                st.text(text_preview)
                    
                    with st.expander("🔍 Lihat semua sumber yang digunakan"):
                        for idx, doc in enumerate(source_docs, 1):
                            page_num = doc.metadata.get('page', 'Tidak diketahui')
                            st.markdown(f"**Sumber {idx} - Halaman {page_num}:**")
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.divider()
                            
                except Exception as e:
                    st.error(f"❌ Terjadi error: {str(e)}")
                    st.info("💡 Pastikan Ollama berjalan dan model yang dipilih sudah di-pull.")

# ============================================
# 4. PESAN UNTUK PENGGUNA (jika belum upload)
# ============================================
else:
    st.info("📌 Silakan upload file PDF untuk memulai.")
    st.markdown("""
    **📚 Contoh file yang bisa diupload:**
    - SOLAS Convention
    - MARPOL Annex
    - COLREG Rules
    - Manual ECDIS / Navigasi lainnya
    """)
    
    st.markdown("""
    ---
    ### 🔧 Prasyarat:
    1. Pastikan **Ollama** berjalan di background
    2. Pastikan model AI sudah di-pull:
       ```bash
       ollama pull qwen2.5:3b
       ```
    """)