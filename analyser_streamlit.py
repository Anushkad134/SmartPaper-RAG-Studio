import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="PDF Assistant | Naive RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .main-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
    }
    .answer-box {
        background-color: #1E293B;
        border-left: 4px solid #38BDF8;
        border-radius: 6px;
        padding: 16px;
        margin-top: 15px;
        font-size: 1.05rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# Core Processing Functions
def process_pdf(file):
    """Extracts text, splits into chunks, and initializes vector store."""
    reader = PdfReader(file)
    raw_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            raw_text += page_text + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="pdf_qa_session"
    )
    
    return vectorstore, len(reader.pages), len(chunks)


# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/books.png", width=64)
    st.header("Document Setup")
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    
    st.divider()
    st.markdown("### Configuration")
    st.caption("• Embedding Model: `text-embedding-3-small`")
    st.caption("• LLM: `gpt-4.1-mini` or `gpt-4o-mini`")
    st.caption("• Chunk Size: 500 characters")


# Main Interface
st.markdown('<div class="main-title">📚 Document Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Retrieval-Augmented Generation (Naive RAG) Pipeline</div>', unsafe_allow_html=True)

if uploaded_file:
    # Use session state to avoid re-embedding on every user interaction
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Processing document & building vector index..."):
            vectorstore, page_count, chunk_count = process_pdf(uploaded_file)
            st.session_state.vectorstore = vectorstore
            st.session_state.page_count = page_count
            st.session_state.chunk_count = chunk_count
            st.session_state.current_file = uploaded_file.name

    # Display Document Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{st.session_state.page_count}</div>
                <div class="metric-label">Pages Extracted</div>
            </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{st.session_state.chunk_count}</div>
                <div class="metric-label">Vector Chunks</div>
            </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown('''
            <div class="metric-card">
                <div class="metric-value">Active</div>
                <div class="metric-label">ChromaDB Status</div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")

    # Q&A Interface
    query = st.text_input("Ask a question about the PDF:", placeholder="e.g., What are the main key points in Section 2?")

    if query:
        with st.spinner("Searching document & generating answer..."):
            vectorstore = st.session_state.vectorstore
            docs = vectorstore.similarity_search(query, k=5)
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)

            prompt = f"""
You are an expert document analysis assistant.

Answer the user's question using ONLY the provided context.
If the exact answer is not contained within the context, state clearly:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{query}

Answer:
"""
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            response = llm.invoke(prompt)

        st.markdown("### Answer")
        st.markdown(f'<div class="answer-box">{response.content}</div>', unsafe_allow_html=True)

        # Context Viewer
        with st.expander("🔍 View Retrieved Context Chunks (Top 5 Matches)"):
            for idx, doc in enumerate(docs, start=1):
                st.markdown(f"**Chunk {idx}:**")
                st.info(doc.page_content)
else:
    st.info("👈 Please upload a PDF document in the sidebar to begin analysis.")