import os
import io
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

# Imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AI Question Paper Generator",
    page_icon="📝",
    layout="wide"
)

# Custom Styling (Pure Dark Theme + Custom Fonts, Sidebar Hidden)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

    /* Global Typography Reset */
    html, body, [class*="css"], .stMarkdown, p, div, span, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Force App Background to Dark Slate/Black */
    .stApp, [data-testid="stMain"], [data-testid="stHeader"] {
        background-color: #0B0F17 !important;
        color: #F1F5F9 !important;
    }

    /* Hide Sidebar Element Entirely */
    [data-testid="stSidebar"] {
        display: none !important;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        max-width: 1000px;
    }

    /* Typography Styling */
    h1, h2, h3, .header-title {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
    }

    .header-title {
        font-size: 2.5rem;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }

    .header-sub {
        font-size: 1.05rem;
        color: #94A3B8 !important;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    .section-title {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1.25rem;
        font-weight: 700;
        color: #38BDF8;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    /* Dark File Uploader Box */
    [data-testid="stFileUploadDropzone"] {
        background-color: #161E2E !important;
        border: 1px dashed #334155 !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stFileUploadDropzone"] * {
        color: #94A3B8 !important;
    }

    /* Dark Metric Cards */
    .metric-card {
        background-color: #161E2E;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 500;
        margin-top: 2px;
    }

    /* Primary Dark Button */
    div.stButton > button:first-child {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100%;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #0369A1 !important;
    }

    /* Download Button Styling */
    div.stDownloadButton > button {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100%;
        transition: all 0.2s ease;
    }
    div.stDownloadButton > button:hover {
        background-color: #059669 !important;
    }

    /* Question Paper Render Container */
    .paper-box {
        background-color: #161E2E;
        border: 1px solid #1E293B;
        border-left: 4px solid #38BDF8;
        border-radius: 10px;
        padding: 28px;
        margin-top: 20px;
        color: #F8FAFC;
        font-size: 1.05rem;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def build_vector_store(file_bytes, file_name):
    """Parses PDF and indexes vector store into ChromaDB with caching."""
    reader = PdfReader(file_bytes)
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
        collection_name=f"paper_gen_{hash(file_name)}"
    )
    
    return vectorstore, len(reader.pages), len(chunks)


def generate_pdf_bytes(paper_text):
    """Converts markdown/text paper into a styled PDF in memory."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    normal_style.fontSize = 11
    normal_style.leading = 15
    normal_style.textColor = colors.HexColor('#0F172A')

    heading_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0284C7'),
        spaceAfter=10
    )

    story = []
    lines = paper_text.split('\n')

    for line in lines:
        clean_line = line.replace('**', '').replace('###', '').replace('##', '').replace('#', '').strip()
        if not clean_line:
            story.append(Spacer(1, 8))
            continue
            
        if line.startswith('#') or line.startswith('**'):
            story.append(Paragraph(clean_line, heading_style))
        else:
            story.append(Paragraph(clean_line, normal_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# App Header
st.markdown('<div class="header-title">📝 AI Question Paper Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Upload syllabus or lecture notes to automatically synthesize a custom examination paper via RAG</div>', unsafe_allow_html=True)

# Document Upload Section
st.markdown('<div class="section-title">1. Upload Course Material</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a Syllabus or Lecture Notes PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing document & generating embeddings..."):
        vectorstore, num_pages, num_chunks = build_vector_store(uploaded_file, uploaded_file.name)

    # Document Stats Bar
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{num_pages}</div>
                <div class="metric-label">Pages Processed</div>
            </div>
        ''', unsafe_allow_html=True)
    with col_b:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{num_chunks}</div>
                <div class="metric-label">Chunks Stored</div>
            </div>
        ''', unsafe_allow_html=True)
    with col_c:
        st.markdown('''
            <div class="metric-card">
                <div class="metric-value">Active</div>
                <div class="metric-label">Vector Store Status</div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")

    # Examination Parameters Form
    st.markdown('<div class="section-title">2. Configure Examination Parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        duration = st.text_input("Exam Duration", value="2 Hours")
        total_marks = st.number_input("Total Marks", min_value=10, max_value=200, value=50, step=5)
    
    with col2:
        difficulty = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])
        num_questions = st.number_input("Number of Questions", min_value=1, max_value=30, value=5, step=1)

    st.markdown("<br>", unsafe_allow_html=True)

    # Generate Action
    if st.button("Generate Question Paper"):
        with st.spinner("Retrieving relevant topics & synthesizing question paper..."):
            docs = vectorstore.similarity_search("key concepts syllabus main topics summary", k=10)
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)

            prompt = f"""
You are an expert academic paper setter.

Task: Generate a formal examination question paper strictly based on the provided context below.

Examination Constraints:
- Duration: {duration}
- Total Marks: {total_marks}
- Difficulty Level: {difficulty}
- Number of Questions: {num_questions}

Requirements:
1. Include a formal header (Course Title/Placeholder, Duration, Total Marks, General Instructions).
2. Distribute the total marks ({total_marks}) logically across all {num_questions} questions.
3. Ensure the question complexity strictly aligns with the chosen difficulty level ({difficulty}).
4. Use ONLY information directly present in the context.

Context:
{context}

Format the response using clear Markdown formatting suitable for display.
"""
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
            response = llm.invoke(prompt)
            
            # Save generated content to session state
            st.session_state["generated_paper"] = response.content
            st.session_state["context_docs"] = docs

    # Display Generated Paper & PDF Download Option
    if "generated_paper" in st.session_state:
        st.markdown('<div class="section-title">3. Generated Examination Paper</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="paper-box">{st.session_state["generated_paper"]}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Download Button
        pdf_bytes = generate_pdf_bytes(st.session_state["generated_paper"])
        st.download_button(
            label="📄 Download Question Paper as PDF",
            data=pdf_bytes,
            file_name="question_paper.pdf",
            mime="application/pdf"
        )

        # Context Inspector
        with st.expander("🔍 View Context Chunks Used"):
            for idx, doc in enumerate(st.session_state["context_docs"], start=1):
                st.markdown(f"**Chunk {idx}:**")
                st.write(doc.page_content)
                if idx < len(st.session_state["context_docs"]):
                    st.divider()

else:
    st.info("Upload a syllabus or notes PDF above to begin configuring the question paper.")