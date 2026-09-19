# SmartPaper-RAG-Studio
<div align="center">

### *Autonomous Exam Question Paper Synthesizer via Retrieval-Augmented Generation*

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-E25A2A?style=for-the-badge)

<p align="center">
  <b>Transform course syllabi and lecture notes into custom examination papers in seconds.</b>
</p>

</div>

---

## 🌟 Overview

**RAG-ExamiNation-AI** is a faculty-focused application built with **Streamlit**, **LangChain**, **ChromaDB**, and **OpenAI**. It processes dense course syllabus or lecture note PDFs, creates vector embeddings using `text-embedding-3-small`, and uses Retrieval-Augmented Generation (RAG) to generate balanced, structured question papers aligned with specified academic parameters.

---

## ✨ Key Features

- 📑 **Zero-Context Hallucination (Strict RAG)**: Questions are derived *only* from the uploaded document context.
- ⚡ **Cached Vector Indexing**: Fast indexing with `st.cache_resource` to avoid repeated vector calculations.
- 🎛️ **Custom Exam Parameters**:
  - **Exam Duration**: Configurable duration string (e.g., *2 Hours*, *180 Minutes*).
  - **Total Marks**: Flexible weightage system (10 to 200 marks).
  - **Difficulty Scaling**: *Easy*, *Medium*, or *Hard* cognitive demand options.
  - **Question Count**: Specify total items to generate.
- 🎨 **Minimal Dark UI**: Sidebar-free interface with custom typography and metric dashboards.
- 📄 **One-Click PDF Export**: Direct client-side PDF generation using **ReportLab**.

---

## 🛠️ Architecture Pipeline
