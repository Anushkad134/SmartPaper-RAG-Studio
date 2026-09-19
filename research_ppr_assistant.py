#"pdf based q and a chatbot" naive rag application
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
#api_key
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

pdf_path="Research 1.pdf"
reader=PdfReader(pdf_path)
text=" "
for page in reader.pages:
    page_text=page.extract_text()
    if page_text:
        text+=page_text

print("pdf is loaded successfully")

#chunking 

splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
chunks=splitter.split_text(text)
print(len(chunks),"chunks are created")

#vector embedding

embeddings=OpenAIEmbeddings(model="text-embedding-3-small")

#vector db
vectorstore=Chroma.from_texts(texts=chunks,embedding=embeddings,collection_name="pdfqa")

#llm model
llm=ChatOpenAI(model="gpt-4.1-mini")

#user query
query=input("Enter your question")

#search chroma_db
docs=vectorstore.similarity_search(query,k=5)

# chunk 100, 299,276,400,489 - merge

context="\n".join(doc.page_content for doc in docs)

prompt = f"""
You are a Research Paper RAG Assistant.

Your job is to answer the user's questions based only on the research paper content provided in the retrieved context.

Follow these rules:

1. Use the retrieved context as your primary and only source of information.
2. Do not make up facts, references, results, numbers, or conclusions that are not present in the context.
3. If the answer is not available in the retrieved context, clearly say:
   "I could not find this information in the provided research paper."
4. Give clear, concise, and beginner-friendly answers.
5. When explaining technical concepts, use simple language and short examples when helpful.
6. If the user asks about a specific section, method, experiment, result, dataset, or conclusion, answer specifically from the retrieved content.
7. Preserve important technical terms, names, numbers, percentages, and experimental results exactly when they are available in the context.
8. Do not use outside knowledge unless the user explicitly asks for a general explanation beyond the paper.
9. If the retrieved context contains conflicting information, mention the conflict instead of choosing one without explanation.
10. Do not mention "retrieved context", "vector database", "embeddings", or internal RAG processes unless the user asks about how the assistant works.

Retrieved Research Paper Context:
{context}

User Question:
{query}

Answer:
"""

response=llm.invoke(prompt)
print(response.content)