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

pdf_path="Anushka Dabhade_CV.pdf"
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
You are a PDF question-answering assistant.

Answer the question using ONLY the
information provided in the context.

If the answer is not available in the
context, say:

"I could not find the answer in the PDF."

Context:
{context}

Question:
{query}

Answer:
"""

response=llm.invoke(prompt)
print(response.content)