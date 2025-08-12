
import os
from pathlib import Path
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma


# Ensure the vector_db folder exists
VECTOR_DB_FOLDER = "vector_db"
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)


# Create or load the Chroma vector store (can be extended for hybrid search)
def create_or_load_vector_store(filename, chunks, embeddings):
    collection_name = f"{filename}_chroma"
    persist_directory = str(Path(VECTOR_DB_FOLDER) / collection_name)
    if os.path.exists(persist_directory):
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    return vector_store

# Build RAG chain
def build_rag_chain(model):
    prompt = """
        You are an assistant for financial data analysis. Use the retrieved context to answer questions. 
        If you don't know the answer, say so. 
        Question: {question}
        Context: {context}
        Answer:
    """
    prompt_template = ChatPromptTemplate.from_template(prompt)
    llm = ChatOllama(model=model, base_url="http://localhost:11434")

    return prompt_template | llm | StrOutputParser()
