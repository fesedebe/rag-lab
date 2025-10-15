# Baseline RAG implementation using LangChain and local FLAN-T5

import os
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import HuggingFacePipeline
from transformers import pipeline
from langchain.chains import RetrievalQA

# Load text files into LangChain docs
def load_docs(folder="data"):
    docs = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            with open(os.path.join(folder, fname), "r") as f:
                text = f.read()
                docs.append(Document(page_content=text, metadata={"source": fname}))
    return docs

def build_rag():
    docs = load_docs()

    # Create retriever - embeddings + vector store
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embedding_model)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Initialize generator (local FLAN-T5)
    generator_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_length=256
    )

    llm = HuggingFacePipeline(pipeline=generator_pipeline)

    # Create RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return rag_chain
#rag_chain = build_rag()
