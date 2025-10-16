# Baseline RAG implementation using LangChain and local FLAN-T5

import os
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import HuggingFacePipeline
from transformers import pipeline
from langchain.chains import RetrievalQA

class BaselineRAG:
    def __init__(self, data_folder: str = "data", model_name: str = "google/flan-t5-base"):
        self.data_folder = data_folder
        self.model_name = model_name
        self.chain = self._build_chain()

    # Load text files into LangChain docs
    def _load_docs(self):
        docs = []
        for fname in os.listdir(self.data_folder):
            if fname.endswith(".txt"):
                with open(os.path.join(self.data_folder, fname), "r", encoding="utf-8") as f:
                    text = f.read()
                    docs.append(Document(page_content=text, metadata={"source": fname}))
        if not docs:
            raise ValueError(f"No .txt files found in {self.data_folder}")
        return docs

    # Build RAG chain: embeddings, retriever, generator
    def _build_chain(self):
        docs = self._load_docs()

        # Create embeddings and store in FAISS vector store
        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embed_model)
        retriever = db.as_retriever(search_kwargs={"k": 3})

        # Initialize generator (local FLAN-T5)
        gen_pipeline = pipeline(
            "text2text-generation",
            model=self.model_name,
            tokenizer=self.model_name,
            max_length=256,
        )
        llm = HuggingFacePipeline(pipeline=gen_pipeline)

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        return rag_chain

    # Query RAG chain, return answer + sources
    def run(self, query: str):
        res = self.chain(query)
        return {
            "query": query,
            "answer": res["result"],
            "sources": res["source_documents"],
        }
