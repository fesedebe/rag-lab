
# Load text files into LangChain documents
import os
from langchain.docstore.document import Document

def load_docs(folder="data"):
    docs = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            with open(os.path.join(folder, fname), "r") as f:
                text = f.read()
                docs.append(Document(page_content=text, metadata={"source": fname}))
    return docs

docs = load_docs()
print(f"Loaded {len(docs)} documents.")

# Create retriever - embeddings + vector store
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Initialize the generator (local FLAN-T5)
from langchain import HuggingFacePipeline
from transformers import pipeline

generator_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_length=256
)

llm = HuggingFacePipeline(pipeline=generator_pipeline)
print("Initialized FLAN-T5 generator.")
