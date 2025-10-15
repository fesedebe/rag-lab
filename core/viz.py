# Visulaize retrieval with streamlit

import streamlit as st
import requests

st.title("RAG Explorer")

query = st.text_input("Enter your question:")
if query:
    with st.spinner("Querying RAG backend..."):
        response = requests.post("http://localhost:8000/query", json={"query": query})
        data = response.json()

    st.subheader("Generated Answer")
    st.success(data["answer"])

    st.subheader("Retrieved Contexts")
    for src in data["sources"]:
        with st.expander(src["source"]):
            st.write(src["content"])
