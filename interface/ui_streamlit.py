# Streamlit UI to query RAG API

import streamlit as st
import requests

API_URL = "http://localhost:8000/query"  # FastAPI backend endpoint

st.set_page_config(page_title="RAG Explorer", layout="wide")
st.title("RAG Explorer")

# Input field for user query
query = st.text_input("Enter your question:")

# Send query to FastAPI backend and display results
if st.button("Run") and query:
    with st.spinner("Querying RAG backend..."):
        try:
            response = requests.post(API_URL, json={"query": query})
            response.raise_for_status()
            data = response.json()

            st.subheader("Generated Answer")
            st.success(data.get("answer", "No answer returned."))

            st.subheader("Retrieved Contexts")
            for src in data.get("sources", []):
                with st.expander(src.get("source", "Unknown Source")):
                    st.write(src.get("content", ""))
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
