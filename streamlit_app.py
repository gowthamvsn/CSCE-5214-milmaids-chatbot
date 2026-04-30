import os
import math
from pathlib import Path
import streamlit as st

from bots.simple_vector_index import load_index, query_answer, create_index

st.set_page_config(page_title="Mil Maids RAG Chatbot", layout="wide")

st.title("Mil Maids Knowledge Chatbot")
st.markdown(
    "Use this app to ask questions about Mil Maids cleaning services. "
    "The answer is generated from the local knowledge base using OpenAI embeddings + retrieval-aware generation."
)

if st.button("Refresh knowledge index"):
    with st.spinner("Refreshing embeddings and document index..."):
        create_index(refresh=True)
        st.success("Knowledge index refreshed successfully.")

query = st.text_input("Ask a question about Mil Maids services:", value="")

if query:
    with st.spinner("Retrieving relevant knowledge and generating an answer..."):
        docs, embeddings = load_index()
        answer, sources = query_answer(query, docs, embeddings)

    st.subheader("Answer")
    st.write(answer)
