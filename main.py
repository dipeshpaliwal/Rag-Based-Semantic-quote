# main.py

import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load model, index, and data
model = SentenceTransformer("fine_tuned_quote_model")
index = faiss.read_index("quotes.faiss")
df = pd.read_csv("quotes_data.csv")

qa_pipeline = pipeline("text-generation", model="gpt2")  # Replace with better model if needed

def rag_query(query, top_k=5):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)
    results = df.iloc[indices[0]]

    context = "\n".join(
        f"\"{row['quote']}\" — {row['author']} (Tags: {row['tags']})"
        for _, row in results.iterrows()
    )
    prompt = f"Use the following quotes to answer: '{query}'\n\n{context}"
    response = qa_pipeline(prompt, max_length=256, do_sample=True)[0]["generated_text"]
    return results.to_dict(orient="records"), response

# Streamlit UI
st.title("📚 Semantic Quote RAG Search")
query = st.text_input("Ask your quote-related query:")

if st.button("Search"):
    quotes, response = rag_query(query)
    for q in quotes:
        st.markdown(f"> {q['quote']}\n\n**— {q['author']}**  \nTags: `{q['tags']}`")
    st.subheader("💬 Generated Answer")
    st.write(response)
