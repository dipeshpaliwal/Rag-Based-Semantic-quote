# 📚 Task 2 – RAG-Based Semantic Quote Retrieval & Structured QA

## 🎯 Objective

Develop a **Retrieval-Augmented Generation (RAG)** system that:
- Retrieves semantically relevant quotes from a corpus
- Uses a language model to generate structured answers
- Provides an interactive **Streamlit** interface for querying quotes

---

## 🧠 Dataset

- Source: [Hugging Face – Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes)
- Fields:
  - `quote`: The quote text
  - `author`: Author name
  - `tags`: Tags or themes

---

## ⚙️ System Architecture

### ✅ Steps Involved:

1. **Data Preparation**  
   - Load dataset using Hugging Face `datasets` library  
   - Clean, lowercase, remove missing values

2. **Model Fine-Tuning**  
   - Use `all-MiniLM-L6-v2` from `sentence-transformers`  
   - Fine-tune using triplet loss on (query, positive, negative) format  
   - Save the trained model

3. **Vector Indexing with FAISS**  
   - Encode all quotes using the fine-tuned model  
   - Store in FAISS index for fast retrieval

4. **RAG Pipeline**  
   - On user query:
     - Convert to embedding → retrieve top-k matches via FAISS
     - Pass results as context to a small generative model (e.g., GPT-2)
     - Generate answer based on context and return quote, author, tags

5. **Streamlit Frontend**  
   - User enters natural query
   - UI shows:
     - Retrieved quotes
     - Authors and tags
     - Final answer
     - JSON-style structured view

---

## 📦 Project Structure

```text
project/
├── train_model.py            # Fine-tune sentence embedding model
├── main.py                   # Streamlit app for semantic quote search
├── quotes_data.csv           # Cleaned quote data
├── quotes.faiss              # FAISS index for semantic retrieval
├── fine_tuned_quote_model/   # Saved model directory
├── requirements.txt
└── README.md
