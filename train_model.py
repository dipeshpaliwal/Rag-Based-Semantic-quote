# train_model.py

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import faiss
import re

# Load and preprocess dataset
dataset = load_dataset("Abirate/english_quotes")
df = pd.DataFrame(dataset['train'])
df['quote'] = df['quote'].fillna('').apply(lambda x: re.sub(r'[^a-zA-Z0-9.,\\s]', '', x.lower()))
df = df[df['quote'].str.strip() != '']

# Create training examples
examples = [InputExample(texts=[row['quote'], row['quote']]) for _, row in df.iterrows()]
dataloader = DataLoader(examples, shuffle=True, batch_size=32)

# Train model
model = SentenceTransformer("all-MiniLM-L6-v2")
loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(dataloader, loss)], epochs=1, warmup_steps=100)
model.save("fine_tuned_quote_model")

# Index and save FAISS
embeddings = model.encode(df['quote'].tolist(), show_progress_bar=True)
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(np.array(embeddings))
faiss.write_index(faiss_index, "quotes.faiss")

# Save dataframe for inference
df.to_csv("quotes_data.csv", index=False)
