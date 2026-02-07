# from datasets import load_dataset
# from rank_bm25 import BM25Okapi

# # Load the document corpus
# corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
# documents = [doc['passage'] for doc in corpus]

# # Tokenize each document (simple lowercase split)
# tokenized_docs = [doc.lower().split() for doc in documents]

# # Build BM25 index
# bm25 = BM25Okapi(tokenized_docs)

# # Test a sample query
# query = "Who invented the telephone?"
# tokenized_query = query.lower().split()
# scores = bm25.get_scores(tokenized_query)

# # Get top 3 results
# top_n = bm25.get_top_n(tokenized_query, documents, n=3)

# print("\nTop 3 Retrieved Documents:")
# for i, doc in enumerate(top_n):
#     print(f"\n--- Doc {i+1} ---\n{doc[:300]}...")  # print first 300 chars


# -------------------
# Dense retriever using BM25
# -------------------
# we will use the rank_bm25 library to implement a simple BM25 retriever. This will serve as our "dense" retriever for demonstration purposes, even though BM25 is technically a sparse retrieval method. The focus here is on showing how to integrate a retriever into the RAG pipeline.
# The code will load a corpus of documents, build a BM25 index, and then retrieve relevant passages based on a sample query. This will be used later in the RAG pipeline to provide context for the generator model.
# Note: In a real implementation, you would likely use a more sophisticated dense retriever (like DPR or a bi-encoder), but BM25 serves as a simple stand-in for demonstration purposes.

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import evaluate
import time

# Load corpus
corpus_dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
documents = [doc["passage"] for doc in corpus_dataset]

# Dense Encoder
encoder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = encoder.encode(documents, convert_to_numpy=True, show_progress_bar=True)

# Build FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Load QA test data
qa_data = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="test")
qa_data = qa_data.select(range(30))

# Load Generator
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# Retrieval + Generation
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

preds, refs = [], []
start = time.time()

for i, item in enumerate(qa_data):
    query = item["question"]
    ground_truth = item["answer"]

    # Embed query
    query_embedding = encoder.encode([query], convert_to_numpy=True)
    _, top_indices = index.search(query_embedding, k=3)
    top_passages = [documents[idx] for idx in top_indices[0]]
    context = " ".join(top_passages)

    # Prompt + Generate
    prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {query}"
    pred = generate_answer(prompt)

    preds.append(pred)
    refs.append(ground_truth)

    print(f"[{i+1}] Q: {query}")
    print(f"    ➤ Pred: {pred}")
    print(f"    ✔ True: {ground_truth}\n")

end = time.time()

# Evaluate
squad_metric = evaluate.load("squad")
results = squad_metric.compute(
    predictions=[{"id": str(i), "prediction_text": p} for i, p in enumerate(preds)],
    references=[{"id": str(i), "answers": {"text": [r], "answer_start": [0]}} for i, r in enumerate(refs)]
)

print("\n Evaluation Results (Dense Retriever):")
print(f"Exact Match: {results['exact_match']:.2f}")
print(f"F1 Score:    {results['f1']:.2f}")
print(f"Avg time per query: {(end - start)/len(preds):.2f} sec")
