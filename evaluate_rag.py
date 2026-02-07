from datasets import load_dataset
from rank_bm25 import BM25Okapi
from transformers import T5ForConditionalGeneration, T5Tokenizer
import evaluate
import time

# -------------------
# Load data
# -------------------
corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
documents = [doc['passage'] for doc in corpus]
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

qa_data = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="test")
qa_data = qa_data.select(range(30))  # just first 30 examples for speed

# -------------------
# Load model + tokenizer
# -------------------
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# -------------------
# Setup metrics
# -------------------
# exact_match = evaluate.load("exact_match")
# f1_metric = evaluate.load("f1")
squad_metric = evaluate.load("squad")


# -------------------
# Answer function
# -------------------
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------
# Run RAG over dataset
# -------------------
preds = []
refs = []

start = time.time()

for i, item in enumerate(qa_data):
    question = item['question']
    ground_truth = item['answer']

    # Retrieve
    query_tokens = question.lower().split()
    top_docs = bm25.get_top_n(query_tokens, documents, n=3)
    context = " ".join(top_docs)

    # Generate
    prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}"
    pred = generate_answer(prompt)

    preds.append(pred)
    refs.append(ground_truth)

    print(f"[{i+1}] Q: {question}")
    print(f"    ➤ Pred: {pred}")
    print(f"    ✔ True: {ground_truth}\n")

end = time.time()

# -------------------
# Evaluation
# -------------------
# em = exact_match.compute(predictions=preds, references=refs)
# f1 = f1_metric.compute(predictions=preds, references=refs)
results = squad_metric.compute(
    predictions=[{"id": str(i), "prediction_text": p} for i, p in enumerate(preds)],
    references=[{"id": str(i), "answers": {"text": [r], "answer_start": [0]}} for i, r in enumerate(refs)]
)

avg_time = (end - start) / len(preds)

print("\n Evaluation Results:")
print(f"Exact Match: {results['exact_match']:.2f}")
print(f"F1 Score:    {results['f1']:.2f}")
print(f"Avg time per query: {avg_time:.2f} sec")

