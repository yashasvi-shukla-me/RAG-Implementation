from datasets import load_dataset
from rank_bm25 import BM25Okapi
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import time

# -------------------
# Load data
# -------------------
corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
documents = [doc['passage'] for doc in corpus]
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

qa_data = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="test")
qa_data = qa_data.select(range(30))  # first 30 for speed

# -------------------
# Load model + tokenizer
# -------------------
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# -------------------
# Answer generation
# -------------------
preds = []
refs = []
questions = []
contexts = []

start = time.time()

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

for i, item in enumerate(qa_data):
    question = item['question']
    ground_truth = item['answer']

    query_tokens = question.lower().split()
    top_docs = bm25.get_top_n(query_tokens, documents, n=3)
    context = " ".join(top_docs)

    prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}"
    answer = generate_answer(prompt)

    preds.append(answer)
    refs.append(ground_truth)
    questions.append(question)
    contexts.append(top_docs)  # keep as list of chunks

    print(f"[{i+1}] Q: {question}")
    print(f"    ➤ Pred: {answer}")
    print(f"    ✔ True: {ground_truth}\n")

end = time.time()
avg_time = (end - start) / len(preds)
print(f"\nAvg time per query: {avg_time:.2f} sec")

# -------------------
# Save RAGAS input
# -------------------
ragas_data = []
for i in range(len(questions)):
    ragas_data.append({
        "question": questions[i],
        "answer": refs[i],
        "prediction": preds[i],
        "retrieved_contexts": contexts[i],
        "reference": refs[i]
    })

with open("ragas_input.json", "w") as f:
    json.dump(ragas_data, f, indent=2)

print("RAGAS input saved to ragas_input.json ")
