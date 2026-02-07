from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import evaluate
import time

# -------------------
# Load data
# -------------------
qa_data = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="test")
qa_data = qa_data.select(range(30))  # first 30 examples

# -------------------
# Load model + tokenizer
# -------------------
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# -------------------
# Setup metrics
# -------------------
squad_metric = evaluate.load("squad")

# -------------------
# Answer function
# -------------------
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------
# Run baseline generation
# -------------------
preds = []
refs = []

start = time.time()

for i, item in enumerate(qa_data):
    question = item['question']
    ground_truth = item['answer']

    # No retrieval — direct question only
    prompt = f"Answer this question:\n{question}"
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
results = squad_metric.compute(
    predictions=[{"id": str(i), "prediction_text": p} for i, p in enumerate(preds)],
    references=[{"id": str(i), "answers": {"text": [r], "answer_start": [0]}} for i, r in enumerate(refs)]
)
avg_time = (end - start) / len(preds)

print("\n📊 Baseline Results (No Retrieval):")
print(f"Exact Match: {results['exact_match']:.2f}")
print(f"F1 Score:    {results['f1']:.2f}")
print(f"Avg time per query: {avg_time:.2f} sec")
