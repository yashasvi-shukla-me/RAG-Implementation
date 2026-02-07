from datasets import load_dataset

# Load the question-answer split
dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="test")

# Print first 3 QA pairs
for i in range(3):
    print(f"\n--- Example {i+1} ---")
    print("Question:", dataset[i]['question'])
    print("Answer:", dataset[i]['answer'])
