from datasets import load_dataset

corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")

# Print a few passages
for i in range(3):
    print(f"\n--- Document {i+1} ---")
    print(corpus[i]['passage'])
