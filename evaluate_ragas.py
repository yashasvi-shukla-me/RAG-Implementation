import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.llms import HuggingFaceLLM
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
    AnswerCorrectness
)

# Load RAGAS input
with open("ragas_input.json", "r") as f:
    data = json.load(f)

ragas_ds = Dataset.from_list(data)

# Use your Hugging Face token (assumed to be exported to HF_TOKEN)
hf_token = os.environ.get("HF_TOKEN")

# Define the HuggingFaceLLM properly (use a small but capable model to stay fast)
llm = HuggingFaceLLM(model="google/flan-t5-base", token=hf_token)

# Initialize metrics with proper LLM wrapper
metrics = [
    Faithfulness(llm=llm),
    AnswerRelevancy(),
    ContextRecall(),
    ContextPrecision(),
    AnswerCorrectness(llm=llm)
]

# Run RAGAS evaluation
result = evaluate(ragas_ds, metrics=metrics)

# Display results
print("\n RAGAS Evaluation Results:")
for metric, score in result.items():
    print(f"{metric}: {score:.2f}")
