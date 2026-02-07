import matplotlib.pyplot as plt

# Data for F1 vs Latency comparison
methods = ['Baseline', 'BM25', 'SBERT (Dense)']
f1_scores = [11.85, 29.05, 34.62]
latency = [0.08, 0.21, 0.49]

# Create the plot
plt.figure(figsize=(6, 4))
plt.scatter(latency, f1_scores, color='black')

# Annotate points
for i, method in enumerate(methods):
    plt.annotate(method, (latency[i] + 0.01, f1_scores[i]))

plt.title("F1 Score vs. Query Latency")
plt.xlabel("Time per Query (seconds)")
plt.ylabel("F1 Score")
plt.grid(True)
plt.tight_layout()

# Save the figure
fig_path = "./f1_latency_comparison.png"
plt.savefig(fig_path)

