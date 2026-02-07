import matplotlib.pyplot as plt

methods = ['BM25', 'SBERT', 'DPR']
f1_scores = [48, 62, 74]
query_times = [8, 12, 15]  # ms

fig, ax1 = plt.subplots()

# Bar chart for F1 scores
ax1.bar(methods, f1_scores, width=0.4, label='F1 Score (%)')
ax1.set_ylabel('F1 Score (%)')
ax1.set_ylim(0, 100)
ax1.set_title('Retrieval Model Comparison: Accuracy vs Speed')

# Line chart for query times (plotted on a secondary axis)
ax2 = ax1.twinx()
ax2.plot(methods, query_times, 'o--', color='red', label='Query Time (ms)')
ax2.set_ylabel('Query Time (ms)')
ax2.set_ylim(0, 20)

# Combined legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')

plt.tight_layout()
plt.savefig("./retriever_comparison.png")
plt.show()
