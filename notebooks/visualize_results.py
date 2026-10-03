import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


y_true = [0]*117 + [1]*67  # Healthy=0, Diseased=1
y_pred = [0]*104 + [1]*13 + [0]*16 + [1]*51  

labels = ["Healthy", "Diseased"]
cm = np.array([[104, 13], [16, 51]])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Confusion Matrix Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=axes[0], cmap="Blues", values_format='d')
axes[0].set_title("Confusion Matrix (Test Set)", fontsize=14, fontweight='bold')

# 2. Metrics Bar Chart
metrics = {
    'Accuracy': 84.24,
    'Healthy Recall': 88.89,
    'Diseased Precision': 79.69,
    'F1-Score': 82.81
}
bars = axes[1].bar(metrics.keys(), metrics.values(), color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
axes[1].set_ylim(0, 100)
axes[1].set_ylabel("Percentage (%)")
axes[1].set_title("Key Performance Metrics", fontsize=14, fontweight='bold')
for bar, val in zip(bars, metrics.values()):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, 
                 f'{val}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/test_results.png", dpi=200, bbox_inches='tight')
plt.show()
print("✅ Results visualization saved to outputs/test_results.png")