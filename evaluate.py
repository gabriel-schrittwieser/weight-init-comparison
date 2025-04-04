# evaluate.py

import json
import matplotlib.pyplot as plt

# -------------------------------
# Load JSON results
# -------------------------------
with open("results_he.json", "r") as f:
    he_results = json.load(f)

with open("results_orthogonal.json", "r") as f:
    orth_results = json.load(f)

epochs = list(range(1, len(he_results["train_losses"]) + 1))

# -------------------------------
# Plot Training Loss
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, he_results["train_losses"], label="He Initialization", linewidth=2)
plt.plot(epochs, orth_results["train_losses"], label="Orthogonal Initialization", linewidth=2)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_comparison.png")
plt.show()

# -------------------------------
# Plot Test Accuracy
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, he_results["test_accuracies"], label="He Initialization", linewidth=2)
plt.plot(epochs, orth_results["test_accuracies"], label="Orthogonal Initialization", linewidth=2)
plt.title("Test Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("test_accuracy_comparison.png")
plt.show()
