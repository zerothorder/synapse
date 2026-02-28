#This particular file is generated using Claude

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load data
preds = pd.read_csv("predictions.csv")
log   = pd.read_csv("training_log.csv")

correct = (preds["true_label"] == preds["predicted"]).sum()
total   = len(preds)
accuracy = correct / total * 100

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0f0f0f")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ACCENT   = "#00ff88"
RED      = "#ff4444"
BLUE     = "#4488ff"
GRAY     = "#888888"
BG       = "#0f0f0f"
PANEL_BG = "#1a1a1a"

def style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.spines[:].set_color("#333333")
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.xaxis.label.set_color(GRAY)
    ax.yaxis.label.set_color(GRAY)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)

# ── 1. Loss Curve ──────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Training Loss")
ax1.plot(log["epoch"], log["loss"], color=ACCENT, linewidth=3, marker="o", markersize=8)
for x, y in zip(log["epoch"], log["loss"]):
    ax1.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                 xytext=(0, 12), ha="center", color=ACCENT, fontsize=9)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_xticks(log["epoch"])

# ── 2. Per-Digit Accuracy ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Accuracy per Digit")
digit_acc = []
for d in range(10):
    mask = preds["true_label"] == d
    acc  = (preds[mask]["predicted"] == d).mean() * 100
    digit_acc.append(acc)
colors = [ACCENT if a >= 95 else BLUE if a >= 90 else RED for a in digit_acc]
bars = ax2.bar(range(10), digit_acc, color=colors, edgecolor="#333", linewidth=0.5)
ax2.set_ylim(80, 100)
ax2.set_xlabel("Digit")
ax2.set_ylabel("Accuracy %")
ax2.set_xticks(range(10))
for bar, acc in zip(bars, digit_acc):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f"{acc:.1f}%", ha="center", va="bottom", color="white", fontsize=8)

# ── 3. Confusion Matrix ────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, "Confusion Matrix")
cm = np.zeros((10, 10), dtype=int)
for _, row in preds.iterrows():
    cm[int(row["true_label"])][int(row["predicted"])] += 1
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
im = ax3.imshow(cm_norm, cmap="Greens", vmin=0, vmax=100)
ax3.set_xticks(range(10))
ax3.set_yticks(range(10))
ax3.set_xlabel("Predicted")
ax3.set_ylabel("True Label")
for i in range(10):
    for j in range(10):
        color = "#0f0f0f" if cm_norm[i,j] > 10 else GRAY
        ax3.text(j, i, f"{cm_norm[i,j]:.0f}", ha="center", va="center",
                 color=color, fontsize=7)
plt.colorbar(im, ax=ax3, fraction=0.046).ax.tick_params(colors=GRAY)

# ── 4. Most Confused Pairs ─────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, "Top Misclassifications")
mistakes = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i,j] > 0:
            mistakes.append((cm[i,j], f"{i}→{j}"))
mistakes.sort(reverse=True)
top = mistakes[:8]
counts = [m[0] for m in top]
labels = [m[1] for m in top]
ax4.barh(labels[::-1], counts[::-1], color=RED, edgecolor="#333", linewidth=0.5)
ax4.set_xlabel("Number of mistakes")
for i, (c, l) in enumerate(zip(counts[::-1], labels[::-1])):
    ax4.text(c + 1, i, str(c), va="center", color="white", fontsize=9)

# ── 5. Prediction Distribution ────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, "Prediction Distribution")
pred_counts = preds["predicted"].value_counts().sort_index()
true_counts = preds["true_label"].value_counts().sort_index()
x = np.arange(10)
w = 0.35
ax5.bar(x - w/2, true_counts, w, label="True",      color=BLUE,  alpha=0.8, edgecolor="#333")
ax5.bar(x + w/2, pred_counts, w, label="Predicted", color=ACCENT, alpha=0.8, edgecolor="#333")
ax5.set_xlabel("Digit")
ax5.set_ylabel("Count")
ax5.set_xticks(x)
ax5.legend(facecolor=PANEL_BG, labelcolor="white", edgecolor="#333")

# ── 6. Summary Stats ──────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(PANEL_BG)
ax6.axis("off")
ax6.set_title("Summary", color="white", fontsize=11, fontweight="bold", pad=10)
stats = [
    ("Architecture",     "784→128→128→10"),
    ("Activation",       "tanh + softmax"),
    ("Optimizer",        "SGD"),
    ("Learning Rate",    "0.01 / (epoch + 1)"),
    ("Epochs",           "3"),
    ("Train Samples",    "60,000"),
    ("Final Loss",       f"{log['loss'].iloc[-1]:.4f}"),
    ("Train Accuracy",   f"{accuracy:.2f}%"),
    ("Best Digit",       f"{np.argmax(digit_acc)} ({max(digit_acc):.1f}%)"),
    ("Worst Digit",      f"{np.argmin(digit_acc)} ({min(digit_acc):.1f}%)"),
]
for idx, (k, v) in enumerate(stats):
    y_pos = 0.92 - idx * 0.09
    ax6.text(0.02, y_pos, k + ":", transform=ax6.transAxes,
             color=GRAY, fontsize=9, va="top")
    ax6.text(0.98, y_pos, v, transform=ax6.transAxes,
             color=ACCENT, fontsize=9, va="top", ha="right", fontweight="bold")

# ── Title ──────────────────────────────────────────────────────────────────────
fig.suptitle(f"Neural Network from Scratch — MNIST  |  Training Accuracy: {accuracy:.2f}%",
             color="white", fontsize=14, fontweight="bold", y=0.98)

plt.savefig("nn_results.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
print(f"Done! Accuracy: {accuracy:.2f}%")