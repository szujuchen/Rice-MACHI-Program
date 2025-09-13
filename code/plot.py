import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["LLaVA", "Route\n(Chain-of-Spot)", "Chain-of-Thought\n(Chain-of-Spot)"]
time = [0.23857, 0.8310188752, 1.62990]
accuracy = [34.54, 35.94, 36.64]

x = np.arange(len(models))  # X locations for the groups
width = 0.4  # Bar width

fig, ax1 = plt.subplots(figsize=(8, 6))

# Bar chart for time (left y-axis)
bars1 = ax1.bar(x - width/2, time, width, label="Time (s)", color="lightblue", alpha=0.5)
ax1.set_ylabel("Time (s)", color="black")
ax1.tick_params(axis='y', labelcolor="black")

# Creating the second y-axis
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, accuracy, width, label="Accuracy (%)", color="navy", alpha=0.5)
ax2.set_ylabel("Accuracy (%)", color="black")
ax2.tick_params(axis='y', labelcolor="black")
for bar, acc in zip(bars2, accuracy):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()-1.5, f"{acc:.2f}%", 
             ha='center', va='bottom', fontsize=10, color="white", fontweight="bold")


# X-axis labels
plt.xticks(x, models)
fig.legend([bars1, bars2], ["Time", "Accuracy"], loc="upper left", frameon=True, fontsize=9, facecolor="white", bbox_to_anchor=(0.08, 0.94))

# Titles and legend
plt.title("Comparison of Models")
fig.tight_layout()

plt.savefig("model_comparison_cos.png", dpi=300, bbox_inches="tight")
