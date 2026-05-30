import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scienceplots
plt.style.use(["science"])
import seaborn as sns
from matplotlib.colors import to_hex

fig, ax = plt.subplots(figsize=(8, 8))

import seaborn as sns

labels = [
    "Anomaly detection",
    "Continual learning",
    "Cloud resource optimization",
    "Green Cloud Computing"
]

colors = sns.color_palette("mako", 4)

label2color = {label: to_hex(color) for label, color in zip(labels, colors)}

label2color["Anomaly detection"] = "#D3DAD9"


circles = [
    Circle((0.35, 0.6), 0.32, alpha=0.35, facecolor=label2color["Anomaly detection"], edgecolor="black", linewidth=1.5),
    Circle((0.65, 0.6), 0.32, alpha=0.35, facecolor=label2color["Continual learning"], edgecolor="black", linewidth=1.5),
    Circle((0.35, 0.35), 0.32, alpha=0.35, facecolor=label2color["Cloud resource optimization"], edgecolor="black", linewidth=1.5),
    Circle((0.65, 0.35), 0.32, alpha=0.35, facecolor=label2color["Green Cloud Computing"], edgecolor="black", linewidth=1.5),
]

for c in circles:
    ax.add_patch(c)


ax.text(0.50, 0.48, "This\nstudy", fontsize=26, fontweight="bold", ha="center", va="center")

ax.text(0.25, 0.73, "Anomaly\ndetection", fontsize=18, fontweight="bold", ha="center", va="center")
ax.text(0.75, 0.73, "Continual\nlearning", fontsize=18, fontweight="bold", ha="center", va="center")
ax.text(0.25, 0.21, "Cloud\nresource\noptimization", fontsize=18, fontweight="bold", ha="center", va="center")
ax.text(0.75, 0.21, "Green\nCloud\nComputing", fontsize=18, fontweight="bold", ha="center", va="center")

ax.text(0.50, 0.72, "Continual\nanomaly\ndetection", fontsize=18, ha="center", va="center")
ax.text(0.50, 0.21, "Sustainable\ncloud\nresource\noptimization", fontsize=18, ha="center", va="center")
ax.text(0.75, 0.48, "Sustainable\ncontinual\nlearning", fontsize=18, ha="center", va="center")
ax.text(0.25, 0.48, "Temporal\nanomaly\ndetection", fontsize=18, ha="center", va="center")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig("venn.pdf", dpi=1500)
plt.show()
