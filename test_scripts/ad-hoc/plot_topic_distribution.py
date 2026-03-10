"""
Plot topic distribution (top / mid) from a posts_analysis JSON file,
with per-subreddit breakdown to reveal cross-community overlap.
"""
import sys
import json
import collections
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ANALYSIS_FILE = Path("data/raw/scrape_004/posts_analysis/analysis_002.json")
OUT_FILE      = ANALYSIS_FILE.parent / "topic_distribution.png"
TOP_N_MID     = 25   # how many mid-level topics to show in the bar chart

# ── load ─────────────────────────────────────────────────────────────────────

data = json.loads(ANALYSIS_FILE.read_text(encoding="utf-8"))

tops, mids = [], []
sub_top: dict[str, list] = collections.defaultdict(list)

for r in data:
    t = r.get("topic", {})
    top = t.get("top") or "UNCLEAR"
    mid = t.get("mid") or "—"
    tops.append(top)
    mids.append(f"{top} → {mid}")
    if r.get("_community"):
        sub_top[r["_community"]].append(top)

top_counts = collections.Counter(tops)
mid_counts = collections.Counter(mids)

print(f"\n{'TOP-LEVEL':=<50}")
for k, v in top_counts.most_common():
    print(f"  {v:4d}  {k}")

print(f"\n{'TOP MID-LEVEL (top {TOP_N_MID})':=<50}")
for k, v in mid_counts.most_common(TOP_N_MID):
    print(f"  {v:4d}  {k}")

# subreddit × top-level matrix
all_tops_sorted = [k for k, _ in top_counts.most_common()]
subs = sorted(sub_top.keys())

matrix = np.zeros((len(subs), len(all_tops_sorted)), dtype=int)
for i, sub in enumerate(subs):
    c = collections.Counter(sub_top[sub])
    for j, top in enumerate(all_tops_sorted):
        matrix[i, j] = c.get(top, 0)

# ── style ─────────────────────────────────────────────────────────────────────
BG    = "#1c1c2e"
PANEL = "#12122a"
MUTED = "#aaaacc"
SPINE = "#444466"
plt.rcParams.update({"text.color": "white", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED})

fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55,
                       height_ratios=[1.2, 1.4, 1.4])

# ── panel 1: top-level bar chart ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(PANEL)

labels1 = [k for k, _ in top_counts.most_common()]
vals1   = [top_counts[k] for k in labels1]
colors1 = plt.cm.tab20(np.linspace(0, 1, len(labels1)))

bars1 = ax1.bar(labels1, vals1, color=colors1, edgecolor=BG, linewidth=0.4)
for bar, v in zip(bars1, vals1):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             str(v), ha="center", va="bottom", fontsize=8, color="white")

ax1.set_title("Top-level Topic Distribution  (n=300)", fontsize=13, pad=8, color="white")
ax1.set_ylabel("Post count", color=MUTED)
ax1.tick_params(axis="x", rotation=30, labelsize=8)
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
for sp in ax1.spines.values(): sp.set_edgecolor(SPINE)

# ── panel 2: mid-level horizontal bar (top N) ─────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(PANEL)

mid_keys  = [k for k, _ in mid_counts.most_common(TOP_N_MID)][::-1]
mid_vals  = [mid_counts[k] for k in mid_keys]
colors2   = plt.cm.tab20(np.linspace(0, 1, len(mid_keys)))

bars2 = ax2.barh(mid_keys, mid_vals, color=colors2, edgecolor=BG, height=0.7)
for bar, v in zip(bars2, mid_vals):
    ax2.text(v + 0.2, bar.get_y() + bar.get_height() / 2,
             str(v), va="center", fontsize=8, color="white")

ax2.set_xlim(0, max(mid_vals) * 1.15)
ax2.set_title(f"Top {TOP_N_MID} Mid-level Topics  (top → mid)", fontsize=12, pad=8, color="white")
ax2.set_xlabel("Post count", color=MUTED)
ax2.tick_params(axis="y", labelsize=7.5)
for sp in ax2.spines.values(): sp.set_edgecolor(SPINE)

# ── panel 3: stacked bar — subreddit × top-level ─────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor(PANEL)

cmap3   = plt.cm.tab20(np.linspace(0, 1, len(all_tops_sorted)))
bottoms = np.zeros(len(subs))

for j, top in enumerate(all_tops_sorted):
    col_vals = matrix[:, j]
    ax3.bar(subs, col_vals, bottom=bottoms,
            label=top, color=cmap3[j], edgecolor=BG, linewidth=0.3)
    bottoms += col_vals

ax3.set_title("Topic Overlap per Subreddit  (stacked by top-level)", fontsize=12, pad=8, color="white")
ax3.set_ylabel("Post count", color=MUTED)
ax3.tick_params(axis="x", rotation=35, labelsize=8)
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
ax3.legend(loc="upper right", fontsize=7, framealpha=0.3,
           facecolor=PANEL, edgecolor=SPINE, labelcolor="white",
           ncol=2)
for sp in ax3.spines.values(): sp.set_edgecolor(SPINE)

plt.suptitle("Post Topic Distribution  |  analysis_002.json",
             color="white", fontsize=14, y=0.995)

fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {OUT_FILE}")
