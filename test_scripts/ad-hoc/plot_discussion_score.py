"""
Plot discussion_score distribution from a posts_analysis JSON file.
"""
import sys
import json
import collections
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ANALYSIS_FILE = Path("data/raw/scrape_004/posts_analysis/analysis_002.json")
OUT_FILE      = ANALYSIS_FILE.parent / "discussion_score_distribution.png"

# ── load ─────────────────────────────────────────────────────────────────────

data = json.loads(ANALYSIS_FILE.read_text(encoding="utf-8"))
scores = [r["discussion_score"] for r in data if r.get("discussion_score") is not None]

sub_scores: dict[str, list] = collections.defaultdict(list)
for r in data:
    if r.get("discussion_score") is not None and r.get("_community"):
        sub_scores[r["_community"]].append(r["discussion_score"])

subs  = sorted(sub_scores, key=lambda s: -np.mean(sub_scores[s]))
means = [np.mean(sub_scores[s]) for s in subs]
mean_all = np.mean(scores)

print(f"n={len(scores)}, mean={mean_all:.3f}, min={min(scores):.2f}, max={max(scores):.2f}")

# ── plot ─────────────────────────────────────────────────────────────────────

BG      = "#1c1c2e"
PANEL   = "#12122a"
MUTED   = "#aaaacc"
SPINE   = "#444466"
cmap    = plt.cm.RdYlGn_r
norm    = plt.Normalize(0, 1)

fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.5)

# ── panel 1: overall histogram ───────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(PANEL)

_, bins, patches = ax1.hist(scores, bins=20, range=(0, 1),
                             edgecolor=BG, linewidth=0.5)
for patch, left in zip(patches, bins[:-1]):
    patch.set_facecolor(cmap(norm(left + 0.025)))

ax1.axvline(mean_all, color="white", linestyle="--", linewidth=1.3, alpha=0.75)
ax1.text(mean_all + 0.013, ax1.get_ylim()[1] * 0.88,
         f"mean = {mean_all:.2f}", color="white", fontsize=9.5)

ax1.set_title(f"Discussion Score Distribution  (n={len(scores)})",
              color="white", fontsize=13, pad=10)
ax1.set_xlabel("discussion_score  (0 = no debate likely → 1 = highly contentious)", color=MUTED)
ax1.set_ylabel("Post count", color=MUTED)
ax1.tick_params(colors=MUTED)
for sp in ax1.spines.values():
    sp.set_edgecolor(SPINE)

# ── panel 2: mean per subreddit ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(PANEL)

bar_colors = [cmap(norm(m)) for m in means]
bars = ax2.barh(subs, means, color=bar_colors, edgecolor=BG, height=0.6)
for bar, m in zip(bars, means):
    ax2.text(m + 0.012, bar.get_y() + bar.get_height() / 2,
             f"{m:.2f}", va="center", color="white", fontsize=8.5)

ax2.axvline(mean_all, color="white", linestyle="--", linewidth=1, alpha=0.5)
ax2.set_xlim(0, 1.08)
ax2.set_title("Mean Discussion Score per Subreddit", color="white", fontsize=12, pad=8)
ax2.set_xlabel("Mean discussion_score", color=MUTED)
ax2.tick_params(colors=MUTED)
for sp in ax2.spines.values():
    sp.set_edgecolor(SPINE)

plt.suptitle("Post Title — Discussion Score  |  analysis_002.json",
             color="white", fontsize=14, y=0.99)

fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {OUT_FILE}")
plt.show()
