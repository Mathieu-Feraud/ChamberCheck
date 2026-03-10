"""
Ad-hoc: plot discussion score distribution + topic group breakdowns
from the latest posts_analysis run in the latest scrape folder.
"""
import sys
import json
import collections
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# ── resolve latest scrape + latest analysis ───────────────────────────────────
raw_dir   = Path("data/raw")
scrape    = sorted([p for p in raw_dir.glob("scrape_*") if p.is_dir()])[-1]
ana_dir   = scrape / "posts_analysis"
ana_file  = sorted([f for f in ana_dir.glob("analysis_*.json") if "_metadata" not in f.name])[-1]
out_file  = ana_dir / (ana_file.stem + "_stats.png")

print(f"Scrape : {scrape.name}")
print(f"File   : {ana_file.name}")

# ── load ──────────────────────────────────────────────────────────────────────
data    = json.loads(ana_file.read_text(encoding="utf-8"))
valid   = [r for r in data if r.get("discussion_score") is not None and "error" not in r]
errors  = len(data) - len(valid)
print(f"Records: {len(data)}  valid={len(valid)}  errors={errors}")

scores   = [r["discussion_score"] for r in valid]
mean_all = np.mean(scores)

# per-subreddit
sub_scores: dict = collections.defaultdict(list)
for r in valid:
    if r.get("_community"):
        sub_scores[r["_community"]].append(r["discussion_score"])
subs  = sorted(sub_scores, key=lambda s: np.mean(sub_scores[s]))
means = [np.mean(sub_scores[s]) for s in subs]

# primary topic top-level counts
top_counts: dict = collections.Counter(
    (r.get("topic") or {}).get("top", "UNCLEAR") for r in valid
)

# primary topic mid-level counts (top-20)
mid_counts: dict = collections.Counter(
    (r.get("topic") or {}).get("mid") or "UNCLEAR" for r in valid
)
top_mids   = mid_counts.most_common(20)
mid_labels = [m for m, _ in top_mids]
mid_vals   = [c for _, c in top_mids]

# secondary topic usage
has_secondary = sum(1 for r in valid if r.get("secondary_topics"))
print(f"Posts with secondary topics: {has_secondary} ({100*has_secondary/len(valid):.1f}%)")

# ── style ─────────────────────────────────────────────────────────────────────
BG    = "#1c1c2e"
PANEL = "#12122a"
MUTED = "#aaaacc"
SPINE = "#444466"
cmap  = plt.cm.RdYlGn_r
norm  = plt.Normalize(0, 1)

fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.35)

# ── panel 1: overall score histogram ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(PANEL)
_, bins, patches = ax1.hist(scores, bins=20, range=(0, 1), edgecolor=BG, linewidth=0.5)
for patch, left in zip(patches, bins[:-1]):
    patch.set_facecolor(cmap(norm(left + 0.025)))
ax1.axvline(mean_all, color="white", linestyle="--", linewidth=1.3, alpha=0.75)
ax1.text(mean_all + 0.013, ax1.get_ylim()[1] * 0.88,
         f"mean={mean_all:.2f}", color="white", fontsize=9)
ax1.set_title(f"Discussion Score Distribution  (n={len(scores)})",
              color="white", fontsize=11, pad=8)
ax1.set_xlabel("discussion_score", color=MUTED, fontsize=9)
ax1.set_ylabel("Post count", color=MUTED, fontsize=9)
ax1.tick_params(colors=MUTED)
for sp in ax1.spines.values(): sp.set_edgecolor(SPINE)

# ── panel 2: mean score per subreddit ────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(PANEL)
bar_colors = [cmap(norm(m)) for m in means]
bars = ax2.barh(subs, means, color=bar_colors, edgecolor=BG, height=0.6)
for bar, m in zip(bars, means):
    ax2.text(m + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{m:.2f}", va="center", color="white", fontsize=8)
ax2.axvline(mean_all, color="white", linestyle="--", linewidth=1, alpha=0.5)
ax2.set_xlim(0, 1.12)
ax2.set_title("Mean Discussion Score per Subreddit", color="white", fontsize=11, pad=8)
ax2.set_xlabel("Mean discussion_score", color=MUTED, fontsize=9)
ax2.tick_params(colors=MUTED)
for sp in ax2.spines.values(): sp.set_edgecolor(SPINE)

# ── panel 3: top-level topic distribution ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(PANEL)
top_labels = [k for k, _ in top_counts.most_common()]
top_vals   = [top_counts[k] for k in top_labels]
bar_cols3  = ["#e05050" if t == "UNCLEAR" else "#5588ff" for t in top_labels]
b3 = ax3.barh(top_labels, top_vals, color=bar_cols3, edgecolor=BG, height=0.6)
for bar, v in zip(b3, top_vals):
    ax3.text(v + 8, bar.get_y() + bar.get_height() / 2,
             str(v), va="center", color="white", fontsize=8)
ax3.set_title("Primary Topic — Top Level", color="white", fontsize=11, pad=8)
ax3.set_xlabel("Post count", color=MUTED, fontsize=9)
ax3.tick_params(colors=MUTED)
for sp in ax3.spines.values(): sp.set_edgecolor(SPINE)

# ── panel 4: mid-level topic top-20 ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(PANEL)
b4 = ax4.barh(mid_labels, mid_vals, color="#7755cc", edgecolor=BG, height=0.6)
for bar, v in zip(b4, mid_vals):
    ax4.text(v + 3, bar.get_y() + bar.get_height() / 2,
             str(v), va="center", color="white", fontsize=8)
ax4.set_title("Primary Topic — Mid Level (top 20)", color="white", fontsize=11, pad=8)
ax4.set_xlabel("Post count", color=MUTED, fontsize=9)
ax4.tick_params(colors=MUTED)
for sp in ax4.spines.values(): sp.set_edgecolor(SPINE)

plt.suptitle(
    f"Post Analysis Stats  |  {ana_file.name}  |  n={len(valid)}  errors={errors}  "
    f"secondary={has_secondary} ({100*has_secondary/len(valid):.0f}%)",
    color="white", fontsize=12, y=1.01,
)

fig.savefig(out_file, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out_file}")
plt.show()
