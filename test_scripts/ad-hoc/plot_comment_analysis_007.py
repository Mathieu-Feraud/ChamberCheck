"""
Ad-hoc: visualise scrape_007 comment analysis results.

Loads:
  data/output/scrape_007/comment_analysis_001.json
  data/raw/scrape_007/pre_process/pre_process_005.json  (for subreddit mapping)

Saves one PNG per figure to:
  data/output/scrape_007/plots/
"""

import sys
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({"font.size": 11, "figure.autolayout": True})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

ANALYSIS_FILE = REPO / "data/output/scrape_007/comment_analysis_001.json"
PREPROCESS_FILE = REPO / "data/raw/scrape_007/pre_process/pre_process_005.json"
OUT_DIR = REPO / "data/output/scrape_007/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBS_ORDER = ["decodingthegurus", "samharris", "philosophy", "lexfridman", "HubermanLab"]
SUB_LABELS = {
    "decodingthegurus": "DecodingTheGurus",
    "samharris": "SamHarris",
    "philosophy": "Philosophy",
    "lexfridman": "LexFridman",
    "HubermanLab": "HubermanLab",
}
PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
SUB_COLORS = dict(zip(SUBS_ORDER, PALETTE))

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading data …")
analysis = json.loads(ANALYSIS_FILE.read_text(encoding="utf-8"))
preprocess = json.loads(PREPROCESS_FILE.read_text(encoding="utf-8"))

# Build post_id → community map
post_community = {p["post_id"]: p["community"] for p in preprocess}

# Filter to successful records with a known community
records = [
    e for e in analysis
    if not e.get("error") and e.get("post_id") in post_community
]
print(f"Records loaded: {len(records):,}")

# Attach community to each record
for r in records:
    r["community"] = post_community[r["post_id"]]

# Coerce numeric fields (some were returned as strings)
NUM_FIELDS = ["toxicity", "discrediting", "defensiveness", "civility"]
for r in records:
    for f in NUM_FIELDS:
        try:
            r[f] = int(r[f])
        except (ValueError, TypeError):
            r[f] = None

# Group records by subreddit
by_sub = defaultdict(list)
for r in records:
    by_sub[r["community"]].append(r)

print("By subreddit:", {s: len(by_sub[s]) for s in SUBS_ORDER})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


def sub_label(s):
    return SUB_LABELS.get(s, s)


def violinplot_grid(data_by_sub, field, title, ylabel, filename, vmin=0, vmax=5):
    """Violin / strip plot of a numeric field, one column per subreddit."""
    subs = [s for s in SUBS_ORDER if s in data_by_sub]
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = list(range(len(subs)))
    parts = ax.violinplot(
        [data_by_sub[s] for s in subs],
        positions=positions,
        showmedians=True,
        showextrema=True,
    )
    for pc, color in zip(parts["bodies"], PALETTE):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.2)
    ax.set_xticks(positions)
    ax.set_xticklabels([sub_label(s) for s in subs], rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(vmin - 0.3, vmax + 0.3)
    ax.set_yticks(range(vmin, vmax + 1))
    ax.yaxis.grid(True, alpha=0.4)
    save(fig, filename)


def mean_bar(data_by_sub, field, title, ylabel, filename, vmin=0, vmax=5):
    """Bar chart of means with error bars (±1 std)."""
    subs = [s for s in SUBS_ORDER if s in data_by_sub]
    means = [np.mean(data_by_sub[s]) for s in subs]
    stds = [np.std(data_by_sub[s]) for s in subs]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(subs)), means, color=[SUB_COLORS[s] for s in subs],
                  yerr=stds, capsize=5, alpha=0.85)
    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels([sub_label(s) for s in subs], rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(vmin, vmax + 0.2)
    ax.yaxis.grid(True, alpha=0.4)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
    save(fig, filename)


# ---------------------------------------------------------------------------
# 1. Violin plots: toxicity, discrediting, defensiveness, civility
# ---------------------------------------------------------------------------
print("\n[1] Toxicity / civility violin plots")
for field, label, fname in [
    ("toxicity",     "Toxicity (0–5)",      "violin_toxicity"),
    ("discrediting", "Discrediting (0–5)",  "violin_discrediting"),
    ("defensiveness","Defensiveness (0–5)", "violin_defensiveness"),
    ("civility",     "Civility (0–5)",      "violin_civility"),
]:
    data_by_sub = {
        s: [r[field] for r in by_sub[s] if r.get(field) is not None]
        for s in SUBS_ORDER
    }
    violinplot_grid(data_by_sub, field,
                    title=f"{label} by subreddit",
                    ylabel=label,
                    filename=fname)

# ---------------------------------------------------------------------------
# 2. Mean bar charts (same four fields, side-by-side comparison)
# ---------------------------------------------------------------------------
print("\n[2] Mean bar charts — four metrics")
for field, label, fname in [
    ("toxicity",     "Mean Toxicity (0–5)",      "bar_toxicity"),
    ("discrediting", "Mean Discrediting (0–5)",  "bar_discrediting"),
    ("defensiveness","Mean Defensiveness (0–5)", "bar_defensiveness"),
    ("civility",     "Mean Civility (0–5)",      "bar_civility"),
]:
    data_by_sub = {
        s: [r[field] for r in by_sub[s] if r.get(field) is not None]
        for s in SUBS_ORDER
    }
    mean_bar(data_by_sub, field, title=f"{label} by subreddit",
             ylabel=label, filename=fname)

# ---------------------------------------------------------------------------
# 3. Multi-metric radar / grouped bar (summary panel)
# ---------------------------------------------------------------------------
print("\n[3] Summary: grouped bar chart — all 4 metrics")
metrics = ["toxicity", "discrediting", "defensiveness", "civility"]
metric_labels = ["Toxicity", "Discrediting", "Defensiveness", "Civility"]
subs = [s for s in SUBS_ORDER if s in by_sub]
x = np.arange(len(metrics))
width = 0.15
fig, ax = plt.subplots(figsize=(12, 6))
for i, sub in enumerate(subs):
    means = []
    for m in metrics:
        vals = [r[m] for r in by_sub[sub] if r.get(m) is not None]
        means.append(np.mean(vals) if vals else 0)
    offset = (i - len(subs) / 2 + 0.5) * width
    bars = ax.bar(x + offset, means, width, label=sub_label(sub),
                  color=PALETTE[i], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 5.2)
ax.set_ylabel("Mean score (0–5)")
ax.set_title("Comment metrics by subreddit — mean scores")
ax.yaxis.grid(True, alpha=0.4)
ax.legend(loc="upper right", fontsize=9)
save(fig, "summary_grouped_bar")

# ---------------------------------------------------------------------------
# 4. Stance distribution by subreddit
# ---------------------------------------------------------------------------
print("\n[4] Stance distribution")
def _stance(r):
    try:
        return int(r["topic"]["stance"]["value"])
    except (TypeError, ValueError, KeyError):
        return None

all_stances = [_stance(r) for r in records]
stance_vals = sorted(set(v for v in all_stances if v is not None))
stance_labels = {v: str(v) for v in stance_vals}

fig, axes = plt.subplots(1, len(subs), figsize=(16, 4), sharey=True)
for ax, sub in zip(axes, subs):
    vals = [sv for r in by_sub[sub] if (sv := _stance(r)) is not None]
    cnt = Counter(vals)
    total = sum(cnt.values()) or 1
    sv = sorted(cnt.keys())
    ax.bar([str(v) for v in sv], [cnt[v] / total * 100 for v in sv],
           color=SUB_COLORS[sub], alpha=0.85)
    ax.set_title(sub_label(sub), fontsize=9)
    ax.set_xlabel("Stance")
    ax.tick_params(axis="x", labelsize=8)
axes[0].set_ylabel("% of comments")
fig.suptitle("Stance value distribution by subreddit", fontsize=12)
save(fig, "stance_distribution")

# ---------------------------------------------------------------------------
# 5. Emotion breakdown (anger, anxiety, disgust) — stacked bar of means
# ---------------------------------------------------------------------------
print("\n[5] Emotion breakdown")
emotions = ["anger", "anxiety", "disgust"]

def _emo_val(r, emo):
    try:
        return int(r["emotion"][emo])
    except (TypeError, ValueError, KeyError):
        return None
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(subs))
bottoms = np.zeros(len(subs))
emotion_colors = ["#e15759", "#f28e2b", "#76b7b2"]
for emo, col in zip(emotions, emotion_colors):
    means = []
    for sub in subs:
        vals = [v for r in by_sub[sub] if (v := _emo_val(r, emo)) is not None]
        means.append(np.mean(vals) if vals else 0)
    ax.bar(x, means, bottom=bottoms, label=emo.capitalize(), color=col, alpha=0.85)
    bottoms += np.array(means)
ax.set_xticks(x)
ax.set_xticklabels([sub_label(s) for s in subs], rotation=20, ha="right")
ax.set_title("Mean emotion scores by subreddit (stacked)")
ax.set_ylabel("Mean score (stacked)")
ax.legend()
ax.yaxis.grid(True, alpha=0.4)
save(fig, "emotion_stacked")

# Grouped version
fig, ax = plt.subplots(figsize=(10, 5))
width = 0.25
for i, (emo, col) in enumerate(zip(emotions, emotion_colors)):
    means = []
    for sub in subs:
        vals = [v for r in by_sub[sub] if (v := _emo_val(r, emo)) is not None]
        means.append(np.mean(vals) if vals else 0)
    offset = (i - 1) * width
    bars = ax.bar(x + offset, means, width, label=emo.capitalize(), color=col, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([sub_label(s) for s in subs], rotation=20, ha="right")
ax.set_title("Mean emotion scores by subreddit (grouped)")
ax.set_ylabel("Mean score (0–5)")
ax.legend()
ax.yaxis.grid(True, alpha=0.4)
save(fig, "emotion_grouped")

# ---------------------------------------------------------------------------
# 6. Comment type frequency by subreddit
# ---------------------------------------------------------------------------
print("\n[6] Comment type frequency")
all_types_counter = Counter(
    t for r in records for t in (r.get("comment_type") or [])
)
# Keep top-10 types
top_types = [t for t, _ in all_types_counter.most_common(10)]

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(top_types))
width = 0.15
for i, sub in enumerate(subs):
    total = max(len(by_sub[sub]), 1)
    freqs = []
    for t in top_types:
        cnt = sum(1 for r in by_sub[sub] if t in (r.get("comment_type") or []))
        freqs.append(cnt / total * 100)
    offset = (i - len(subs) / 2 + 0.5) * width
    ax.bar(x + offset, freqs, width, label=sub_label(sub), color=PALETTE[i], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(top_types, rotation=35, ha="right")
ax.set_ylabel("% of comments (multi-label)")
ax.set_title("Comment type frequency by subreddit (top 10 types)")
ax.yaxis.grid(True, alpha=0.4)
ax.legend(loc="upper right", fontsize=9)
save(fig, "comment_types")

# ---------------------------------------------------------------------------
# 7. Epistemic risk ordinal heatmap
# ---------------------------------------------------------------------------
print("\n[7] Epistemic risk heatmap")
EPISTEMIC_ORDER = {
    "claim_strength":    ["N/A", "weak", "moderate", "strong"],
    "evidence_quality":  ["N/A", "anecdotal", "weak", "moderate", "strong"],
    "reasoning_depth":   ["N/A", "shallow", "moderate", "deep"],
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (field, cats) in zip(axes, EPISTEMIC_ORDER.items()):
    matrix = []
    for sub in subs:
        row = []
        total = max(len(by_sub[sub]), 1)
        cnt = Counter(
            r["epistemic_risk"].get(field)
            for r in by_sub[sub]
            if isinstance(r.get("epistemic_risk"), dict)
        )
        for cat in cats:
            row.append(cnt.get(cat, 0) / total * 100)
        matrix.append(row)
    arr = np.array(matrix)
    im = ax.imshow(arr, aspect="auto", cmap="YlOrRd", vmin=0, vmax=arr.max())
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(subs)))
    ax.set_yticklabels([sub_label(s) for s in subs], fontsize=9)
    ax.set_title(field.replace("_", " ").title())
    for r in range(len(subs)):
        for c in range(len(cats)):
            ax.text(c, r, f"{arr[r, c]:.0f}%", ha="center", va="center",
                    fontsize=8, color="black" if arr[r, c] < 60 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("Epistemic risk — % of comments per subreddit", fontsize=12)
save(fig, "epistemic_risk_heatmap")

# ---------------------------------------------------------------------------
# 8. Overall summary heatmap (mean numeric scores per subreddit)
# ---------------------------------------------------------------------------
print("\n[8] Summary heatmap")
metrics_h = ["toxicity", "discrediting", "defensiveness", "civility"]
metric_h_labels = ["Toxicity", "Discrediting", "Defensiveness", "Civility"]
matrix = []
for sub in subs:
    row = []
    for m in metrics_h:
        vals = [r[m] for r in by_sub[sub] if r.get(m) is not None]
        row.append(np.mean(vals) if vals else 0)
    matrix.append(row)
arr = np.array(matrix)
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(arr, aspect="auto", cmap="coolwarm", vmin=0, vmax=5)
ax.set_xticks(range(len(metric_h_labels)))
ax.set_xticklabels(metric_h_labels)
ax.set_yticks(range(len(subs)))
ax.set_yticklabels([sub_label(s) for s in subs])
ax.set_title("Mean comment scores by subreddit")
for r in range(len(subs)):
    for c in range(len(metrics_h)):
        ax.text(c, r, f"{arr[r, c]:.2f}", ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white" if arr[r, c] > 3 or arr[r, c] < 1 else "black")
plt.colorbar(im, ax=ax, label="Mean score (0–5)")
save(fig, "summary_heatmap")

print(f"\nAll plots saved to {OUT_DIR}")
