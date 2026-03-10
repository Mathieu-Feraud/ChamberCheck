"""
Quick V3 metrics visualisation.

Merges comment_analysis_001.json + clean lines from comment_analysis_002.jsonl,
computes all nine V3 metrics (aggregate + per-subreddit), and writes a
multi-panel PNG to data/output/scrape_006/v3_metrics_plot.png.

Usage (from project root):
    .\\venv\\Scripts\\python.exe test_scripts\\ad-hoc\\plot_v3_metrics.py
"""

import json
import sys
from dataclasses import is_dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ChamberCheck.CC_derived_metrics.derived_metrics import V3Metrics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR    = Path("data/output/scrape_006")
ANALYSIS_001  = OUTPUT_DIR / "comment_analysis_001.json"
ANALYSIS_002  = OUTPUT_DIR / "comment_analysis_002.jsonl"
FILTERED_FILE = Path("data/raw/scrape_006/comments/comments_filtered_001.json")
OUT_PNG       = OUTPUT_DIR / "v3_metrics_plot.png"

# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
records = json.loads(ANALYSIS_001.read_text(encoding="utf-8"))
existing_ids = {r["comment_id"] for r in records if r.get("comment_id")}

if ANALYSIS_002.exists():
    for line in ANALYSIS_002.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error") or not r.get("comment_id"):
            continue
        if r["comment_id"] not in existing_ids:
            records.append(r)
            existing_ids.add(r["comment_id"])

print(f"Merged records: {len(records)}")

# write temp file so from_files can load it
import tempfile, os
tmp = Path(tempfile.mktemp(suffix=".json"))
tmp.write_text(json.dumps(records), encoding="utf-8")

m = V3Metrics.from_files(str(tmp), str(FILTERED_FILE))
tmp.unlink()

subreddits = m.get_subreddits()
print(f"Subreddits: {subreddits}")

agg            = m.compute_all()
by_sub         = {sr: m.compute_all(sr) for sr in subreddits}

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG     = "#1c1c2e"
PANEL  = "#12122a"
ACCENT = "#7c7cff"
MUTED  = "#aaaacc"
SPINE  = "#444466"
GOOD   = "#50fa7b"
BAD    = "#ff5555"
WARN   = "#f1fa8c"

SUB_COLOURS = {
    "Christianity": "#bd93f9",
    "antiwork":     "#ff79c6",
    "atheism":      "#8be9fd",
    "conservative": "#ffb86c",
}
DEFAULT_COLOURS = ["#bd93f9", "#ff79c6", "#8be9fd", "#ffb86c",
                   "#50fa7b", "#f1fa8c", "#ff5555", "#6272a4"]

def sub_colour(i, sr):
    return SUB_COLOURS.get(sr, DEFAULT_COLOURS[i % len(DEFAULT_COLOURS)])

def val(d):
    """Extract .value from a MetricResult dataclass or dict."""
    if d is None:
        return None
    if is_dataclass(d) and not isinstance(d, type):
        return d.value
    if isinstance(d, dict) and "value" in d:
        return d["value"]
    return None

def _sample_size(d):
    """Extract sample_size from a MetricResult or dict."""
    if d is None:
        return 0
    if is_dataclass(d) and not isinstance(d, type):
        return getattr(d, "sample_size", 0)
    if isinstance(d, dict):
        return d.get("sample_size", 0)
    return 0

def v0(d):
    """Like val() but returns 0.0 instead of None."""
    v = val(d)
    return 0.0 if v is None else v

def _bar_colour(v, neutral=0.0, flip=False):
    """Green when direction looks healthy, red when bad."""
    if v is None:
        return MUTED
    delta = v - neutral
    if flip:
        delta = -delta
    return GOOD if delta <= 0 else BAD

# ---------------------------------------------------------------------------
# Helper: axis styling
# ---------------------------------------------------------------------------
def _style(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(title, color="white", fontsize=9, pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.title.set_color("white")

def _hline(ax, y=0.0):
    ax.axhline(y, color=SPINE, linewidth=0.8, linestyle="--")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
PLOT_DIR = OUTPUT_DIR / "v3_plots"
PLOT_DIR.mkdir(exist_ok=True)
saved = []

def _new_fig(w=11, h=6, title=""):
    f = plt.figure(figsize=(w, h))
    f.patch.set_facecolor(BG)
    if title:
        f.suptitle(title, color="white", fontsize=11, y=1.01, va="bottom")
    return f

def _save(f, name):
    path = PLOT_DIR / name
    f.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(f)
    saved.append(path)
    print(f"  Saved -> {path}")

SUB_LABEL = "7,674 comments  |  4 subreddits  |  scrape_006 partial"

# ---------------------------------------------------------------------------
# directions used by CSAD / TD
# ---------------------------------------------------------------------------
directions  = ["majority_to_minority", "minority_to_majority", "same_baseline"]
dir_labels  = ["Maj->Min", "Min->Maj", "Same-stance baseline"]
dir_colours = [BAD, ACCENT, WARN]

def _direction_bars(ax, metric_key, title, ylabel=""):
    x = np.arange(len(subreddits))
    width = 0.22
    for i, (dir_k, lbl, col) in enumerate(zip(directions, dir_labels, dir_colours)):
        vals_d = [v0(by_sub[sr][metric_key].get(dir_k)) for sr in subreddits]
        ax.bar(x + i*width, vals_d, width, label=lbl,
               color=col, edgecolor=BG, linewidth=0.5, alpha=0.82)
    for i, (dir_k, col) in enumerate(zip(directions, dir_colours)):
        agg_raw = agg[metric_key]
        agg_dir = agg_raw.get(dir_k) if isinstance(agg_raw, dict) else None
        agg_v = val(agg_dir)
        if agg_v is not None:
            ax.plot(x + i*width, [agg_v]*len(subreddits),
                    "w--", linewidth=0.7, alpha=0.5)
    _hline(ax)
    ax.set_xticks(x + width)
    ax.set_xticklabels(subreddits, fontsize=10)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=MUTED, edgecolor=SPINE)
    _style(ax, title, ylabel=ylabel)

# ============================================================
# Plot 1 — CSS
# ============================================================
fig = _new_fig(8, 5, "1a. CSS — Counter-Stance Silence Rate")
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL)
css_vals = [v0(by_sub[sr]["CSS"]) for sr in subreddits]
css_raw  = [val(by_sub[sr]["CSS"]) for sr in subreddits]
colours  = [_bar_colour(v, neutral=0) for v in css_vals]
bars = ax.bar(subreddits, css_vals, color=colours, edgecolor=BG, linewidth=0.7)
_hline(ax)
for b, raw in zip(bars, css_raw):
    lbl = f"{raw:.3f}" if raw is not None else "n/a"
    yoff = b.get_height() + 0.005 if b.get_height() >= 0 else b.get_height() - 0.02
    ax.text(b.get_x() + b.get_width()/2, yoff, lbl,
            ha="center", va="bottom", color="white", fontsize=10)
ax.tick_params(axis="x", labelsize=11)
_style(ax, f"Positive = minority comments silenced more  |  {SUB_LABEL}",
       ylabel="silence_counter - silence_majority")
_save(fig, "01_css.png")

# ============================================================
# Plot 2 — CSEQ
# ============================================================
fig = _new_fig(13, 6, "1b. CSEQ — Cross-Stance Engagement Quality by Direction")
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL)

cseq_dir_keys = ["majority_to_minority", "minority_to_majority"]
cseq_dir_lbls = ["Majority -> Minority", "Minority -> Majority"]
cseq_dir_cols = [BAD, ACCENT]
cseq_met_keys = ["mean_discrediting", "mean_evidence_quality", "mean_reasoning_depth"]
cseq_met_lbls = ["Discrediting", "Evid. Quality", "Reasoning Depth"]

n_sr  = len(subreddits)
n_dir = len(cseq_dir_keys)
n_met = len(cseq_met_keys)
group_w = 0.8
slot_w  = group_w / (n_dir * n_met)
x = np.arange(n_sr)

# group by metric then direction: [Disc Maj→Min, Disc Min→Maj | EvQ Maj→Min, EvQ Min→Maj | Depth Maj→Min, Depth Min→Maj]
# colour = metric, alpha = direction (dark = Maj→Min, light = Min→Maj)
met_hex = ["#e05050", "#5b9cf6", "#9b59b6"]   # Discrediting / EvQ / Reasoning
dir_alphas = [0.92, 0.50]                       # Maj→Min / Min→Maj
for mi, (mk, ml, mc) in enumerate(zip(cseq_met_keys, cseq_met_lbls, met_hex)):
    for di, (dk, dl) in enumerate(zip(cseq_dir_keys, cseq_dir_lbls)):
        slot_idx = mi * n_dir + di
        offset = (slot_idx - (n_dir * n_met - 1) / 2) * slot_w
        vals = [by_sub[sr]["CSEQ"].get(dk, {}).get(mk) or 0.0 for sr in subreddits]
        label = f"{ml} | {dl}" if di == 0 else f"_ {ml} | {dl}"
        ax.bar(x + offset, vals, slot_w * 0.88,
               color=mc, edgecolor=BG, linewidth=0.4,
               alpha=dir_alphas[di], label=label)

ax.set_xticks(x)
ax.set_xticklabels(subreddits, fontsize=11)
ax.legend(fontsize=8, facecolor=PANEL, labelcolor=MUTED, edgecolor=SPINE,
          loc="upper right", ncol=3)
ax.tick_params(axis="y", labelsize=9)
_style(ax,
       "Grouped by metric  |  Dark = Majority→Minority  Light = Minority→Majority\n"
       f"Red=Discrediting  Blue=Evid.Quality  Purple=Reasoning  |  {SUB_LABEL}",
       ylabel="Mean score")
_save(fig, "02_cseq.png")

# ============================================================
# Plot 3 — RDB + uRDB side by side
# ============================================================
fig = _new_fig(14, 6, "3. Reply Direction Bias")
gs3 = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

ax_rdb = fig.add_subplot(gs3[0])
ax_rdb.set_facecolor(PANEL)
x = np.arange(len(subreddits))
width = 0.26
rdb_series = [
    ("Majority (pos)",  [v0(by_sub[sr]["RDB"]["pro"])       for sr in subreddits], GOOD),
    ("Minority (neg)",  [v0(by_sub[sr]["RDB"]["con"])       for sr in subreddits], BAD),
    ("Aggregate bias",  [v0(by_sub[sr]["RDB"]["aggregate"]) for sr in subreddits], ACCENT),
]
for i, (lbl, vals_r, col) in enumerate(rdb_series):
    ax_rdb.bar(x + i*width, vals_r, width, label=lbl,
               color=col, edgecolor=BG, linewidth=0.5, alpha=0.85)
ax_rdb.axhline(0.5, color=WARN, linewidth=1.2, linestyle=":", alpha=0.85, label="0.5 random")
ax_rdb.set_xticks(x + width)
ax_rdb.set_xticklabels(subreddits, fontsize=11)
ax_rdb.legend(fontsize=8, facecolor=PANEL, labelcolor=MUTED, edgecolor=SPINE)
_style(ax_rdb, "3a. RDB — fraction of replies directed at same-stance comments\n(dotted line = random baseline 0.5)",
       ylabel="Same-stance reply fraction")

ax_urdb = fig.add_subplot(gs3[1])
ax_urdb.set_facecolor(PANEL)
urdb_vals = [v0(by_sub[sr]["uRDB"]) for sr in subreddits]
colours   = [sub_colour(i, sr) for i, sr in enumerate(subreddits)]
ax_urdb.bar(subreddits, urdb_vals, color=colours, edgecolor=BG, linewidth=0.7, width=0.5)
ax_urdb.axhline(0.5, color=WARN, linewidth=1.2, linestyle=":", alpha=0.85, label="0.5 random")
urdb_agg = val(agg["uRDB"])
if urdb_agg is not None:
    ax_urdb.axhline(urdb_agg, color=ACCENT, linewidth=1.5, linestyle="--", alpha=0.9,
                    label=f"aggregate = {urdb_agg:.3f}")
ax_urdb.legend(fontsize=8, facecolor=PANEL, labelcolor=MUTED, edgecolor=SPINE)
for b, v in zip(ax_urdb.patches[:len(subreddits)], urdb_vals):
    ax_urdb.text(b.get_x() + b.get_width()/2, v + 0.01,
                 f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=10)
ax_urdb.tick_params(axis="x", labelsize=11)
_style(ax_urdb, "3b. uRDB — per-user bias averaged within thread\n(>0.5 = echo-chamber behaviour)",
       ylabel="Mean user same-stance fraction")
_save(fig, "03_rdb_urdb.png")

# ============================================================
# Plot 4 — SBI  (tall horizontal bar chart)
# ============================================================
sbi_all      = agg["SBI"]
sbi_filtered = {t: v for t, v in sbi_all.items()
                if isinstance(v, dict) and v.get("total_classified", 0) >= 10}
sbi_sorted   = sorted(sbi_filtered.items(), key=lambda x: x[1]["sbi"])
n_topics     = len(sbi_sorted)
fig_h        = max(6, n_topics * 0.38)

fig = _new_fig(13, fig_h, "2a. SBI — Stance Balance Index per Topic")
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL)

topics  = [t[:60] + "…" if len(t) > 60 else t for t, _ in sbi_sorted]
sbi_v   = [v["sbi"]             for _, v in sbi_sorted]
n_total = [v["total_classified"] for _, v in sbi_sorted]
pos_c   = [v["pos_count"]        for _, v in sbi_sorted]
neg_c   = [v["neg_count"]        for _, v in sbi_sorted]

cmap     = plt.cm.RdYlGn
bar_cols = [cmap(v) for v in sbi_v]
bars_sbi = ax.barh(topics, sbi_v, color=bar_cols, edgecolor=BG, linewidth=0.5, height=0.7)
for b, n, pc, nc in zip(bars_sbi, n_total, pos_c, neg_c):
    ax.text(b.get_width() + 0.007, b.get_y() + b.get_height()/2,
            f"n={n}  (+{pc}/-{nc})", va="center", color=MUTED, fontsize=8)
ax.set_xlim(0, 0.65)
ax.axvline(0.5, color=WARN, linewidth=0.9, linestyle=":", alpha=0.7)
ax.tick_params(axis='y', labelsize=8.5)
ax.tick_params(axis='x', labelsize=9)
_style(ax,
       "0 = completely one-sided  |  0.5 = perfectly balanced  |  min 10 classified comments\n"
       f"{SUB_LABEL}",
       xlabel="SBI")
fig.tight_layout()
_save(fig, "04_sbi.png")

# ============================================================
# Plot 5 — EAS
# ============================================================
fig = _new_fig(11, 6, "4a. EAS — Emotional Amplification Score")
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL)
emotions    = ["anger", "anxiety", "disgust"]
emo_colours = [BAD, WARN, "#ff6e67"]
x = np.arange(len(subreddits))
width = 0.24
for i, (emo, col) in enumerate(zip(emotions, emo_colours)):
    rhos = [by_sub[sr]["EAS"].get(emo, {}).get("rho") or 0.0 for sr in subreddits]
    p_vals = [by_sub[sr]["EAS"].get(emo, {}).get("p_value") for sr in subreddits]
    bars_e = ax.bar(x + i*width, rhos, width, label=emo,
                    color=col, edgecolor=BG, linewidth=0.5, alpha=0.85)
    for b, rho, pv in zip(bars_e, rhos, p_vals):
        marker = "*" if (pv is not None and pv < 0.05) else ""
        if abs(rho) > 0.005:
            ax.text(b.get_x() + b.get_width()/2, rho + 0.003,
                    f"{rho:.3f}{marker}", ha="center", va="bottom",
                    color="white", fontsize=8)
_hline(ax)
ax.set_xticks(x + width)
ax.set_xticklabels(subreddits, fontsize=11)
ax.legend(fontsize=9, facecolor=PANEL, labelcolor=MUTED, edgecolor=SPINE)
_style(ax,
       "Spearman rho between upvotes and emotion score  |  * = p < 0.05\n"
       f"Positive = more emotional posts get more upvotes  |  {SUB_LABEL}",
       ylabel="Spearman rho")
_save(fig, "05_eas.png")

# ============================================================
# Plot 6 — MSDG
# ============================================================
fig = _new_fig(9, 5, "2b. MSDG — Minority Stance Defensiveness Gap")
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL)
msdg_vals = [val(by_sub[sr]["MSDG"]) for sr in subreddits]
msdg_ns   = [_sample_size(by_sub[sr]["MSDG"]) for sr in subreddits]
colours   = [sub_colour(i, sr) for i, sr in enumerate(subreddits)]
valid = [(sr, v, n, c) for sr, v, n, c in zip(subreddits, msdg_vals, msdg_ns, colours)
         if v is not None and n >= 5]
if valid:
    sr_v, vv, nn, cc = zip(*valid)
    bars_msdg = ax.bar(sr_v, vv, color=cc, edgecolor=BG, linewidth=0.7, width=0.5)
    for b, n in zip(bars_msdg, nn):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
                f"n={n}", ha="center", color=MUTED, fontsize=9)
    _hline(ax)
    ax.tick_params(axis="x", labelsize=11)
else:
    ax.text(0.5, 0.5, "Insufficient minority stance data\n(need >= 5 eligible topics)",
            ha="center", va="center", color=MUTED, fontsize=11,
            transform=ax.transAxes)
_style(ax,
       "Positive = minority-stance comments use more defensive language\n"
       f"Topics excluded if SBI > {0.4} or minority n < 10  |  {SUB_LABEL}",
       ylabel="MSDG (mean defensiveness gap)")
_save(fig, "06_msdg.png")

# ============================================================
# Plot 7 — CSAD
# ============================================================
fig = _new_fig(12, 6, "4b. CSAD — Cross-Stance Anger Differential")
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL)
_direction_bars(ax, "CSAD",
                "Mean anger score per direction  (white dashed = aggregate)\n"
                f"Lower cross-stance vs baseline = healthier  |  {SUB_LABEL}",
                ylabel="Mean anger score")
_save(fig, "07_csad.png")

# ============================================================
# Plot 8 — TD
# ============================================================
fig = _new_fig(12, 6, "5. TD — Toxicity Differential")
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL)
_direction_bars(ax, "TD",
                "Mean toxicity score per direction  (white dashed = aggregate)\n"
                f"Lower cross-stance vs baseline = healthier  |  {SUB_LABEL}",
                ylabel="Mean toxicity score")
_save(fig, "08_td.png")

print(f"\nAll plots saved to {PLOT_DIR}")

