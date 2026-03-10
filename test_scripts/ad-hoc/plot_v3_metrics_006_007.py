"""
V3 metrics visualisation — scrape_006 + scrape_007 combined (9 subreddits).

Reads pre-computed metrics JSON files from both scrapes and plots all metrics
together, colour-coded by scrape. A divider is drawn between the two groups.

Outputs PNGs to:  data/output/scrape_006_scrape_007/v3_plots/

Usage (from project root):
    .\\venv\\Scripts\\python.exe test_scripts\\ad-hoc\\plot_v3_metrics_006_007.py
"""

import json
import sys
from dataclasses import is_dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("data/output/scrape_006_scrape_007")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_006 = Path("data/output/scrape_006/v3_metrics_003.json")
METRICS_007 = Path("data/output/scrape_007/v3_metrics_008.json")

# ---------------------------------------------------------------------------
# Load pre-computed metrics
# ---------------------------------------------------------------------------
print("Loading metrics …")
d6 = json.load(open(METRICS_006, encoding="utf-8"))
d7 = json.load(open(METRICS_007, encoding="utf-8"))

# Per-subreddit data: merge both dicts
by_sub_6 = d6["by_subreddit"]   # Christianity, antiwork, atheism, conservative
by_sub_7 = d7["by_subreddit"]   # HubermanLab, decodingthegurus, lexfridman, philosophy, samharris

# Ordered: 006 subs first, then 007 subs
SUBS_006 = ["antiwork", "atheism", "Christianity", "conservative"]
SUBS_007 = ["decodingthegurus", "HubermanLab", "lexfridman", "philosophy", "samharris"]
subreddits = SUBS_006 + SUBS_007
n_006 = len(SUBS_006)
n_007 = len(SUBS_007)

by_sub = {**by_sub_6, **by_sub_7}

# Aggregates — used for SBI topic chart and EAS
agg6 = d6["aggregate"]
agg7 = d7["aggregate"]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
BG     = "#1c1c2e"
PANEL  = "#12122a"
ACCENT = "#7c7cff"
MUTED  = "#aaaacc"
SPINE  = "#444466"
GOOD   = "#50fa7b"
BAD    = "#ff5555"
WARN   = "#f1fa8c"
DIVIDER = "#666688"

# 006 subs — warm/earthy
COLOURS_006 = {
    "antiwork":    "#f8961e",
    "atheism":     "#f3722c",
    "Christianity":"#f94144",
    "conservative":"#90be6d",
}
# 007 subs — cool/dracula palette (same as plot_v3_metrics_007.py)
COLOURS_007 = {
    "decodingthegurus": "#bd93f9",
    "HubermanLab":      "#50fa7b",
    "lexfridman":       "#ffb86c",
    "philosophy":       "#8be9fd",
    "samharris":        "#ff79c6",
}
ALL_COLOURS = {**COLOURS_006, **COLOURS_007}

def sub_colour(sr):
    return ALL_COLOURS.get(sr, MUTED)

SHORT = {
    "decodingthegurus": "DecodingThe\nGurus",
    "Christianity":     "Christianity",
    "conservative":     "conservative",
}

def sub_label(sr):
    return SHORT.get(sr, sr)

SUB_LABELS = [sub_label(s) for s in subreddits]
colours_all = [sub_colour(sr) for sr in subreddits]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def val(d):
    if d is None:
        return None
    if is_dataclass(d) and not isinstance(d, type):
        return d.value
    if isinstance(d, dict) and "value" in d:
        return d["value"]
    return None

def _sample_size(d):
    if d is None:
        return 0
    if is_dataclass(d) and not isinstance(d, type):
        return getattr(d, "sample_size", 0)
    if isinstance(d, dict):
        return d.get("sample_size", 0)
    return 0

def v0(d):
    v = val(d)
    return 0.0 if v is None else v

def _bar_colour(v, neutral=0.0, flip=False):
    if v is None:
        return MUTED
    delta = v - neutral
    if flip:
        delta = -delta
    return GOOD if delta <= 0 else BAD

def _ci(d):
    if d is None:
        return None, None
    if is_dataclass(d) and not isinstance(d, type):
        return getattr(d, "ci_lower", None), getattr(d, "ci_upper", None)
    if isinstance(d, dict):
        return d.get("ci_lower"), d.get("ci_upper")
    return None, None

# ---------------------------------------------------------------------------
# Axis styling helpers
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

def _scrape_divider(ax, x_axis=True):
    """Draw a vertical line between scrape_006 and scrape_007 subreddit groups."""
    split = n_006 - 0.5
    if x_axis:
        ax.axvline(split, color=DIVIDER, linewidth=1.0, linestyle=":", alpha=0.6)

def _scrape_labels(ax, y_frac=1.02):
    """Add 006 / 007 group annotations above the x-axis."""
    ax.annotate("scrape_006", xy=(n_006 / 2 - 0.5, y_frac), xycoords=("data", "axes fraction"),
                ha="center", color=COLOURS_006["antiwork"], fontsize=8, alpha=0.8)
    ax.annotate("scrape_007", xy=(n_006 + n_007 / 2 - 0.5, y_frac), xycoords=("data", "axes fraction"),
                ha="center", color=COLOURS_007["decodingthegurus"], fontsize=8, alpha=0.8)

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
PLOT_DIR = OUTPUT_DIR / "v3_plots"
PLOT_DIR.mkdir(exist_ok=True)
saved = []

def _new_fig(w=13, h=6, title=""):
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
    print(f"  Saved -> {path.name}")

# ---------------------------------------------------------------------------
# Direction-bars helper (used for CSAD and TD)
# No aggregate line — two separate scrape pools, no single valid aggregate
# ---------------------------------------------------------------------------
directions  = ["majority_to_minority", "minority_to_majority", "same_baseline"]
dir_labels  = ["Maj->Min", "Min->Maj", "Same-stance baseline"]
dir_colours = [BAD, ACCENT, WARN]

def _direction_bars(ax, metric_key, title, ylabel=""):
    x = np.arange(len(subreddits))
    width = 0.22
    for i, (dir_k, lbl, col) in enumerate(zip(directions, dir_labels, dir_colours)):
        raw_d  = [val(by_sub[sr][metric_key].get(dir_k)) for sr in subreddits]
        vals_d = [v if v is not None else 0.0 for v in raw_d]
        cis_d  = [_ci(by_sub[sr][metric_key].get(dir_k)) for sr in subreddits]
        ax.bar(x + i*width, vals_d, width, label=lbl,
               color=col, edgecolor=BG, linewidth=0.5, alpha=0.82)
        err_lo = [abs(v - lo) if (v is not None and lo is not None) else 0
                  for v, (lo, hi) in zip(raw_d, cis_d)]
        err_hi = [abs(hi - v) if (v is not None and hi is not None) else 0
                  for v, (lo, hi) in zip(raw_d, cis_d)]
        ax.errorbar(x + i*width, vals_d, yerr=[err_lo, err_hi], fmt="none",
                    ecolor="white", elinewidth=1.2, capsize=4, capthick=1.2, alpha=0.65)
    _hline(ax)
    ax.set_xticks(x + width)
    ax.set_xticklabels(SUB_LABELS, fontsize=9)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=MUTED, edgecolor=SPINE)
    _style(ax, title, ylabel=ylabel)

SUB_LABEL = "26,996 comments  |  9 subreddits"

# ============================================================
# Plot 1 — CSS
# ============================================================
print("\n[1] CSS")
fig = _new_fig(11, 5, "1a. CSS — Counter-Stance Silence Rate")
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL)
css_raw  = [val(by_sub[sr]["CSS"]) for sr in subreddits]
css_vals = [v if v is not None else 0.0 for v in css_raw]
css_cis  = [_ci(by_sub[sr]["CSS"]) for sr in subreddits]
bars = ax.bar(SUB_LABELS, css_vals, color=colours_all, edgecolor=BG, linewidth=0.7)
err_lo = [abs(v - lo) if (v is not None and lo is not None) else 0 for v, (lo, hi) in zip(css_raw, css_cis)]
err_hi = [abs(hi - v) if (v is not None and hi is not None) else 0 for v, (lo, hi) in zip(css_raw, css_cis)]
x_pos  = [b.get_x() + b.get_width() / 2 for b in bars]
ax.errorbar(x_pos, css_vals, yerr=[err_lo, err_hi], fmt="none",
            ecolor="white", elinewidth=1.5, capsize=5, capthick=1.5, alpha=0.7)
_hline(ax)
for b, raw in zip(bars, css_raw):
    lbl = f"{raw:.3f}" if raw is not None else "n/a"
    yoff = b.get_height() + 0.005 if b.get_height() >= 0 else b.get_height() - 0.02
    ax.text(b.get_x() + b.get_width()/2, yoff, lbl,
            ha="center", va="bottom", color="white", fontsize=9)
ax.tick_params(axis="x", labelsize=9)
_style(ax, f"Positive = minority comments silenced more  |  error bars = 95% CI  |  {SUB_LABEL}",
       ylabel="silence_counter - silence_majority")
_save(fig, "1a.CSS.png")

# ============================================================
# Plot 2 — CSEQ
# ============================================================
print("[2] CSEQ")
fig = _new_fig(16, 6, "1b. CSEQ — Cross-Stance Engagement Quality by Direction")
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL)

cseq_dir_keys = ["majority_to_minority", "minority_to_majority"]
cseq_dir_lbls = ["Majority -> Minority", "Minority -> Majority"]
cseq_met_keys = ["mean_discrediting", "mean_evidence_quality", "mean_reasoning_depth"]
cseq_met_lbls = ["Discrediting", "Evid. Quality", "Reasoning Depth"]

n_sr  = len(subreddits)
n_dir = len(cseq_dir_keys)
n_met = len(cseq_met_keys)
group_w = 0.8
slot_w  = group_w / (n_dir * n_met)
x = np.arange(n_sr)

met_hex    = ["#e05050", "#5b9cf6", "#9b59b6"]
dir_alphas = [0.92, 0.50]
for mi, (mk, ml, mc) in enumerate(zip(cseq_met_keys, cseq_met_lbls, met_hex)):
    ci_key = "ci_" + mk.replace("mean_", "")
    for di, (dk, dl) in enumerate(zip(cseq_dir_keys, cseq_dir_lbls)):
        slot_idx = mi * n_dir + di
        offset = (slot_idx - (n_dir * n_met - 1) / 2) * slot_w
        raw_v  = [by_sub[sr]["CSEQ"].get(dk, {}).get(mk) for sr in subreddits]
        vals   = [v if v is not None else 0.0 for v in raw_v]
        ci_lst = [by_sub[sr]["CSEQ"].get(dk, {}).get(ci_key) or [None, None] for sr in subreddits]
        label  = f"{ml} | {dl}" if di == 0 else f"_ {ml} | {dl}"
        ax.bar(x + offset, vals, slot_w * 0.88,
               color=mc, edgecolor=BG, linewidth=0.4,
               alpha=dir_alphas[di], label=label)
        err_lo = [abs(v - lo) if (v is not None and lo is not None) else 0
                  for v, (lo, hi) in zip(raw_v, ci_lst)]
        err_hi = [abs(hi - v) if (v is not None and hi is not None) else 0
                  for v, (lo, hi) in zip(raw_v, ci_lst)]
        ax.errorbar(x + offset, vals, yerr=[err_lo, err_hi], fmt="none",
                    ecolor="white", elinewidth=1.0, capsize=3, capthick=1.0, alpha=0.6)

ax.set_xticks(x)
ax.set_xticklabels(SUB_LABELS, fontsize=9)
ax.legend(fontsize=8, facecolor=PANEL, labelcolor=MUTED, edgecolor=SPINE,
          loc="upper right", ncol=3)
ax.tick_params(axis="y", labelsize=9)
_style(ax,
       "Grouped by metric  |  Dark = Majority->Minority  Light = Minority->Majority\n"
       f"Red=Discrediting  Blue=Evid.Quality  Purple=Reasoning  |  error bars = 95% CI  |  {SUB_LABEL}",
       ylabel="Mean score")
_save(fig, "1b.CSEQ.png")

# ============================================================
# Plot 3a — RDB
# ============================================================
print("[3a] RDB")
fig = _new_fig(14, 6, "3a. RDB — Reply Direction Bias")
ax_rdb = fig.add_subplot(111)
ax_rdb.set_facecolor(PANEL)
x = np.arange(len(subreddits))
width = 0.26
rdb_series = [
    ("Majority (pos)",  [v0(by_sub[sr]["RDB"]["pro"])       for sr in subreddits],
                        [val(by_sub[sr]["RDB"]["pro"])      for sr in subreddits],
                        [_ci(by_sub[sr]["RDB"]["pro"])      for sr in subreddits], GOOD),
    ("Minority (neg)",  [v0(by_sub[sr]["RDB"]["con"])       for sr in subreddits],
                        [val(by_sub[sr]["RDB"]["con"])      for sr in subreddits],
                        [_ci(by_sub[sr]["RDB"]["con"])      for sr in subreddits], BAD),
    ("Aggregate bias",  [v0(by_sub[sr]["RDB"]["aggregate"]) for sr in subreddits],
                        None, None, ACCENT),
]
for i, (lbl, vals_r, raw_r, cis_r, col) in enumerate(rdb_series):
    ax_rdb.bar(x + i*width, vals_r, width, label=lbl,
               color=col, edgecolor=BG, linewidth=0.5, alpha=0.85)
    if cis_r is not None:
        err_lo = [abs(v - lo) if (v is not None and lo is not None) else 0
                  for v, (lo, hi) in zip(raw_r, cis_r)]
        err_hi = [abs(hi - v) if (v is not None and hi is not None) else 0
                  for v, (lo, hi) in zip(raw_r, cis_r)]
        ax_rdb.errorbar(x + i*width, vals_r, yerr=[err_lo, err_hi], fmt="none",
                        ecolor="white", elinewidth=1.2, capsize=4, capthick=1.2, alpha=0.65)
ax_rdb.axhline(0.5, color=SPINE, linewidth=0.8, linestyle="--", alpha=0.5)
_scrape_divider(ax_rdb)
ax_rdb.set_xticks(x + width)
ax_rdb.set_xticklabels(SUB_LABELS, fontsize=10)
ax_rdb.legend(fontsize=9, facecolor=PANEL, labelcolor=MUTED, edgecolor=SPINE)
_style(ax_rdb, "Fraction of replies directed at same-stance comments  |  error bars = 95% CI on pro/con",
       ylabel="Same-stance reply fraction")
_save(fig, "3a.RDB.png")

# ============================================================
# Plot 3b — uRDB
# ============================================================
print("[3b] uRDB")
fig = _new_fig(11, 5, "3b. uRDB — User-Level Reply Direction Bias")
ax_urdb = fig.add_subplot(111)
ax_urdb.set_facecolor(PANEL)
urdb_raw  = [val(by_sub[sr]["uRDB"]) for sr in subreddits]
urdb_vals = [v if v is not None else 0.0 for v in urdb_raw]
urdb_cis  = [_ci(by_sub[sr]["uRDB"]) for sr in subreddits]
urdb_bars = ax_urdb.bar(SUB_LABELS, urdb_vals, color=colours_all, edgecolor=BG, linewidth=0.7, width=0.5)
urdb_err_lo = [abs(v - lo) if (v is not None and lo is not None) else 0
               for v, (lo, hi) in zip(urdb_raw, urdb_cis)]
urdb_err_hi = [abs(hi - v) if (v is not None and hi is not None) else 0
               for v, (lo, hi) in zip(urdb_raw, urdb_cis)]
urdb_xpos = [b.get_x() + b.get_width() / 2 for b in urdb_bars]
ax_urdb.errorbar(urdb_xpos, urdb_vals, yerr=[urdb_err_lo, urdb_err_hi], fmt="none",
                 ecolor="white", elinewidth=1.5, capsize=5, capthick=1.5, alpha=0.7)
ax_urdb.axhline(0.5, color=SPINE, linewidth=0.8, linestyle="--", alpha=0.5)
ax_urdb.legend(fontsize=9, facecolor=PANEL, labelcolor=MUTED, edgecolor=SPINE)
for b, v in zip(urdb_bars, urdb_vals):
    ax_urdb.text(b.get_x() + b.get_width()/2, v + 0.01,
                 f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=10)
ax_urdb.tick_params(axis="x", labelsize=10)
_style(ax_urdb, "Per-user bias averaged within thread  |  error bars = 95% CI",
       ylabel="Mean user same-stance fraction")
_save(fig, "3b.uRDB.png")

# ============================================================
# Plot 4 — SBI  (merge both scrape topic dicts)
# ============================================================
print("[4] SBI")
sbi_combined = {}
for t, v in agg6.get("SBI", {}).items():
    if isinstance(v, dict):
        sbi_combined[t] = v
for t, v in agg7.get("SBI", {}).items():
    if isinstance(v, dict):
        sbi_combined[t] = v

sbi_filtered = {t: v for t, v in sbi_combined.items()
                if v.get("total_classified", 0) >= 10}
sbi_sorted   = sorted(sbi_filtered.items(), key=lambda kv: kv[1]["sbi"])
n_topics     = len(sbi_sorted)
fig_h        = max(6, n_topics * 0.38)

fig = _new_fig(13, fig_h, "2a. SBI — Stance Balance Index per Topic")
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL)

topics  = [t[:65] + "…" if len(t) > 65 else t for t, _ in sbi_sorted]
sbi_v   = [v["sbi"]              for _, v in sbi_sorted]
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
ax.tick_params(axis="y", labelsize=8.5)
ax.tick_params(axis="x", labelsize=9)
_style(ax,
       f"0 = completely one-sided  |  0.5 = perfectly balanced  |  min 10 classified comments\n{SUB_LABEL}",
       xlabel="SBI")
fig.tight_layout()
_save(fig, "2a.SBI.png")

# ============================================================
# Plot 5 — EAS  (per-subreddit, merged palette)
# ============================================================
print("[5] EAS")
fig = _new_fig(14, 6, "4a. EAS — Emotional Amplification Score")
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL)
emotions    = ["anger", "anxiety", "disgust"]
emo_colours = [BAD, WARN, "#ff6e67"]
x = np.arange(len(subreddits))
width = 0.24
for i, (emo, col) in enumerate(zip(emotions, emo_colours)):
    rhos   = [by_sub[sr]["EAS"].get(emo, {}).get("rho") or 0.0 for sr in subreddits]
    p_vals = [by_sub[sr]["EAS"].get(emo, {}).get("p_value") for sr in subreddits]
    bars_e = ax.bar(x + i*width, rhos, width, label=emo,
                    color=col, edgecolor=BG, linewidth=0.5, alpha=0.85)
    raw_rhos = [by_sub[sr]["EAS"].get(emo, {}).get("rho") for sr in subreddits]
    eas_cis  = [by_sub[sr]["EAS"].get(emo, {}) for sr in subreddits]
    eas_lo   = [abs(v - d.get("ci_lower")) if (v is not None and d.get("ci_lower") is not None) else 0
                for v, d in zip(raw_rhos, eas_cis)]
    eas_hi   = [abs(d.get("ci_upper") - v) if (v is not None and d.get("ci_upper") is not None) else 0
                for v, d in zip(raw_rhos, eas_cis)]
    ax.errorbar(x + i*width, rhos, yerr=[eas_lo, eas_hi], fmt="none",
                ecolor="white", elinewidth=1.2, capsize=4, capthick=1.2, alpha=0.65)
    for b, rho, pv in zip(bars_e, rhos, p_vals):
        marker = "*" if (pv is not None and pv < 0.05) else ""
        if abs(rho) > 0.005:
            ax.text(b.get_x() + b.get_width()/2, rho + (0.003 if rho >= 0 else -0.012),
                    f"{rho:.3f}{marker}", ha="center",
                    va="bottom" if rho >= 0 else "top",
                    color="white", fontsize=8)
_hline(ax)
ax.set_xticks(x + width)
ax.set_xticklabels(SUB_LABELS, fontsize=9)
ax.legend(fontsize=9, facecolor=PANEL, labelcolor=MUTED, edgecolor=SPINE)
_style(ax,
       f"Spearman rho between upvotes and emotion score  |  * = p < 0.05  |  error bars = 95% CI",
       ylabel="Spearman rho")
_save(fig, "4a.EAS.png")

# ============================================================
# Plot 6 — MSDG
# ============================================================
print("[6] MSDG")
fig = _new_fig(12, 5, "2b. MSDG — Minority Stance Defensiveness Gap")
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL)
msdg_vals = [val(by_sub[sr]["MSDG"]) for sr in subreddits]
msdg_cis  = [_ci(by_sub[sr]["MSDG"]) for sr in subreddits]
msdg_ns   = [_sample_size(by_sub[sr]["MSDG"]) for sr in subreddits]
valid = [(sl, v, ci, n, c)
         for sl, v, ci, n, c in zip(SUB_LABELS, msdg_vals, msdg_cis, msdg_ns, colours_all)
         if v is not None and n >= 5]
if valid:
    sl_v, vv, cis_v, nn, cc = zip(*valid)
    bars_msdg = ax.bar(sl_v, vv, color=cc, edgecolor=BG, linewidth=0.7, width=0.5)
    err_lo = [abs(v - lo) if lo is not None else 0 for v, (lo, hi) in zip(vv, cis_v)]
    err_hi = [abs(hi - v) if hi is not None else 0 for v, (lo, hi) in zip(vv, cis_v)]
    x_pos  = [b.get_x() + b.get_width() / 2 for b in bars_msdg]
    ax.errorbar(x_pos, vv, yerr=[err_lo, err_hi], fmt="none",
                ecolor="white", elinewidth=1.5, capsize=5, capthick=1.5, alpha=0.7)
    for b, n in zip(bars_msdg, nn):
        yoff = b.get_height() + 0.003 if b.get_height() >= 0 else b.get_height() - 0.008
        ax.text(b.get_x() + b.get_width()/2, yoff,
                f"n={n}", ha="center", color=MUTED, fontsize=9)
    _hline(ax)
    ax.tick_params(axis="x", labelsize=9)
else:
    ax.text(0.5, 0.5, "Insufficient minority stance data",
            ha="center", va="center", color=MUTED, fontsize=11,
            transform=ax.transAxes)
_style(ax,
       f"Positive = minority-stance comments use more defensive language  |  error bars = 95% CI\n{SUB_LABEL}",
       ylabel="MSDG (mean defensiveness gap)")
_save(fig, "2b.MSDG.png")

# ============================================================
# Plot 7 — CSAD
# ============================================================
print("[7] CSAD")
fig = _new_fig(15, 6, "4b. CSAD — Cross-Stance Anger Differential")
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL)
_direction_bars(ax, "CSAD",
                f"Mean anger score per direction  |  error bars = 95% CI\nLower cross-stance vs baseline = healthier  |  {SUB_LABEL}",
                ylabel="Mean anger score")
_save(fig, "4b.CSAD.png")

# ============================================================
# Plot 8 — TD
# ============================================================
print("[8] TD")
fig = _new_fig(15, 6, "5. TD — Toxicity Differential")
ax = fig.add_subplot(111)
ax.set_facecolor(PANEL)
_direction_bars(ax, "TD",
                f"Mean toxicity score per direction  |  error bars = 95% CI\nLower cross-stance vs baseline = healthier  |  {SUB_LABEL}",
                ylabel="Mean toxicity score")
_save(fig, "5.TD.png")

print(f"\nAll plots saved to {PLOT_DIR}")
