"""
Echo Chamber Metrics — V3 implementation.

Nine metrics operationalising the five behavioural antecedents of echo
chamber formation, per echo_chamber_metrics_V3.docx.

Metric inventory
----------------
1a  CSS   Counter-Stance Silence Rate
1b  CSEQ  Counter-Stance Engagement Quality
2a  SBI   Stance Balance Index          (per-topic, not aggregated)
2b  MSDG  Minority Stance Defensiveness Gap
3a  RDB   Reply Direction Bias
3b  uRDB  User-Level Reply Direction Bias
4a  EAS   Emotional Amplification Score
4b  CSAD  Cross-Stance Anger Differential
5   TD    Toxicity Differential

Shared design constants (all sourced from constants.py):
  * V3_STANCE_THRESHOLD          |stance| <= threshold → neutral, excluded
  * V3_MAJORITY_MIN_FRACTION     majority must be >= 60 % of classified comments
  * V3_MSDG_MIN_MINORITY_PER_TOPIC minimum minority comments per subreddit pool
  * V3_URDB_MIN_REPLIES_PER_USER minimum replies per user per thread for uRDB
  * V3_EAS_TOPIC_MIN_COMMENTS    minimum comments per topic for topic-level EAS
  * V3_EAS_BOOTSTRAP_ITERS       bootstrap iterations for EAS CIs

Data loading
------------
``V3Metrics.from_files(analysis_file, filtered_file)`` joins:
  - ``comment_analysis_NNN.json``   — LLM analysis output (this module)
  - ``comments/comments_filtered_NNN.json`` — scrape data (community, parent_id,
    upvotes, author, depth)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from ..constants import (
    V3_EAS_BOOTSTRAP_ITERS,
    V3_EAS_TOPIC_MIN_COMMENTS,
    V3_MAJORITY_MIN_FRACTION,
    V3_MSDG_MIN_MINORITY_PER_TOPIC,
    V3_STANCE_THRESHOLD,
    V3_TOPIC_MIN_COMMENTS,
    V3_URDB_MIN_REPLIES_PER_USER,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    name: str
    value: Optional[Any]
    sample_size: int
    notes: str = ""
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _numeric(val: Any) -> Optional[float]:
    """Return *val* as float, or None for "N/A", None, and non-numeric."""
    if val is None:
        return None
    if isinstance(val, str):
        if val.strip().upper() == "N/A":
            return None
        try:
            return float(val)
        except ValueError:
            return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _stance_sign(stance: Optional[float], threshold: float = V3_STANCE_THRESHOLD) -> Optional[int]:
    """Return +1, -1, or None (neutral / missing)."""
    if stance is None:
        return None
    if stance > threshold:
        return 1
    if stance < -threshold:
        return -1
    return None  # neutral


def _weighted_mean(
    values: List[float], weights: List[float]
) -> Optional[float]:
    """Weighted arithmetic mean; returns None if total weight is zero."""
    total_w = sum(weights)
    if total_w == 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / total_w


def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_iter: int = V3_EAS_BOOTSTRAP_ITERS,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    rhos = np.empty(n_iter)
    n = len(x)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        rhos[i] = spearmanr(x[idx], y[idx]).statistic
    lower = float(np.quantile(rhos, alpha / 2))
    upper = float(np.quantile(rhos, 1 - alpha / 2))
    return lower, upper


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_iter: int = V3_EAS_BOOTSTRAP_ITERS,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap percentile CI for the mean of *values*."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_iter)
    for i in range(n_iter):
        means[i] = values[rng.integers(0, n, size=n)].mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _bootstrap_diff_means_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_iter: int = V3_EAS_BOOTSTRAP_ITERS,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap percentile CI for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        diffs[i] = a[rng.integers(0, na, size=na)].mean() - b[rng.integers(0, nb, size=nb)].mean()
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class V3Metrics:
    """
    Compute V3 echo-chamber metrics from the comment analysis output.

    Parameters
    ----------
    comments:
        List of merged per-comment records.  Each record has the keys
        described in ``from_files``.
    """

    def __init__(self, comments: List[Dict[str, Any]]) -> None:
        self.comments = comments
        self._build_indexes()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        analysis_file: str,
        filtered_file: str,
    ) -> "V3Metrics":
        """
        Load and join analysis + scraped data.

        Parameters
        ----------
        analysis_file:
            Path to ``comment_analysis_NNN.json`` (list of LLM results).
        filtered_file:
            Path to ``comments/comments_filtered_NNN.json``.
        """
        # --- load analysis ---
        analysis_records: List[Dict] = json.loads(
            Path(analysis_file).read_text(encoding="utf-8")
        )
        analysis_map: Dict[str, Dict] = {}
        for rec in analysis_records:
            cid = rec.get("comment_id")
            if cid:
                analysis_map[cid] = rec

        # --- load filtered comments + community lookup ---
        raw = json.loads(Path(filtered_file).read_text(encoding="utf-8"))
        posts: List[Dict] = raw if isinstance(raw, list) else raw.get("posts", [])

        scraped_map: Dict[str, Dict] = {}   # comment_id → scraped fields
        post_community: Dict[str, str] = {}  # post_id → community

        for post in posts:
            pid = post.get("post_id", "")
            community = post.get("community", "")
            if pid:
                post_community[pid] = community
            for c in post.get("comments", []):
                cid = c.get("comment_id")
                if cid:
                    scraped_map[cid] = {
                        "parent_id": c.get("parent_id"),
                        "author": c.get("author"),
                        "upvotes": c.get("upvotes"),
                        "depth": c.get("depth"),
                        "post_id": c.get("post_id") or pid,
                        "community": community,
                    }

        # --- merge ---
        merged: List[Dict] = []
        for cid, analysis in analysis_map.items():
            scraped = scraped_map.get(cid, {})
            post_id = scraped.get("post_id") or analysis.get("post_id") or ""
            community = scraped.get("community") or post_community.get(post_id, "")

            topic_obj = analysis.get("topic") or {}
            stance_obj = topic_obj.get("stance") or {}
            ep_obj = analysis.get("epistemic_risk") or {}
            emotion_obj = analysis.get("emotion") or {}

            merged.append({
                "comment_id": cid,
                "post_id": post_id,
                "community": community,
                "author": scraped.get("author"),
                "parent_id": scraped.get("parent_id"),
                "upvotes": _numeric(scraped.get("upvotes")),
                "depth": scraped.get("depth"),
                # LLM analysis fields
                "topic": topic_obj.get("label") if isinstance(topic_obj, dict) else None,
                "stance": _numeric(stance_obj.get("value") if isinstance(stance_obj, dict) else None),
                "anger": _numeric(emotion_obj.get("anger")),
                "anxiety": _numeric(emotion_obj.get("anxiety")),
                "disgust": _numeric(emotion_obj.get("disgust")),
                "discrediting": _numeric(analysis.get("discrediting")),
                "defensiveness": _numeric(analysis.get("defensiveness")),
                "toxicity": _numeric(analysis.get("toxicity")),
                "evidence_quality": _numeric(ep_obj.get("evidence_quality") if isinstance(ep_obj, dict) else None),
                "reasoning_depth": _numeric(ep_obj.get("reasoning_depth") if isinstance(ep_obj, dict) else None),
            })

        return cls(merged)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_indexes(self) -> None:
        self._by_id: Dict[str, Dict] = {c["comment_id"]: c for c in self.comments}
        self._post_ids: set = {c["post_id"] for c in self.comments if c.get("post_id")}

        # reply_count: how many sampled comments reply to each comment
        self._reply_count: Dict[str, int] = {}
        # parent_to_replies: comment_id → list of child comment dicts
        self._parent_to_replies: Dict[str, List[Dict]] = {}
        for c in self.comments:
            pid = c.get("parent_id")
            if pid:
                self._reply_count[pid] = self._reply_count.get(pid, 0) + 1
                self._parent_to_replies.setdefault(pid, []).append(c)

        # thread_id: trace parent chain to find the top-level comment
        self._thread_id: Dict[str, str] = {}
        for c in self.comments:
            cid = c["comment_id"]
            self._thread_id[cid] = self._resolve_thread_root(cid)

        # annotate each comment record with thread_id
        for c in self.comments:
            c["thread_id"] = self._thread_id[c["comment_id"]]

        # annotate each comment with its absolute stance sign
        self._compute_abs_stance_signs()

    def _compute_abs_stance_signs(self) -> None:
        """
        Compute the absolute stance sign of every comment relative to the
        post/topic (not just its immediate parent) and store it as
        ``c["abs_stance_sign"]``.

        The stance field captured by the LLM is relative to the *immediate
        parent* comment.  For a chain  A → B → C:
          - A (depth-1): agrees with post  → abs = +1
          - B:  disagreement with A        → abs = -1  (opposes post)
          - C:  disagreement with B        → abs = +1  (double negation)

        Rule: abs_stance(node) = sign(raw_stance(node)) × abs_stance(parent)

        If the parent is the post (not in the comment index) or if any node
        in the chain lacks a clear stance, abs_stance_sign = None.
        """
        cache: Dict[str, Optional[int]] = {}

        def resolve(cid: str, visited: set) -> Optional[int]:
            if cid in cache:
                return cache[cid]
            if cid in visited:       # cycle guard
                cache[cid] = None
                return None
            node = self._by_id.get(cid)
            if node is None:
                cache[cid] = None
                return None
            raw = _stance_sign(node.get("stance"))
            if raw is None:
                cache[cid] = None
                return None
            pid = node.get("parent_id")
            if pid is None or pid in self._post_ids or pid not in self._by_id:
                # root comment — raw stance is already absolute
                cache[cid] = raw
                return raw
            parent_abs = resolve(pid, visited | {cid})
            if parent_abs is None:
                cache[cid] = None
                return None
            result = raw * parent_abs
            cache[cid] = result
            return result

        for c in self.comments:
            c["abs_stance_sign"] = resolve(c["comment_id"], set())

    def _resolve_thread_root(self, comment_id: str) -> str:
        visited: set = set()
        cid = comment_id
        while cid not in visited:
            visited.add(cid)
            node = self._by_id.get(cid)
            if node is None:
                break
            pid = node.get("parent_id")
            if pid is None or pid in self._post_ids or pid not in self._by_id:
                return cid  # cid is the root
            cid = pid
        return comment_id  # fallback (cycle guard)

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def _filter(self, subreddit: Optional[str] = None) -> List[Dict]:
        if subreddit is None:
            return self.comments
        return [c for c in self.comments if c.get("community") == subreddit]

    def subset(
        self,
        subreddit: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> "V3Metrics":
        """Return a new V3Metrics restricted to the given subreddit and/or topic."""
        filtered = self.comments
        if subreddit is not None:
            filtered = [c for c in filtered if c.get("community") == subreddit]
        if topic is not None:
            filtered = [c for c in filtered if c.get("topic") == topic]
        return V3Metrics(filtered)

    def get_subreddits(self) -> List[str]:
        return sorted({c["community"] for c in self.comments if c.get("community")})

    def get_topics(self, min_comments: int = 0) -> List[str]:
        """Return sorted topic labels with at least *min_comments* comments."""
        counts: Dict[str, int] = {}
        for c in self.comments:
            t = c.get("topic")
            if t:
                counts[t] = counts.get(t, 0) + 1
        return sorted(t for t, n in counts.items() if n >= min_comments)

    # ------------------------------------------------------------------
    # Grouping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by(comments: List[Dict], key: str) -> Dict[str, List[Dict]]:
        groups: Dict[str, List[Dict]] = {}
        for c in comments:
            k = c.get(key)
            if k is not None:
                groups.setdefault(str(k), []).append(c)
        return groups

    # ------------------------------------------------------------------
    # 1a. CSS — Counter-Stance Silence Rate
    # ------------------------------------------------------------------

    def _css_thread_data(
        self, subreddit: Optional[str] = None
    ) -> List[Tuple[float, int, str]]:
        """
        Internal helper: returns [(css_val, counter_count, topic), …] for all
        eligible threads in *pool*.  Used by both css() and css_by_topic().
        """
        pool = self._filter(subreddit)
        threads = self._group_by(pool, "thread_id")

        thread_data: List[Tuple[float, int, str]] = []

        for thread_id, thread_comments in threads.items():
            # Only consider replies (depth > 0); top-level comments directed at
            # the post are excluded — their silence is not a meaningful signal
            # of community suppression.
            classified = [
                (c, c.get("abs_stance_sign"))
                for c in thread_comments
                if c.get("abs_stance_sign") is not None
                and (c.get("depth") or 0) > 0
            ]
            if not classified:
                continue

            pos = [c for c, s in classified if s == 1]
            neg = [c for c, s in classified if s == -1]
            total_classified = len(classified)

            majority_n = max(len(pos), len(neg))
            if majority_n / total_classified < V3_MAJORITY_MIN_FRACTION:
                continue

            majority_sign = 1 if len(pos) >= len(neg) else -1
            counter = [c for c, s in classified if s != majority_sign]
            same = [c for c, s in classified if s == majority_sign]

            if not counter:
                continue

            # Build a stance lookup for all comments in this thread (all depths,
            # so replies outside the classified pool are still usable).
            stance_by_id = {
                c["comment_id"]: c.get("abs_stance_sign")
                for c in thread_comments
            }

            def _has_opposing_reply(comment: Dict, opposing_sign: int) -> bool:
                """Return True if any direct reply holds *opposing_sign* stance."""
                for reply in self._parent_to_replies.get(comment["comment_id"], []):
                    if stance_by_id.get(reply["comment_id"]) == opposing_sign:
                        return True
                return False

            # A counter-stance comment is "silenced" if it receives no
            # majority-stance replies (i.e. no pushback from the majority).
            silence_counter = sum(
                1 for c in counter
                if not _has_opposing_reply(c, majority_sign)
            ) / len(counter)

            # A same-stance comment is "silenced" if it receives no
            # counter-stance replies.
            silence_same = (
                sum(
                    1 for c in same
                    if not _has_opposing_reply(c, -majority_sign)
                ) / len(same)
                if same else 0.0
            )

            css_val = silence_counter - silence_same

            topic = next(
                (c.get("topic") for c in thread_comments if c.get("topic")),
                "UNKNOWN",
            )
            thread_data.append((css_val, len(counter), topic))

        return thread_data

    def css(self, subreddit: Optional[str] = None) -> MetricResult:
        """
        Counter-Stance Silence Rate.

        Fraction of counter-stance comments that receive no majority-stance
        reply, minus the same fraction for majority-stance comments.
        Positive = community ignores dissent more than it ignores agreement.
        Aggregated as a simple mean across eligible threads.
        """
        thread_data = self._css_thread_data(subreddit)

        if not thread_data:
            label = f" ({subreddit})" if subreddit else ""
            return MetricResult("CSS", None, 0, f"No eligible threads{label}")

        vals = [v for v, _, _ in thread_data]
        arr = np.array(vals)
        result = float(arr.mean())
        ci_lo, ci_hi = _bootstrap_mean_ci(arr)
        return MetricResult("CSS", result, len(thread_data), ci_lower=ci_lo, ci_upper=ci_hi)

    def css_by_topic(
        self, subreddit: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Per-topic CSS values.

        Returns a dict keyed by topic label.  Each value has:
        ``value`` (mean CSS for that topic across threads),
        ``n_counter`` (total counter-stance comments contributing),
        ``n_threads`` (number of eligible threads).
        """
        thread_data = self._css_thread_data(subreddit)
        by_topic: Dict[str, List[Tuple[float, int]]] = {}
        for v, n, topic in thread_data:
            by_topic.setdefault(topic, []).append((v, n))
        out: Dict[str, Dict[str, Any]] = {}
        for topic, vals in by_topic.items():
            out[topic] = {
                "value": float(np.mean([v for v, _ in vals])),
                "n_counter": int(sum(n for _, n in vals)),
                "n_threads": len(vals),
            }
        return out

    # ------------------------------------------------------------------
    # 1b. CSEQ — Counter-Stance Engagement Quality
    # ------------------------------------------------------------------

    def cseq(self, subreddit: Optional[str] = None) -> Dict[str, Any]:
        """
        Counter-Stance Engagement Quality.

        Returns a dict with keys: ``aggregate``, ``majority_to_minority``,
        ``minority_to_majority``.  Each is a dict with ``mean_discrediting``,
        ``mean_evidence_quality``, ``mean_reasoning_depth``,
        ``rho_discredit_evidence``, ``sample_size``.

        Majority / minority are determined per-thread using ``abs_stance_sign``:
        whichever polarity (+1 or -1) has more total upvotes in the thread is
        "majority".  Falls back to comment count when all upvotes are zero.

        Direction labels describe the *replier* relative to the *parent*:
          majority_to_minority — majority replier engaging with minority parent
          minority_to_majority — minority replier engaging with majority parent
        """
        pool = self._filter(subreddit)
        threads = self._group_by(pool, "thread_id")

        # Each record is one individual cross-stance reply
        records: List[Dict] = []

        for thread_id, thread_comments in threads.items():
            classified: Dict[str, int] = {
                c["comment_id"]: c.get("abs_stance_sign")  # type: ignore[assignment]
                for c in thread_comments
                if c.get("abs_stance_sign") is not None
            }
            if not classified:
                continue

            # determine majority by total upvotes on each side
            pos_ups = sum(
                c.get("upvotes") or 0
                for c in thread_comments
                if classified.get(c["comment_id"]) == 1
            )
            neg_ups = sum(
                c.get("upvotes") or 0
                for c in thread_comments
                if classified.get(c["comment_id"]) == -1
            )
            if pos_ups + neg_ups == 0:
                # fall back to comment count when all upvotes are zero
                pos_n = sum(1 for s in classified.values() if s == 1)
                neg_n = sum(1 for s in classified.values() if s == -1)
                if pos_n + neg_n == 0:
                    continue
                majority_sign = 1 if pos_n >= neg_n else -1
            else:
                majority_sign = 1 if pos_ups >= neg_ups else -1

            for c in thread_comments:
                cid = c["comment_id"]
                reply_sign = classified.get(cid)
                if reply_sign is None:
                    continue
                parent_sign = classified.get(c.get("parent_id", ""))
                if parent_sign is None:
                    continue
                if reply_sign == parent_sign:
                    continue  # same-stance: not a cross-stance reply

                # label by the replier's relation to majority
                direction = (
                    "majority_to_minority"
                    if reply_sign == majority_sign
                    else "minority_to_majority"
                )
                records.append({
                    "direction": direction,
                    "discrediting": _numeric(c.get("discrediting")),
                    "evidence_quality": _numeric(c.get("evidence_quality")),
                    "reasoning_depth": _numeric(c.get("reasoning_depth")),
                })

        def _summarise(recs: List[Dict]) -> Dict:
            if not recs:
                return {"sample_size": 0, "mean_discrediting": None,
                        "mean_evidence_quality": None, "mean_reasoning_depth": None,
                        "rho_discredit_evidence": None,
                        "ci_discrediting": [None, None],
                        "ci_evidence_quality": [None, None],
                        "ci_reasoning_depth": [None, None]}
            disc = [r["discrediting"] for r in recs if r["discrediting"] is not None]
            ev = [r["evidence_quality"] for r in recs if r["evidence_quality"] is not None]
            rd = [r["reasoning_depth"] for r in recs if r["reasoning_depth"] is not None]

            rho = None
            if len(disc) >= 5 and len(ev) >= 5:
                # align on records where both are present
                pairs = [(r["discrediting"], r["evidence_quality"])
                         for r in recs if r["discrediting"] is not None and r["evidence_quality"] is not None]
                if len(pairs) >= 5:
                    d_arr = np.array([p[0] for p in pairs])
                    e_arr = np.array([p[1] for p in pairs])
                    rho_val = spearmanr(d_arr, e_arr).statistic
                    rho = float(rho_val) if not np.isnan(rho_val) else None

            ci_disc = list(_bootstrap_mean_ci(np.array(disc))) if len(disc) >= 5 else [None, None]
            ci_ev   = list(_bootstrap_mean_ci(np.array(ev)))   if len(ev)   >= 5 else [None, None]
            ci_rd   = list(_bootstrap_mean_ci(np.array(rd)))   if len(rd)   >= 5 else [None, None]

            return {
                "sample_size": len(recs),
                "mean_discrediting": float(np.mean(disc)) if disc else None,
                "mean_evidence_quality": float(np.mean(ev)) if ev else None,
                "mean_reasoning_depth": float(np.mean(rd)) if rd else None,
                "rho_discredit_evidence": rho,
                "ci_discrediting": ci_disc,
                "ci_evidence_quality": ci_ev,
                "ci_reasoning_depth": ci_rd,
            }

        majority_to_minority = [r for r in records if r["direction"] == "majority_to_minority"]
        minority_to_majority = [r for r in records if r["direction"] == "minority_to_majority"]

        return {
            "aggregate": _summarise(records),
            "majority_to_minority": _summarise(majority_to_minority),
            "minority_to_majority": _summarise(minority_to_majority),
        }

    # ------------------------------------------------------------------
    # 2a. SBI — Stance Balance Index  (per-topic, not aggregated)
    # ------------------------------------------------------------------

    def sbi(self, subreddit: Optional[str] = None) -> Dict[str, Dict]:
        """
        Stance Balance Index, reported per topic.

        Returns a dict keyed by topic label.  Each value has:
        ``sbi``, ``pos_count``, ``neg_count``, ``total_classified``.
        """
        pool = self._filter(subreddit)
        by_topic = self._group_by(pool, "topic")
        out: Dict[str, Dict] = {}

        for topic, comments in by_topic.items():
            pos = sum(1 for c in comments if c.get("abs_stance_sign") == 1)
            neg = sum(1 for c in comments if c.get("abs_stance_sign") == -1)
            total = pos + neg
            sbi_val = min(pos, neg) / total if total > 0 else None
            out[topic] = {
                "sbi": sbi_val,
                "pos_count": pos,
                "neg_count": neg,
                "total_classified": total,
            }

        return out

    # ------------------------------------------------------------------
    # 2b. MSDG — Minority Stance Defensiveness Gap
    # ------------------------------------------------------------------

    def msdg(self, subreddit: Optional[str] = None) -> MetricResult:
        """
        Minority Stance Defensiveness Gap.

        Positive value → minority-stance comments use more defensive language.
        Computed directly across all classified comments in the pool (no
        per-topic grouping).  Requires at least V3_MSDG_MIN_MINORITY_PER_TOPIC
        minority-stance comments with a defensiveness score.
        """
        pool = self._filter(subreddit)

        pos = [(c, _numeric(c.get("defensiveness")))
               for c in pool if c.get("abs_stance_sign") == 1]
        neg = [(c, _numeric(c.get("defensiveness")))
               for c in pool if c.get("abs_stance_sign") == -1]

        minority, majority = (pos, neg) if len(pos) <= len(neg) else (neg, pos)

        minority_vals = [d for _, d in minority if d is not None]
        majority_vals = [d for _, d in majority if d is not None]

        label = f" ({subreddit})" if subreddit else ""
        if len(minority_vals) < V3_MSDG_MIN_MINORITY_PER_TOPIC:
            return MetricResult(
                "MSDG", None, 0,
                f"Insufficient minority defensiveness data{label} (n={len(minority_vals)})"
            )
        if not majority_vals:
            return MetricResult("MSDG", None, 0, f"No majority defensiveness data{label}")

        gap = float(np.mean(minority_vals)) - float(np.mean(majority_vals))
        ci_lo, ci_hi = _bootstrap_diff_means_ci(
            np.array(minority_vals), np.array(majority_vals)
        )
        return MetricResult("MSDG", gap, len(minority_vals), ci_lower=ci_lo, ci_upper=ci_hi)

    # ------------------------------------------------------------------
    # 3a. RDB — Reply Direction Bias
    # ------------------------------------------------------------------

    def rdb(self, subreddit: Optional[str] = None) -> Dict[str, Any]:
        """
        Reply Direction Bias.

        Returns a dict with ``aggregate``, ``pro`` (same-stance rate for
        positive-stance parents), ``con`` (same for negative-stance parents),
        each as a ``MetricResult``.

        Positive aggregate value → users preferentially reply to same-stance
        content beyond what chance predicts.
        """
        pool = self._filter(subreddit)
        threads = self._group_by(pool, "thread_id")

        # per-direction lists: (rate_same, weight=1 per parent)
        agg_vals: List[float] = []
        pro_vals: List[float] = []
        con_vals: List[float] = []
        topic_agg: Dict[str, List[Tuple[float, int]]] = {}

        for thread_id, thread_comments in threads.items():
            classified = {
                c["comment_id"]: c.get("abs_stance_sign")
                for c in thread_comments
                if c.get("abs_stance_sign") is not None
            }
            if not classified:
                continue

            n_pos = sum(1 for s in classified.values() if s == 1)
            n_neg = sum(1 for s in classified.values() if s == -1)
            total = n_pos + n_neg
            if total < 2:
                continue

            p_pos = n_pos / total
            p_neg = n_neg / total
            expected_same = p_pos ** 2 + p_neg ** 2

            parent_rates_agg: List[float] = []
            parent_rates_pro: List[float] = []
            parent_rates_con: List[float] = []

            for c in thread_comments:
                cid = c["comment_id"]
                p_sign = classified.get(cid)
                if p_sign is None:
                    continue
                replies = self._parent_to_replies.get(cid, [])
                classified_replies = [
                    r for r in replies if classified.get(r["comment_id"]) is not None
                ]
                if not classified_replies:
                    continue

                n_same = sum(
                    1 for r in classified_replies
                    if classified.get(r["comment_id"]) == p_sign
                )
                rate_same = n_same / len(classified_replies)

                parent_rates_agg.append(rate_same)
                if p_sign == 1:
                    parent_rates_pro.append(rate_same)
                else:
                    parent_rates_con.append(rate_same)

            if not parent_rates_agg:
                continue

            observed_same = float(np.mean(parent_rates_agg))
            rdb_thread = observed_same - expected_same
            topic = next((c.get("topic") for c in thread_comments if c.get("topic")), "UNKNOWN")

            agg_vals.append(rdb_thread)
            topic_agg.setdefault(topic, []).append((rdb_thread, len(parent_rates_agg)))
            if parent_rates_pro:
                pro_vals.append(float(np.mean(parent_rates_pro)))
            if parent_rates_con:
                con_vals.append(float(np.mean(parent_rates_con)))

        def _res(name: str, vals: List[float]) -> MetricResult:
            if not vals:
                return MetricResult(name, None, 0, "No eligible threads")
            arr = np.array(vals)
            ci_lo, ci_hi = _bootstrap_mean_ci(arr)
            return MetricResult(name, float(arr.mean()), len(vals), ci_lower=ci_lo, ci_upper=ci_hi)

        # subreddit-level aggregate via topic-weighted mean
        t_vals, t_wts = [], []
        total_obs = 0
        for tvs in topic_agg.values():
            tv = _weighted_mean([v for v, _ in tvs], [w for _, w in tvs])
            tw = sum(w for _, w in tvs)
            if tv is not None:
                t_vals.append(tv)
                t_wts.append(tw)
                total_obs += int(tw)

        agg_result = MetricResult(
            "RDB_aggregate",
            _weighted_mean(t_vals, t_wts),
            total_obs,
        )

        return {
            "aggregate": agg_result,
            "pro": _res("RDB_pro", pro_vals),
            "con": _res("RDB_con", con_vals),
        }

    # ------------------------------------------------------------------
    # 3b. uRDB — User-Level Reply Direction Bias
    # ------------------------------------------------------------------

    def urdb(self, subreddit: Optional[str] = None) -> MetricResult:
        """
        User-Level Reply Direction Bias.

        For each user with >= V3_URDB_MIN_REPLIES_PER_USER classified replies
        in a thread, compute proportion of replies directed at same-stance
        parents.  Aggregated via weighted mean (weight = eligible user count
        per topic).
        """
        pool = self._filter(subreddit)
        threads = self._group_by(pool, "thread_id")

        topic_urdb: Dict[str, List[Tuple[float, int]]] = {}

        for thread_id, thread_comments in threads.items():
            classified = {
                c["comment_id"]: c.get("abs_stance_sign")
                for c in thread_comments
                if c.get("abs_stance_sign") is not None
            }
            if not classified:
                continue

            # group replies by author
            by_author: Dict[str, List[Dict]] = {}
            for c in thread_comments:
                author = c.get("author")
                if not author:
                    continue
                pid = c.get("parent_id")
                if pid and classified.get(c["comment_id"]) is not None:
                    by_author.setdefault(author, []).append(c)

            user_rates: List[float] = []
            for author, replies in by_author.items():
                if len(replies) < V3_URDB_MIN_REPLIES_PER_USER:
                    continue
                n_same = sum(
                    1 for r in replies
                    if (r_sign := classified.get(r["comment_id"])) is not None
                    and (p_sign := classified.get(r.get("parent_id", ""))) is not None
                    and r_sign == p_sign
                )
                user_rates.append(n_same / len(replies))

            if not user_rates:
                continue

            thread_urdb = float(np.mean(user_rates))
            topic = next((c.get("topic") for c in thread_comments if c.get("topic")), "UNKNOWN")
            topic_urdb.setdefault(topic, []).append((thread_urdb, len(user_rates)))

        if not topic_urdb:
            label = f" ({subreddit})" if subreddit else ""
            return MetricResult("uRDB", None, 0, f"No eligible threads{label}")

        t_vals, t_wts = [], []
        total_obs = 0
        all_urdb_rates: List[float] = []
        for tvs in topic_urdb.values():
            tv = _weighted_mean([v for v, _ in tvs], [w for _, w in tvs])
            tw = sum(w for _, w in tvs)
            if tv is not None:
                t_vals.append(tv)
                t_wts.append(tw)
                total_obs += int(tw)
                all_urdb_rates.extend([v for v, _ in tvs])

        result_val = _weighted_mean(t_vals, t_wts)
        if result_val is None:
            return MetricResult("uRDB", None, 0, "No eligible threads")
        ci_lo, ci_hi = _bootstrap_mean_ci(np.array(all_urdb_rates))
        return MetricResult("uRDB", result_val, total_obs, ci_lower=ci_lo, ci_upper=ci_hi)

    # ------------------------------------------------------------------
    # 4a. EAS — Emotional Amplification Score
    # ------------------------------------------------------------------

    def eas(
        self,
        subreddit: Optional[str] = None,
        by_topic: bool = False,
    ) -> Dict[str, Any]:
        """
        Emotional Amplification Score.

        Returns a dict keyed by emotion (``anger``, ``anxiety``, ``disgust``),
        each with ``rho``, ``p_value``, ``n``, ``ci_lower``, ``ci_upper``.

        When *by_topic* is True, also returns a ``topics`` sub-dict
        (secondary analysis, only for topics with >= V3_EAS_TOPIC_MIN_COMMENTS).
        """
        pool = self._filter(subreddit)

        def _compute_eas(comments: List[Dict], label: str) -> Dict:
            result: Dict[str, Any] = {}
            for emotion in ("anger", "anxiety", "disgust"):
                pairs = [
                    (c["upvotes"], c[emotion])
                    for c in comments
                    if c.get("upvotes") is not None and c.get(emotion) is not None
                ]
                if len(pairs) < 5:
                    result[emotion] = {"rho": None, "p_value": None, "n": len(pairs),
                                       "ci_lower": None, "ci_upper": None}
                    continue
                upv = np.array([p[0] for p in pairs])
                emo = np.array([p[1] for p in pairs])
                sp = spearmanr(upv, emo)
                rho = float(sp.statistic)
                p_val = float(sp.pvalue)
                ci_lo, ci_hi = _bootstrap_spearman_ci(upv, emo)
                result[emotion] = {
                    "rho": rho,
                    "p_value": p_val,
                    "n": len(pairs),
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                }
            return result

        out = _compute_eas(pool, subreddit or "all")

        if by_topic:
            by_topic_data: Dict[str, Dict] = {}
            for topic, tcomments in self._group_by(pool, "topic").items():
                if len(tcomments) >= V3_EAS_TOPIC_MIN_COMMENTS:
                    by_topic_data[topic] = _compute_eas(tcomments, topic)
            out["topics"] = by_topic_data

        return out

    # ------------------------------------------------------------------
    # 4b. CSAD — Cross-Stance Anger Differential
    # ------------------------------------------------------------------

    def csad(self, subreddit: Optional[str] = None) -> Dict[str, Any]:
        """
        Cross-Stance Anger Differential.

        Returns ``aggregate``, ``majority_to_minority``, ``minority_to_majority``,
        ``same_baseline`` as ``MetricResult`` objects.

        Each cross-stance reply is labelled by the *replier's* relation to the
        thread majority (determined by total upvotes, falls back to comment count).
        Positive aggregate → cross-stance replies carry more anger than same-stance.
        """
        return self._stance_emotion_metric(
            metric_name="CSAD",
            emotion_field="anger",
            subreddit=subreddit,
        )

    # ------------------------------------------------------------------
    # 5. TD — Toxicity Differential
    # ------------------------------------------------------------------

    def td(self, subreddit: Optional[str] = None) -> Dict[str, Any]:
        """
        Toxicity Differential.

        Identical structure to CSAD but uses ``toxicity`` instead of ``anger``.
        """
        return self._stance_emotion_metric(
            metric_name="TD",
            emotion_field="toxicity",
            subreddit=subreddit,
        )

    # ------------------------------------------------------------------
    # Shared CSAD / TD engine
    # ------------------------------------------------------------------

    def _stance_emotion_metric(
        self,
        metric_name: str,
        emotion_field: str,
        subreddit: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Iterates individual replies.  For each reply whose ``abs_stance_sign``
        differs from its parent's, the direction is labelled by the *replier's*
        relation to the thread majority (determined by total upvotes per side,
        with comment-count fallback).  Emotion value is taken from the replier.
        """
        pool = self._filter(subreddit)
        threads = self._group_by(pool, "thread_id")

        majority_to_minority_vals: List[float] = []
        minority_to_majority_vals: List[float] = []
        same_vals: List[float] = []

        topic_cross: Dict[str, List[Tuple[float, int]]] = {}
        topic_same: Dict[str, List[Tuple[float, int]]] = {}

        for thread_id, thread_comments in threads.items():
            classified = {
                c["comment_id"]: c.get("abs_stance_sign")
                for c in thread_comments
                if c.get("abs_stance_sign") is not None
            }
            if not classified:
                continue

            # determine majority by total upvotes on each side
            pos_ups = sum(
                c.get("upvotes") or 0
                for c in thread_comments
                if classified.get(c["comment_id"]) == 1
            )
            neg_ups = sum(
                c.get("upvotes") or 0
                for c in thread_comments
                if classified.get(c["comment_id"]) == -1
            )
            if pos_ups + neg_ups == 0:
                n_pos = sum(1 for s in classified.values() if s == 1)
                n_neg = sum(1 for s in classified.values() if s == -1)
                majority_sign = 1 if n_pos >= n_neg else -1
            else:
                majority_sign = 1 if pos_ups >= neg_ups else -1

            for c in thread_comments:
                cid = c["comment_id"]
                reply_sign = classified.get(cid)
                if reply_sign is None:
                    continue
                parent_sign = classified.get(c.get("parent_id", ""))
                if parent_sign is None:
                    continue

                emotion = _numeric(c.get(emotion_field))
                if emotion is None:
                    continue

                topic = c.get("topic") or "UNKNOWN"

                if reply_sign != parent_sign:
                    # cross-stance reply
                    topic_cross.setdefault(topic, []).append((emotion, 1))
                    if reply_sign == majority_sign:
                        majority_to_minority_vals.append(emotion)
                    else:
                        minority_to_majority_vals.append(emotion)
                else:
                    # same-stance reply
                    same_vals.append(emotion)
                    topic_same.setdefault(topic, []).append((emotion, 1))

        def _agg_via_topics(tdict: Dict[str, List[Tuple[float, int]]]) -> Tuple[Optional[float], int]:
            t_vals, t_wts = [], []
            tot = 0
            for tvs in tdict.values():
                tv = _weighted_mean([v for v, _ in tvs], [w for _, w in tvs])
                tw = sum(w for _, w in tvs)
                if tv is not None:
                    t_vals.append(tv)
                    t_wts.append(tw)
                    tot += int(tw)
            return _weighted_mean(t_vals, t_wts), tot

        cross_mean, cross_n = _agg_via_topics(topic_cross)
        same_mean, same_n = _agg_via_topics(topic_same)

        aggregate_val = (
            (cross_mean - same_mean)
            if cross_mean is not None and same_mean is not None
            else None
        )

        def _mr(name: str, vals: List[float]) -> MetricResult:
            if not vals:
                return MetricResult(name, None, 0, "No eligible replies")
            arr = np.array(vals)
            ci_lo, ci_hi = _bootstrap_mean_ci(arr)
            return MetricResult(name, float(arr.mean()), len(vals), ci_lower=ci_lo, ci_upper=ci_hi)

        all_cross = majority_to_minority_vals + minority_to_majority_vals
        agg_ci_lo, agg_ci_hi = None, None
        if all_cross and same_vals:
            agg_ci_lo, agg_ci_hi = _bootstrap_diff_means_ci(
                np.array(all_cross), np.array(same_vals)
            )

        return {
            "aggregate": MetricResult(
                f"{metric_name}_aggregate", aggregate_val, cross_n + same_n,
                ci_lower=agg_ci_lo, ci_upper=agg_ci_hi
            ),
            "majority_to_minority": _mr(f"{metric_name}_majority_to_minority", majority_to_minority_vals),
            "minority_to_majority": _mr(f"{metric_name}_minority_to_majority", minority_to_majority_vals),
            "same_baseline": MetricResult(
                f"{metric_name}_same", same_mean, same_n
            ),
        }

    # ------------------------------------------------------------------
    # compute_all
    # ------------------------------------------------------------------

    def compute_all(
        self, subreddit: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run all nine V3 metrics for *subreddit* (or the full dataset)."""
        return {
            "CSS": self.css(subreddit),
            "CSEQ": self.cseq(subreddit),
            "SBI": self.sbi(subreddit),
            "MSDG": self.msdg(subreddit),
            "RDB": self.rdb(subreddit),
            "uRDB": self.urdb(subreddit),
            "EAS": self.eas(subreddit, by_topic=True),
            "CSAD": self.csad(subreddit),
            "TD": self.td(subreddit),
        }

    def compute_all_by_subreddit(self) -> Dict[str, Dict[str, Any]]:
        """Run all nine metrics for each subreddit separately."""
        return {sr: self.compute_all(sr) for sr in self.get_subreddits()}

    def compute_all_by_subreddit_topic(
        self, min_comments: int = V3_TOPIC_MIN_COMMENTS
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Run all nine metrics for each (subreddit, topic) pair.

        Only topics with >= *min_comments* comments are included.  Topics
        that don't meet the threshold within the subreddit slice are silently
        skipped — they will still appear in the subreddit-level results.
        """
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for sr in self.get_subreddits():
            sr_sub = self.subset(subreddit=sr)
            topics = sr_sub.get_topics(min_comments=min_comments)
            if not topics:
                continue
            result[sr] = {}
            for topic in topics:
                topic_sub = sr_sub.subset(topic=topic)
                result[sr][topic] = topic_sub.compute_all()
        return result
