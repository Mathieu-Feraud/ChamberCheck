"""Derived chamber-check metrics built from ABN model outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.stats import spearmanr


@dataclass
class MetricResult:
    name: str
    value: Optional[Union[float, Dict[str, Any]]]
    sample_size: int
    notes: str = ""


class CC_Metrics:
    """Compute derived CC metrics from ABN structured entries."""

    def __init__(
        self,
        entries: List[Dict[str, Any]],
        comment_index: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
        post_index: Optional[Dict[str, Dict[str, Any]]] = None,
        post_comment_authors: Optional[Dict[str, set]] = None,
    ):
        self.entries = entries
        self.comment_index = comment_index or {}
        self.post_index = post_index or {}
        self.post_comment_authors = post_comment_authors or {}

    @classmethod
    def from_abn_entries_file(cls, file_path: str) -> "CC_Metrics":
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, list):
            raise ValueError(f"Expected list payload in {file_path}")

        return cls(entries=payload)

    @classmethod
    def from_abn_llm_run_metadata(cls, metadata_file_path: str) -> "CC_Metrics":
        metadata_path = Path(metadata_file_path)
        with open(metadata_path, "r", encoding="utf-8") as file:
            run_metadata = json.load(file)

        entries_path = Path(str(run_metadata.get("output_file", "")))
        if not entries_path.is_absolute():
            entries_path = Path.cwd() / entries_path
        if not entries_path.exists():
            raise ValueError(f"Entries file not found from metadata: {entries_path}")

        entries = cls.from_abn_entries_file(str(entries_path)).entries

        prompts_metadata_value = run_metadata.get("metadata_json_path")
        prompts_metadata_path = Path(str(prompts_metadata_value)) if prompts_metadata_value else None
        if prompts_metadata_path and not prompts_metadata_path.is_absolute():
            prompts_metadata_path = Path.cwd() / prompts_metadata_path

        comment_index: Dict[str, Dict[str, Optional[str]]] = {}
        post_index: Dict[str, Dict[str, Any]] = {}
        post_comment_authors: Dict[str, set] = {}
        if prompts_metadata_path and prompts_metadata_path.exists():
            with open(prompts_metadata_path, "r", encoding="utf-8") as file:
                prompts_metadata = json.load(file)

            source_file_value = str(prompts_metadata.get("source_file", ""))
            source_files = [Path(chunk.strip()) for chunk in source_file_value.split(",") if chunk.strip()]

            for raw_file in source_files:
                raw_path = raw_file if raw_file.is_absolute() else Path.cwd() / raw_file
                if not raw_path.exists():
                    continue

                with open(raw_path, "r", encoding="utf-8") as file:
                    raw_payload = json.load(file)

                posts = raw_payload.get("posts", []) if isinstance(raw_payload, dict) else []
                for post in posts:
                    if not isinstance(post, dict):
                        continue
                    post_id = post.get("post_id")
                    if not isinstance(post_id, str) or not post_id:
                        continue
                    post_index[post_id] = {
                        "community": post.get("community") or raw_path.stem,
                        "upvotes": post.get("upvotes"),
                        "downvotes": post.get("downvotes"),
                        "topic": post.get("topic"),
                        "topic_label": post.get("topic_label"),
                    }

                comments = raw_payload.get("comments", []) if isinstance(raw_payload, dict) else []
                for comment in comments:
                    if not isinstance(comment, dict):
                        continue

                    comment_id = comment.get("comment_id")
                    if not isinstance(comment_id, str) or not comment_id:
                        continue

                    comment_index[comment_id] = {
                        "parent_id": comment.get("parent_id"),
                        "post_id": comment.get("post_id"),
                        "community": comment.get("community") or raw_path.stem,
                        "author": comment.get("author"),
                        "upvotes": comment.get("upvotes"),
                        "depth": comment.get("depth"),
                        "created_at": comment.get("created_at"),
                    }

                    post_id = comment.get("post_id")
                    author = comment.get("author")
                    if isinstance(post_id, str) and post_id:
                        if post_id not in post_comment_authors:
                            post_comment_authors[post_id] = set()
                        if isinstance(author, str) and author:
                            post_comment_authors[post_id].add(author)

        return cls(
            entries=entries,
            comment_index=comment_index,
            post_index=post_index,
            post_comment_authors=post_comment_authors,
        )

    @staticmethod
    def _extract_stance_value(entry: Dict[str, Any]) -> Optional[float]:
        topic = entry.get("topic") if isinstance(entry, dict) else None
        stance = topic.get("stance") if isinstance(topic, dict) else None
        raw_value = stance.get("value") if isinstance(stance, dict) else None

        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        if isinstance(raw_value, str):
            candidate = raw_value.strip()
            if not candidate or candidate.upper() == "N/A":
                return None
            try:
                return float(candidate)
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_numeric_field(entry: Dict[str, Any], field_name: str) -> Optional[float]:
        if not isinstance(entry, dict):
            return None
        raw_value = entry.get(field_name)
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        if isinstance(raw_value, str):
            candidate = raw_value.strip()
            if not candidate or candidate.upper() == "N/A":
                return None
            try:
                return float(candidate)
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_epistemic_numeric_field(entry: Dict[str, Any], field_name: str) -> Optional[float]:
        if not isinstance(entry, dict):
            return None

        epistemic_risk = entry.get("epistemic_risk")
        if not isinstance(epistemic_risk, dict):
            return None

        raw_value = epistemic_risk.get(field_name)
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        if isinstance(raw_value, str):
            candidate = raw_value.strip()
            if not candidate or candidate.upper() == "N/A":
                return None
            try:
                return float(candidate)
            except ValueError:
                return None
        return None

    @staticmethod
    def _clean_topic_label(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        candidate = value.strip()
        if not candidate or candidate.upper() == "N/A":
            return None
        return candidate

    @staticmethod
    def _extract_topic_label(entry: Dict[str, Any]) -> Optional[str]:
        if not isinstance(entry, dict):
            return None
        topic = entry.get("topic") if isinstance(entry.get("topic"), dict) else None
        label = topic.get("label") if isinstance(topic, dict) else None
        cleaned_label = CC_Metrics._clean_topic_label(label)
        if cleaned_label:
            return cleaned_label

        return CC_Metrics._clean_topic_label(entry.get("parent_topic"))

    @staticmethod
    def _stance_sign(value: Optional[float]) -> Optional[int]:
        if value is None:
            return None
        if value > 3:
            return 1
        if value < -3:
            return -1
        return None

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _collect_analyzed_comments(self, subreddit: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        analyzed: Dict[str, Dict[str, Any]] = {}
        for item in self.entries:
            if not isinstance(item, dict):
                continue
            comment_id = item.get("comment_id")
            entry = item.get("entry")
            if not isinstance(comment_id, str) or not comment_id or not isinstance(entry, dict):
                continue

            if subreddit:
                meta = self.comment_index.get(comment_id, {})
                community = meta.get("community") if isinstance(meta, dict) else None
                if community != subreddit:
                    continue

            analyzed[comment_id] = entry
        return analyzed

    def _resolve_topic_to_post_ids(
        self,
        subreddit: Optional[str] = None,
        analyzed_by_comment_id: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, set]:
        if analyzed_by_comment_id is None:
            analyzed_by_comment_id = self._collect_analyzed_comments(subreddit=subreddit)

        topic_to_post_ids: Dict[str, set] = {}

        inferred_topic_by_post: Dict[str, str] = {}
        if analyzed_by_comment_id:
            topic_votes_by_post: Dict[str, Dict[str, int]] = {}
            for comment_id, entry in analyzed_by_comment_id.items():
                topic_label = self._extract_topic_label(entry)
                if not topic_label:
                    continue

                meta = self.comment_index.get(comment_id, {})
                post_id = meta.get("post_id") if isinstance(meta, dict) else None
                if not isinstance(post_id, str) or not post_id:
                    continue

                post_votes = topic_votes_by_post.setdefault(post_id, {})
                post_votes[topic_label] = post_votes.get(topic_label, 0) + 1

            for post_id, topic_votes in topic_votes_by_post.items():
                if not topic_votes:
                    continue
                max_count = max(topic_votes.values())
                winners = sorted([label for label, count in topic_votes.items() if count == max_count])
                if winners:
                    inferred_topic_by_post[post_id] = winners[0]

        for post_id, post_meta in self.post_index.items():
            if not isinstance(post_meta, dict):
                continue
            if subreddit:
                community = post_meta.get("community")
                if community != subreddit:
                    continue

            topic_label = (
                self._clean_topic_label(post_meta.get("topic"))
                or self._clean_topic_label(post_meta.get("topic_label"))
                or inferred_topic_by_post.get(post_id)
            )

            if not topic_label:
                continue

            topic_to_post_ids.setdefault(topic_label, set()).add(post_id)

        return topic_to_post_ids

    def _resolve_relation_map_to_post(
        self,
        analyzed_by_comment_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Optional[int]]:
        parent_map: Dict[str, Optional[str]] = {}
        post_map: Dict[str, Optional[str]] = {}
        for comment_id in analyzed_by_comment_id:
            meta = self.comment_index.get(comment_id, {})
            parent_map[comment_id] = meta.get("parent_id") if isinstance(meta, dict) else None
            post_map[comment_id] = meta.get("post_id") if isinstance(meta, dict) else None

        relation_cache: Dict[str, Optional[int]] = {}

        def resolve(comment_id: str) -> Optional[int]:
            if comment_id in relation_cache:
                return relation_cache[comment_id]

            entry = analyzed_by_comment_id.get(comment_id)
            if not isinstance(entry, dict):
                relation_cache[comment_id] = None
                return None

            sign = self._stance_sign(self._extract_stance_value(entry))
            if sign is None:
                relation_cache[comment_id] = None
                return None

            parent_id = parent_map.get(comment_id)
            post_id = post_map.get(comment_id)

            if isinstance(parent_id, str) and isinstance(post_id, str) and parent_id == post_id:
                relation_cache[comment_id] = sign
                return sign

            if not isinstance(parent_id, str) or not parent_id or parent_id not in analyzed_by_comment_id:
                relation_cache[comment_id] = None
                return None

            parent_relation = resolve(parent_id)
            if parent_relation is None:
                relation_cache[comment_id] = None
                return None

            relation_cache[comment_id] = sign * parent_relation
            return relation_cache[comment_id]

        for cid in analyzed_by_comment_id:
            resolve(cid)

        return relation_cache

    def _continuation_depths(self, comment_ids: List[str]) -> Dict[str, Optional[int]]:
        id_set = set(comment_ids)
        children: Dict[str, List[str]] = {cid: [] for cid in comment_ids}
        depth_map: Dict[str, Optional[int]] = {}

        for cid in comment_ids:
            meta = self.comment_index.get(cid, {})
            depth_raw = meta.get("depth") if isinstance(meta, dict) else None
            try:
                depth_map[cid] = int(depth_raw) if depth_raw is not None else None
            except (TypeError, ValueError):
                depth_map[cid] = None

        for cid in comment_ids:
            meta = self.comment_index.get(cid, {})
            parent_id = meta.get("parent_id") if isinstance(meta, dict) else None
            if isinstance(parent_id, str) and parent_id in id_set:
                children[parent_id].append(cid)

        cache: Dict[str, Optional[int]] = {}

        def deepest_descendant_depth(cid: str) -> Optional[int]:
            if cid in cache:
                return cache[cid]

            own_depth = depth_map.get(cid)
            if own_depth is None:
                cache[cid] = None
                return None

            max_depth = own_depth
            for child in children.get(cid, []):
                child_depth = deepest_descendant_depth(child)
                if child_depth is not None and child_depth > max_depth:
                    max_depth = child_depth

            cache[cid] = max_depth
            return max_depth

        output: Dict[str, Optional[int]] = {}
        for cid in comment_ids:
            own_depth = depth_map.get(cid)
            deepest = deepest_descendant_depth(cid)
            if own_depth is None or deepest is None:
                output[cid] = None
            else:
                output[cid] = max(0, deepest - own_depth)

        return output

    def _filter_newest_comment_ids(
        self,
        comment_ids: List[str],
        percentile: float,
    ) -> List[str]:
        timestamps: Dict[str, datetime] = {}
        values: List[float] = []
        for cid in comment_ids:
            meta = self.comment_index.get(cid, {})
            ts = self._parse_timestamp(meta.get("created_at") if isinstance(meta, dict) else None)
            if ts is None:
                continue
            timestamps[cid] = ts
            values.append(ts.timestamp())

        if not values:
            return []

        threshold = float(np.quantile(values, percentile / 100.0))
        return [cid for cid, ts in timestamps.items() if ts.timestamp() >= threshold]

    def compute_all(self) -> Dict[str, Dict[str, Any]]:
        return self.compute_all_by_subreddit()

    def compute_all_by_subreddit(self) -> Dict[str, Dict[str, Any]]:
        output: Dict[str, Dict[str, Any]] = {}
        for subreddit in self.get_subreddits():
            output[subreddit] = {
                "Selective Engagement": self.selective_engagement(subreddit=subreddit),
                "Discreditation Rate": self.discreditation_rate(subreddit=subreddit),
                "Dropout Rate": self.dropout_rate(subreddit=subreddit),
                "Expressive Participation Gap (by_topic)": self.expressive_participation_gap_by_topic(subreddit=subreddit),
                "Linguistic Self-Protection Rate (LSPR) (by_topic)": self.lspr_by_topic(subreddit=subreddit),
                "Counter-Evidence Exposure Rate (CER) (by_topic)": self.cer_by_topic(subreddit=subreddit),
                "Constructive Counter-View Engagement (CCVE) (by_topic)": self.ccve_by_topic(subreddit=subreddit),
                "Counter-Evidence Sentiment Shift (CESS) (by_topic)": self.cess_by_topic(subreddit=subreddit),
                "Engagement Asymmetry Index (EAI) (by_topic)": self.eai_by_topic(subreddit=subreddit),
                "Cross-Stance Interaction Rate (CSIR) (by_topic)": self.csir_by_topic(subreddit=subreddit),
                "Visible Opinion Compression (VOC) (by_topic)": self.voc_by_topic(subreddit=subreddit),
                "Visibility Suppression Ratio (VSR)": self.vsr(),
                "Low-Support Claim Amplification (LSCA) (by_topic)": self.lsca_by_topic(subreddit=subreddit),
                "Emotional Amplification Score (EAS) (by_topic)": self.eas_by_topic(subreddit=subreddit),
            }
        return output

    def compute_all_by_subreddit_topic(self) -> Dict[str, Dict[str, Dict[str, MetricResult]]]:
        output: Dict[str, Dict[str, Dict[str, MetricResult]]] = {}
        for subreddit in self.get_subreddits():
            output[subreddit] = {
                "Expressive Participation Gap": self.expressive_participation_gap_by_topic(subreddit=subreddit),
                "Linguistic Self-Protection Rate (LSPR)": self.lspr_by_topic(subreddit=subreddit),
                "Counter-Evidence Exposure Rate (CER)": self.cer_by_topic(subreddit=subreddit),
                "Constructive Counter-View Engagement (CCVE)": self.ccve_by_topic(subreddit=subreddit),
                "Counter-Evidence Sentiment Shift (CESS)": self.cess_by_topic(subreddit=subreddit),
                "Engagement Asymmetry Index (EAI)": self.eai_by_topic(subreddit=subreddit),
                "Cross-Stance Interaction Rate (CSIR)": self.csir_by_topic(subreddit=subreddit),
                "Visible Opinion Compression (VOC)": self.voc_by_topic(subreddit=subreddit),
                "Low-Support Claim Amplification (LSCA)": self.lsca_by_topic(subreddit=subreddit),
                "Emotional Amplification Score (EAS)": self.eas_by_topic(subreddit=subreddit),
            }
        return output

    def get_subreddits(self) -> List[str]:
        values = set()
        for item in self.entries:
            if not isinstance(item, dict):
                continue
            comment_id = item.get("comment_id")
            if not isinstance(comment_id, str):
                continue
            meta = self.comment_index.get(comment_id, {})
            community = meta.get("community") if isinstance(meta, dict) else None
            if isinstance(community, str) and community:
                values.add(community)
        return sorted(values)

    def selective_engagement(self, subreddit: Optional[str] = None) -> MetricResult:
        if not self.entries:
            return MetricResult("Selective Engagement", None, 0, "No entries")

        analyzed_by_comment_id: Dict[str, Dict[str, Any]] = {}
        for item in self.entries:
            if not isinstance(item, dict):
                continue
            comment_id = item.get("comment_id")
            entry = item.get("entry")
            if isinstance(comment_id, str) and comment_id and isinstance(entry, dict):
                if subreddit:
                    meta = self.comment_index.get(comment_id, {})
                    community = meta.get("community") if isinstance(meta, dict) else None
                    if community != subreddit:
                        continue
                analyzed_by_comment_id[comment_id] = entry

        if not analyzed_by_comment_id:
            detail = f" for subreddit={subreddit}" if subreddit else ""
            return MetricResult("Selective Engagement", None, 0, f"No analyzable comment_id entries{detail}")

        local_parent_map: Dict[str, Optional[str]] = {}
        local_post_map: Dict[str, Optional[str]] = {}
        for comment_id in analyzed_by_comment_id:
            meta = self.comment_index.get(comment_id, {})
            local_parent_map[comment_id] = meta.get("parent_id") if isinstance(meta, dict) else None
            local_post_map[comment_id] = meta.get("post_id") if isinstance(meta, dict) else None

        relation_cache: Dict[str, Optional[int]] = {}

        def resolve_relation_to_post(comment_id: str) -> Optional[int]:
            if comment_id in relation_cache:
                return relation_cache[comment_id]

            entry = analyzed_by_comment_id.get(comment_id)
            if not isinstance(entry, dict):
                relation_cache[comment_id] = None
                return None

            sign = self._stance_sign(self._extract_stance_value(entry))
            if sign is None:
                relation_cache[comment_id] = None
                return None

            parent_id = local_parent_map.get(comment_id)
            post_id = local_post_map.get(comment_id)

            if isinstance(parent_id, str) and isinstance(post_id, str) and parent_id == post_id:
                relation_cache[comment_id] = sign
                return sign

            if not isinstance(parent_id, str) or not parent_id:
                relation_cache[comment_id] = None
                return None

            if parent_id not in analyzed_by_comment_id:
                relation_cache[comment_id] = None
                return None

            parent_relation = resolve_relation_to_post(parent_id)
            if parent_relation is None:
                relation_cache[comment_id] = None
                return None

            relation_cache[comment_id] = sign * parent_relation
            return relation_cache[comment_id]

        aligned_count = 0
        counter_count = 0
        excluded_count = 0

        for comment_id in analyzed_by_comment_id:
            relation = resolve_relation_to_post(comment_id)
            if relation == 1:
                aligned_count += 1
            elif relation == -1:
                counter_count += 1
            else:
                excluded_count += 1

        denominator = aligned_count + counter_count
        value = (aligned_count / denominator) if denominator else None
        notes = (
            f"aligned={aligned_count}, counter={counter_count}, excluded={excluded_count}; "
            "thresholds: >3 aligned, <-3 counter; neutral/missing excluded with descendants unresolved"
        )
        if subreddit:
            notes = f"subreddit={subreddit}; {notes}"
        return MetricResult("Selective Engagement", value, denominator, notes)

    def selective_engagement_by_subreddit(self) -> Dict[str, MetricResult]:
        results: Dict[str, MetricResult] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.selective_engagement(subreddit=subreddit)
        return results

    def discreditation_rate(self, subreddit: Optional[str] = None) -> MetricResult:
        qualifying_count = 0
        denominator_count = 0
        excluded_count = 0

        for item in self.entries:
            if not isinstance(item, dict):
                continue

            comment_id = item.get("comment_id")
            if not isinstance(comment_id, str) or not comment_id:
                continue

            if subreddit:
                meta = self.comment_index.get(comment_id, {})
                community = meta.get("community") if isinstance(meta, dict) else None
                if community != subreddit:
                    continue

            entry = item.get("entry")
            if not isinstance(entry, dict):
                excluded_count += 1
                continue

            discrediting_value = self._extract_numeric_field(entry, "discrediting")
            if discrediting_value is None:
                excluded_count += 1
                continue

            denominator_count += 1
            if discrediting_value > 3:
                qualifying_count += 1

        value = (qualifying_count / denominator_count) if denominator_count else None
        notes = (
            f"qualifying(discrediting>3)={qualifying_count}, denominator={denominator_count}, "
            f"excluded={excluded_count}"
        )
        if subreddit:
            notes = f"subreddit={subreddit}; {notes}"

        return MetricResult("Discreditation Rate", value, denominator_count, notes)

    def discreditation_rate_by_subreddit(self) -> Dict[str, MetricResult]:
        results: Dict[str, MetricResult] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.discreditation_rate(subreddit=subreddit)
        return results

    def dropout_rate(self, subreddit: Optional[str] = None) -> MetricResult:
        epsilon = 0.01
        min_group_n = 10
        analyzed_by_comment_id = self._collect_analyzed_comments(subreddit=subreddit)
        if not analyzed_by_comment_id:
            detail = f" for subreddit={subreddit}" if subreddit else ""
            return MetricResult("Dropout Rate", None, 0, f"No analyzable comments{detail}")

        relation_map = self._resolve_relation_map_to_post(analyzed_by_comment_id)

        def resolve_thread_root(comment_id: str) -> Optional[str]:
            current = comment_id
            visited = set()
            while True:
                if current in visited:
                    return None
                visited.add(current)

                meta = self.comment_index.get(current, {})
                parent_id = meta.get("parent_id") if isinstance(meta, dict) else None
                post_id = meta.get("post_id") if isinstance(meta, dict) else None

                if not isinstance(parent_id, str) or not parent_id:
                    return None
                if isinstance(post_id, str) and parent_id == post_id:
                    return current
                current = parent_id

        def is_descendant(descendant_id: str, ancestor_id: str) -> bool:
            current = descendant_id
            visited = set()
            while True:
                if current in visited:
                    return False
                visited.add(current)

                meta = self.comment_index.get(current, {})
                parent_id = meta.get("parent_id") if isinstance(meta, dict) else None
                post_id = meta.get("post_id") if isinstance(meta, dict) else None

                if not isinstance(parent_id, str) or not parent_id:
                    return False
                if isinstance(post_id, str) and parent_id == post_id:
                    return False
                if parent_id == ancestor_id:
                    return True

                current = parent_id

        threads: Dict[str, List[str]] = {}
        for cid in analyzed_by_comment_id:
            root = resolve_thread_root(cid)
            if not isinstance(root, str) or not root:
                continue
            threads.setdefault(root, []).append(cid)

        counter_depths: List[float] = []
        aligned_depths: List[float] = []
        excluded_missing_time = 0
        excluded_missing_relation = 0

        for _, thread_ids in threads.items():
            thread_with_ts: List[str] = []
            ts_values: List[float] = []
            ts_map: Dict[str, float] = {}
            for cid in thread_ids:
                meta = self.comment_index.get(cid, {})
                ts = self._parse_timestamp(meta.get("created_at") if isinstance(meta, dict) else None)
                if ts is None:
                    excluded_missing_time += 1
                    continue
                ts_epoch = ts.timestamp()
                thread_with_ts.append(cid)
                ts_values.append(ts_epoch)
                ts_map[cid] = ts_epoch

            if not thread_with_ts:
                continue

            cutoff = float(np.quantile(ts_values, 0.75))
            filtered_ids = [cid for cid in thread_with_ts if ts_map[cid] < cutoff]
            if not filtered_ids:
                continue

            replies_below: Dict[str, int] = {}
            for cid in filtered_ids:
                replies_below[cid] = sum(
                    1
                    for other in filtered_ids
                    if other != cid and is_descendant(other, cid)
                )

            max_replies = max(replies_below.values()) if replies_below else 0
            denominator = max_replies if max_replies > 0 else 1

            for cid in filtered_ids:
                relation = relation_map.get(cid)
                if relation not in (1, -1):
                    excluded_missing_relation += 1
                    continue

                relative_depth = replies_below[cid] / denominator
                if relation == -1:
                    counter_depths.append(relative_depth)
                else:
                    aligned_depths.append(relative_depth)

        counter_n = len(counter_depths)
        aligned_n = len(aligned_depths)
        counter_zero_n = sum(1 for value in counter_depths if value == 0)
        aligned_zero_n = sum(1 for value in aligned_depths if value == 0)
        counter_zero_rate = (counter_zero_n / counter_n) if counter_n else None
        aligned_zero_rate = (aligned_zero_n / aligned_n) if aligned_n else None

        if not counter_depths or not aligned_depths:
            notes = (
                f"counter_n={counter_n}, aligned_n={aligned_n}, "
                f"counter_zero_n={counter_zero_n}, aligned_zero_n={aligned_zero_n}, "
                f"excluded_missing_time={excluded_missing_time}, excluded_missing_relation={excluded_missing_relation}, "
                "insufficient counter/aligned data after per-thread <75th percentile filter"
            )
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"
            return MetricResult("Dropout Rate", None, counter_n + aligned_n, notes)

        if counter_n < min_group_n or aligned_n < min_group_n:
            notes = (
                f"counter_n={counter_n}, aligned_n={aligned_n}, min_group_n={min_group_n}, "
                f"counter_zero_n={counter_zero_n}, aligned_zero_n={aligned_zero_n}, "
                f"counter_zero_rate={counter_zero_rate}, aligned_zero_rate={aligned_zero_rate}, "
                f"excluded_missing_time={excluded_missing_time}, excluded_missing_relation={excluded_missing_relation}; "
                "insufficient sample size for robust depth comparison"
            )
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"
            return MetricResult("Dropout Rate", None, counter_n + aligned_n, notes)

        mean_counter = float(np.mean(counter_depths))
        mean_aligned = float(np.mean(aligned_depths))
        value = 1.0 - ((mean_counter + epsilon) / (mean_aligned + epsilon))
        notes = (
            f"counter_n={counter_n}, aligned_n={aligned_n}, min_group_n={min_group_n}, "
            f"counter_zero_n={counter_zero_n}, aligned_zero_n={aligned_zero_n}, "
            f"counter_zero_rate={counter_zero_rate}, aligned_zero_rate={aligned_zero_rate}, "
            f"mean_counter={mean_counter}, mean_aligned={mean_aligned}, epsilon={epsilon}, "
            f"excluded_missing_time={excluded_missing_time}, excluded_missing_relation={excluded_missing_relation}, "
            "dropout formula=1-(mean_counter+epsilon)/(mean_aligned+epsilon), "
            "timestamp filter=per-thread comments with timestamp < 75th percentile"
        )
        if subreddit:
            notes = f"subreddit={subreddit}; {notes}"
        return MetricResult("Dropout Rate", value, counter_n + aligned_n, notes)

    def dropout_rate_debug(
        self,
        subreddit: Optional[str] = None,
        max_threads: Optional[int] = None,
    ) -> Dict[str, Any]:
        epsilon = 0.01
        min_group_n = 10
        analyzed_by_comment_id = self._collect_analyzed_comments(subreddit=subreddit)
        if not analyzed_by_comment_id:
            return {
                "subreddit": subreddit,
                "threads": [],
                "counter_depths": [],
                "aligned_depths": [],
                "mean_counter": None,
                "mean_aligned": None,
                "counter_n": 0,
                "aligned_n": 0,
                "counter_zero_n": 0,
                "aligned_zero_n": 0,
                "counter_zero_rate": None,
                "aligned_zero_rate": None,
                "dropout_rate": None,
                "excluded_missing_time": 0,
                "excluded_missing_relation": 0,
                "notes": "No analyzable comments",
            }

        relation_map = self._resolve_relation_map_to_post(analyzed_by_comment_id)

        def resolve_thread_root(comment_id: str) -> Optional[str]:
            current = comment_id
            visited = set()
            while True:
                if current in visited:
                    return None
                visited.add(current)

                meta = self.comment_index.get(current, {})
                parent_id = meta.get("parent_id") if isinstance(meta, dict) else None
                post_id = meta.get("post_id") if isinstance(meta, dict) else None

                if not isinstance(parent_id, str) or not parent_id:
                    return None
                if isinstance(post_id, str) and parent_id == post_id:
                    return current
                current = parent_id

        def is_descendant(descendant_id: str, ancestor_id: str) -> bool:
            current = descendant_id
            visited = set()
            while True:
                if current in visited:
                    return False
                visited.add(current)

                meta = self.comment_index.get(current, {})
                parent_id = meta.get("parent_id") if isinstance(meta, dict) else None
                post_id = meta.get("post_id") if isinstance(meta, dict) else None

                if not isinstance(parent_id, str) or not parent_id:
                    return False
                if isinstance(post_id, str) and parent_id == post_id:
                    return False
                if parent_id == ancestor_id:
                    return True

                current = parent_id

        threads: Dict[str, List[str]] = {}
        for cid in analyzed_by_comment_id:
            root = resolve_thread_root(cid)
            if not isinstance(root, str) or not root:
                continue
            threads.setdefault(root, []).append(cid)

        counter_depths: List[float] = []
        aligned_depths: List[float] = []
        excluded_missing_time = 0
        excluded_missing_relation = 0
        thread_debug_rows: List[Dict[str, Any]] = []

        thread_items = sorted(threads.items(), key=lambda item: item[0])
        if isinstance(max_threads, int) and max_threads >= 0:
            thread_items = thread_items[:max_threads]

        for thread_root, thread_ids in thread_items:
            thread_with_ts: List[str] = []
            ts_values: List[float] = []
            ts_map: Dict[str, float] = {}
            for cid in thread_ids:
                meta = self.comment_index.get(cid, {})
                ts = self._parse_timestamp(meta.get("created_at") if isinstance(meta, dict) else None)
                if ts is None:
                    excluded_missing_time += 1
                    continue
                ts_epoch = ts.timestamp()
                thread_with_ts.append(cid)
                ts_values.append(ts_epoch)
                ts_map[cid] = ts_epoch

            if not thread_with_ts:
                thread_debug_rows.append(
                    {
                        "thread_root": thread_root,
                        "thread_size": len(thread_ids),
                        "with_timestamp": 0,
                        "cutoff_p75": None,
                        "filtered_count": 0,
                        "max_replies_below": 0,
                        "comments": [],
                    }
                )
                continue

            cutoff = float(np.quantile(ts_values, 0.75))
            filtered_ids = [cid for cid in thread_with_ts if ts_map[cid] < cutoff]

            replies_below: Dict[str, int] = {}
            for cid in filtered_ids:
                replies_below[cid] = sum(
                    1
                    for other in filtered_ids
                    if other != cid and is_descendant(other, cid)
                )

            max_replies = max(replies_below.values()) if replies_below else 0
            denominator = max_replies if max_replies > 0 else 1

            per_comment_rows: List[Dict[str, Any]] = []
            for cid in filtered_ids:
                relation = relation_map.get(cid)
                relative_depth = replies_below[cid] / denominator
                if relation == -1:
                    counter_depths.append(relative_depth)
                elif relation == 1:
                    aligned_depths.append(relative_depth)
                else:
                    excluded_missing_relation += 1

                per_comment_rows.append(
                    {
                        "comment_id": cid,
                        "timestamp": ts_map[cid],
                        "relation": relation,
                        "replies_below": replies_below[cid],
                        "relative_depth": relative_depth,
                    }
                )

            thread_debug_rows.append(
                {
                    "thread_root": thread_root,
                    "thread_size": len(thread_ids),
                    "with_timestamp": len(thread_with_ts),
                    "cutoff_p75": cutoff,
                    "filtered_count": len(filtered_ids),
                    "max_replies_below": max_replies,
                    "comments": per_comment_rows,
                }
            )

        counter_n = len(counter_depths)
        aligned_n = len(aligned_depths)
        counter_zero_n = sum(1 for value in counter_depths if value == 0)
        aligned_zero_n = sum(1 for value in aligned_depths if value == 0)
        counter_zero_rate = (counter_zero_n / counter_n) if counter_n else None
        aligned_zero_rate = (aligned_zero_n / aligned_n) if aligned_n else None
        mean_counter = float(np.mean(counter_depths)) if counter_depths else None
        mean_aligned = float(np.mean(aligned_depths)) if aligned_depths else None
        dropout_value = None
        if (
            mean_counter is not None
            and mean_aligned is not None
            and counter_n >= min_group_n
            and aligned_n >= min_group_n
        ):
            dropout_value = 1.0 - ((mean_counter + epsilon) / (mean_aligned + epsilon))

        return {
            "subreddit": subreddit,
            "threads": thread_debug_rows,
            "counter_depths": counter_depths,
            "aligned_depths": aligned_depths,
            "mean_counter": mean_counter,
            "mean_aligned": mean_aligned,
            "counter_n": counter_n,
            "aligned_n": aligned_n,
            "counter_zero_n": counter_zero_n,
            "aligned_zero_n": aligned_zero_n,
            "counter_zero_rate": counter_zero_rate,
            "aligned_zero_rate": aligned_zero_rate,
            "epsilon": epsilon,
            "min_group_n": min_group_n,
            "dropout_rate": dropout_value,
            "excluded_missing_time": excluded_missing_time,
            "excluded_missing_relation": excluded_missing_relation,
        }

    def dropout_rate_by_subreddit(self) -> Dict[str, MetricResult]:
        results: Dict[str, MetricResult] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.dropout_rate(subreddit=subreddit)
        return results

    def expressive_participation_gap_by_topic(self, subreddit: Optional[str] = None) -> Dict[str, MetricResult]:
        analyzed_by_comment_id = self._collect_analyzed_comments(subreddit=subreddit)
        topic_to_post_ids = self._resolve_topic_to_post_ids(
            subreddit=subreddit,
            analyzed_by_comment_id=analyzed_by_comment_id,
        )

        results: Dict[str, MetricResult] = {}
        for topic_label, post_ids in sorted(topic_to_post_ids.items(), key=lambda item: item[0].lower()):
            epg_values: List[float] = []
            excluded_votes_zero = 0
            excluded_missing_post = 0

            for post_id in post_ids:
                post_meta = self.post_index.get(post_id)
                if not isinstance(post_meta, dict):
                    excluded_missing_post += 1
                    continue

                upvotes_raw = post_meta.get("upvotes")
                downvotes_raw = post_meta.get("downvotes")
                try:
                    upvotes = int(upvotes_raw) if upvotes_raw is not None else 0
                except (TypeError, ValueError):
                    upvotes = 0
                try:
                    downvotes = int(downvotes_raw) if downvotes_raw is not None else 0
                except (TypeError, ValueError):
                    downvotes = 0

                votes = upvotes + downvotes
                if votes <= 0:
                    excluded_votes_zero += 1
                    continue

                commenters = len(self.post_comment_authors.get(post_id, set()))
                epg_p = 1.0 - (commenters / float(votes))
                epg_values.append(epg_p)

            value = float(np.mean(epg_values)) if epg_values else None
            notes = (
                f"topic={topic_label}; valid_posts={len(epg_values)}, total_posts={len(post_ids)}, "
                f"excluded_votes_zero={excluded_votes_zero}, excluded_missing_post={excluded_missing_post}; "
                "EPG_p=1-(distinct_commenters/votes), EPG_s,t=mean(EPG_p)"
            )
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"

            results[topic_label] = MetricResult(
                "Expressive Participation Gap",
                value,
                len(epg_values),
                notes,
            )

        return results

    def expressive_participation_gap(self, subreddit: Optional[str] = None) -> MetricResult:
        by_topic = self.expressive_participation_gap_by_topic(subreddit=subreddit)
        if not by_topic:
            detail = f" for subreddit={subreddit}" if subreddit else ""
            return MetricResult("Expressive Participation Gap", None, 0, f"No topic/post coverage{detail}")

        values = [item.value for item in by_topic.values() if item.value is not None]
        total_valid_posts = sum(item.sample_size for item in by_topic.values())
        if not values:
            notes = "No valid posts with votes>0 for computed topics"
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"
            return MetricResult("Expressive Participation Gap", None, total_valid_posts, notes)

        topic_summary = "; ".join(
            f"{topic}={result.value:.4f}" if result.value is not None else f"{topic}=None"
            for topic, result in sorted(by_topic.items(), key=lambda item: item[0].lower())
        )
        notes = (
            f"topics={len(by_topic)}, valid_posts_total={total_valid_posts}; "
            f"topic_values: {topic_summary}"
        )
        if subreddit:
            notes = f"subreddit={subreddit}; {notes}"

        return MetricResult(
            "Expressive Participation Gap",
            float(np.mean(values)),
            total_valid_posts,
            notes,
        )

    def expressive_participation_gap_by_subreddit(self) -> Dict[str, MetricResult]:
        results: Dict[str, MetricResult] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.expressive_participation_gap(subreddit=subreddit)
        return results

    def expressive_participation_gap_by_subreddit_topic(self) -> Dict[str, Dict[str, MetricResult]]:
        results: Dict[str, Dict[str, MetricResult]] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.expressive_participation_gap_by_topic(subreddit=subreddit)
        return results

    def lspr_by_topic(self, subreddit: Optional[str] = None) -> Dict[str, MetricResult]:
        analyzed_by_comment_id = self._collect_analyzed_comments(subreddit=subreddit)
        topic_to_post_ids = self._resolve_topic_to_post_ids(
            subreddit=subreddit,
            analyzed_by_comment_id=analyzed_by_comment_id,
        )

        post_to_topic: Dict[str, str] = {}
        for topic_label, post_ids in topic_to_post_ids.items():
            for post_id in post_ids:
                post_to_topic[post_id] = topic_label

        defensive_by_topic: Dict[str, List[float]] = {}
        excluded_non_numeric = 0
        for comment_id, entry in analyzed_by_comment_id.items():
            if not isinstance(entry, dict):
                continue

            comment_meta = self.comment_index.get(comment_id, {})
            post_id = comment_meta.get("post_id") if isinstance(comment_meta, dict) else None
            if not isinstance(post_id, str) or not post_id:
                continue

            topic_label = post_to_topic.get(post_id)
            if not topic_label:
                continue

            defensive = self._extract_numeric_field(entry, "defensive")
            if defensive is None:
                excluded_non_numeric += 1
                continue

            defensive_by_topic.setdefault(topic_label, []).append(defensive)

        results: Dict[str, MetricResult] = {}
        for topic_label, values in sorted(defensive_by_topic.items(), key=lambda item: item[0].lower()):
            mean_defensive = float(np.mean(values)) if values else None
            normalized = (mean_defensive / 10.0) if mean_defensive is not None else None
            notes = (
                f"topic={topic_label}; valid_comments={len(values)}, excluded_non_numeric={excluded_non_numeric}; "
                "LSPR_s,t=mean(defensive)/10; compare topics within the same subreddit"
            )
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"

            results[topic_label] = MetricResult(
                "Linguistic Self-Protection Rate (LSPR)",
                normalized,
                len(values),
                notes,
            )

        return results

    def lspr(self, subreddit: Optional[str] = None, topic: Optional[str] = None) -> MetricResult:
        by_topic = self.lspr_by_topic(subreddit=subreddit)
        if topic:
            topic_result = by_topic.get(topic)
            if topic_result is None:
                notes = f"No LSPR coverage for topic={topic}"
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                return MetricResult("Linguistic Self-Protection Rate (LSPR)", None, 0, notes)
            return topic_result

        if not by_topic:
            detail = f" for subreddit={subreddit}" if subreddit else ""
            return MetricResult("Linguistic Self-Protection Rate (LSPR)", None, 0, f"No topic/comment coverage{detail}")

        values = [item.value for item in by_topic.values() if item.value is not None]
        total_valid_comments = sum(item.sample_size for item in by_topic.values())
        if not values:
            notes = "No valid defensive scores for computed topics"
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"
            return MetricResult("Linguistic Self-Protection Rate (LSPR)", None, total_valid_comments, notes)

        topic_summary = "; ".join(
            f"{topic_name}={result.value:.4f}" if result.value is not None else f"{topic_name}=None"
            for topic_name, result in sorted(by_topic.items(), key=lambda item: item[0].lower())
        )
        notes = (
            f"topics={len(by_topic)}, valid_comments_total={total_valid_comments}; "
            f"topic_values: {topic_summary}; compare topics within the same subreddit"
        )
        if subreddit:
            notes = f"subreddit={subreddit}; {notes}"

        return MetricResult(
            "Linguistic Self-Protection Rate (LSPR)",
            float(np.mean(values)),
            total_valid_comments,
            notes,
        )

    def lspr_by_subreddit(self) -> Dict[str, MetricResult]:
        results: Dict[str, MetricResult] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.lspr(subreddit=subreddit)
        return results

    def lspr_by_subreddit_topic(self) -> Dict[str, Dict[str, MetricResult]]:
        results: Dict[str, Dict[str, MetricResult]] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.lspr_by_topic(subreddit=subreddit)
        return results

    def cer_by_topic(self, subreddit: Optional[str] = None) -> Dict[str, MetricResult]:
        analyzed_by_comment_id = self._collect_analyzed_comments(subreddit=subreddit)
        topic_to_post_ids = self._resolve_topic_to_post_ids(
            subreddit=subreddit,
            analyzed_by_comment_id=analyzed_by_comment_id,
        )

        post_to_topic: Dict[str, str] = {}
        for topic_label, post_ids in topic_to_post_ids.items():
            for post_id in post_ids:
                post_to_topic[post_id] = topic_label

        topic_valid_counts: Dict[str, int] = {topic: 0 for topic in topic_to_post_ids}
        topic_counter_evidence_counts: Dict[str, int] = {topic: 0 for topic in topic_to_post_ids}
        topic_excluded_invalid: Dict[str, int] = {topic: 0 for topic in topic_to_post_ids}

        for comment_id, entry in analyzed_by_comment_id.items():
            if not isinstance(entry, dict):
                continue

            comment_meta = self.comment_index.get(comment_id, {})
            post_id = comment_meta.get("post_id") if isinstance(comment_meta, dict) else None
            if not isinstance(post_id, str) or not post_id:
                continue

            topic_label = post_to_topic.get(post_id)
            if not topic_label:
                continue

            stance = self._extract_stance_value(entry)
            evidence_quality = self._extract_epistemic_numeric_field(entry, "evidence_quality")
            reasoning_depth = self._extract_epistemic_numeric_field(entry, "reasoning_depth")

            if stance is None or evidence_quality is None or reasoning_depth is None:
                topic_excluded_invalid[topic_label] = topic_excluded_invalid.get(topic_label, 0) + 1
                continue

            topic_valid_counts[topic_label] = topic_valid_counts.get(topic_label, 0) + 1
            if stance < 0 and (evidence_quality >= 4 or reasoning_depth >= 4):
                topic_counter_evidence_counts[topic_label] = topic_counter_evidence_counts.get(topic_label, 0) + 1

        results: Dict[str, MetricResult] = {}
        for topic_label in sorted(topic_to_post_ids.keys(), key=lambda item: item.lower()):
            valid_count = topic_valid_counts.get(topic_label, 0)
            counter_count = topic_counter_evidence_counts.get(topic_label, 0)
            excluded_count = topic_excluded_invalid.get(topic_label, 0)

            value = (counter_count / valid_count) if valid_count else None
            notes = (
                f"topic={topic_label}; valid_comments={valid_count}, "
                f"counter_evidence_comments={counter_count}, excluded_invalid={excluded_count}; "
                "counter condition: stance<0 and (evidence_quality>=4 or reasoning_depth>=4); "
                "compare topics within the same subreddit"
            )
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"

            results[topic_label] = MetricResult(
                "Counter-Evidence Exposure Rate (CER)",
                value,
                valid_count,
                notes,
            )

        return results

    def cer(self, subreddit: Optional[str] = None, topic: Optional[str] = None) -> MetricResult:
        by_topic = self.cer_by_topic(subreddit=subreddit)
        if topic:
            cleaned_topic = self._clean_topic_label(topic) or topic
            topic_result = by_topic.get(cleaned_topic)
            if topic_result is None:
                notes = f"No CER coverage for topic={topic}"
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                return MetricResult("Counter-Evidence Exposure Rate (CER)", None, 0, notes)
            return topic_result

        if not by_topic:
            detail = f" for subreddit={subreddit}" if subreddit else ""
            return MetricResult("Counter-Evidence Exposure Rate (CER)", None, 0, f"No topic/comment coverage{detail}")

        values = [item.value for item in by_topic.values() if item.value is not None]
        total_valid_comments = sum(item.sample_size for item in by_topic.values())
        if not values:
            notes = "No valid CER comments for computed topics"
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"
            return MetricResult("Counter-Evidence Exposure Rate (CER)", None, total_valid_comments, notes)

        topic_summary = "; ".join(
            f"{topic_name}={result.value:.4f}" if result.value is not None else f"{topic_name}=None"
            for topic_name, result in sorted(by_topic.items(), key=lambda item: item[0].lower())
        )
        notes = (
            f"topics={len(by_topic)}, valid_comments_total={total_valid_comments}; "
            f"topic_values: {topic_summary}; compare topics within the same subreddit"
        )
        if subreddit:
            notes = f"subreddit={subreddit}; {notes}"

        return MetricResult(
            "Counter-Evidence Exposure Rate (CER)",
            float(np.mean(values)),
            total_valid_comments,
            notes,
        )

    def cer_by_subreddit(self) -> Dict[str, MetricResult]:
        results: Dict[str, MetricResult] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.cer(subreddit=subreddit)
        return results

    def cer_by_subreddit_topic(self) -> Dict[str, Dict[str, MetricResult]]:
        results: Dict[str, Dict[str, MetricResult]] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.cer_by_topic(subreddit=subreddit)
        return results

    def ccve_by_topic(self, subreddit: Optional[str] = None) -> Dict[str, MetricResult]:
        analyzed_by_comment_id = self._collect_analyzed_comments(subreddit=subreddit)
        topic_to_post_ids = self._resolve_topic_to_post_ids(
            subreddit=subreddit,
            analyzed_by_comment_id=analyzed_by_comment_id,
        )

        post_to_topic: Dict[str, str] = {}
        for topic_label, post_ids in topic_to_post_ids.items():
            for post_id in post_ids:
                post_to_topic[post_id] = topic_label

        counter_evidence_ids_by_topic: Dict[str, set] = {topic: set() for topic in topic_to_post_ids}
        for comment_id, entry in analyzed_by_comment_id.items():
            if not isinstance(entry, dict):
                continue

            comment_meta = self.comment_index.get(comment_id, {})
            post_id = comment_meta.get("post_id") if isinstance(comment_meta, dict) else None
            if not isinstance(post_id, str) or not post_id:
                continue

            topic_label = post_to_topic.get(post_id)
            if not topic_label:
                continue

            stance = self._extract_stance_value(entry)
            evidence_quality = self._extract_epistemic_numeric_field(entry, "evidence_quality")
            reasoning_depth = self._extract_epistemic_numeric_field(entry, "reasoning_depth")
            if stance is None or evidence_quality is None or reasoning_depth is None:
                continue

            if stance < 0 and (evidence_quality >= 4 or reasoning_depth >= 4):
                counter_evidence_ids_by_topic.setdefault(topic_label, set()).add(comment_id)

        replies_to_counter_by_topic: Dict[str, int] = {topic: 0 for topic in topic_to_post_ids}
        constructive_replies_by_topic: Dict[str, int] = {topic: 0 for topic in topic_to_post_ids}
        excluded_invalid_fields_by_topic: Dict[str, int] = {topic: 0 for topic in topic_to_post_ids}

        for comment_id, entry in analyzed_by_comment_id.items():
            if not isinstance(entry, dict):
                continue

            comment_meta = self.comment_index.get(comment_id, {})
            post_id = comment_meta.get("post_id") if isinstance(comment_meta, dict) else None
            parent_id = comment_meta.get("parent_id") if isinstance(comment_meta, dict) else None
            if not isinstance(post_id, str) or not post_id or not isinstance(parent_id, str) or not parent_id:
                continue

            topic_label = post_to_topic.get(post_id)
            if not topic_label:
                continue

            counter_ids = counter_evidence_ids_by_topic.get(topic_label, set())
            if parent_id not in counter_ids:
                continue

            reasoning_depth = self._extract_epistemic_numeric_field(entry, "reasoning_depth")
            toxicity = self._extract_numeric_field(entry, "toxicity")
            discrediting = self._extract_numeric_field(entry, "discrediting")

            if reasoning_depth is None or toxicity is None or discrediting is None:
                excluded_invalid_fields_by_topic[topic_label] = excluded_invalid_fields_by_topic.get(topic_label, 0) + 1
                continue

            replies_to_counter_by_topic[topic_label] = replies_to_counter_by_topic.get(topic_label, 0) + 1
            if reasoning_depth >= 4 and toxicity <= 3 and discrediting <= 3:
                constructive_replies_by_topic[topic_label] = constructive_replies_by_topic.get(topic_label, 0) + 1

        results: Dict[str, MetricResult] = {}
        for topic_label in sorted(topic_to_post_ids.keys(), key=lambda item: item.lower()):
            replies_count = replies_to_counter_by_topic.get(topic_label, 0)
            constructive_count = constructive_replies_by_topic.get(topic_label, 0)
            excluded_count = excluded_invalid_fields_by_topic.get(topic_label, 0)
            counter_count = len(counter_evidence_ids_by_topic.get(topic_label, set()))

            value = (constructive_count / replies_count) if replies_count else None
            notes = (
                f"topic={topic_label}; counter_evidence_comments={counter_count}, "
                f"replies_to_counter_evidence={replies_count}, constructive_replies={constructive_count}, "
                f"excluded_invalid={excluded_count}; constructive condition: reasoning_depth>=4 and toxicity<=3 and discrediting<=3; "
                "compare topics within the same subreddit"
            )
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"

            results[topic_label] = MetricResult(
                "Constructive Counter-View Engagement (CCVE)",
                value,
                replies_count,
                notes,
            )

        return results

    def ccve(self, subreddit: Optional[str] = None, topic: Optional[str] = None) -> MetricResult:
        by_topic = self.ccve_by_topic(subreddit=subreddit)
        if topic:
            cleaned_topic = self._clean_topic_label(topic) or topic
            topic_result = by_topic.get(cleaned_topic)
            if topic_result is None:
                notes = f"No CCVE coverage for topic={topic}"
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                return MetricResult("Constructive Counter-View Engagement (CCVE)", None, 0, notes)
            return topic_result

        if not by_topic:
            detail = f" for subreddit={subreddit}" if subreddit else ""
            return MetricResult("Constructive Counter-View Engagement (CCVE)", None, 0, f"No topic/comment coverage{detail}")

        values = [item.value for item in by_topic.values() if item.value is not None]
        total_replies = sum(item.sample_size for item in by_topic.values())
        if not values:
            notes = "No replies to counter-evidence comments for computed topics"
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"
            return MetricResult("Constructive Counter-View Engagement (CCVE)", None, total_replies, notes)

        topic_summary = "; ".join(
            f"{topic_name}={result.value:.4f}" if result.value is not None else f"{topic_name}=None"
            for topic_name, result in sorted(by_topic.items(), key=lambda item: item[0].lower())
        )
        notes = (
            f"topics={len(by_topic)}, replies_total={total_replies}; "
            f"topic_values: {topic_summary}; compare topics within the same subreddit"
        )
        if subreddit:
            notes = f"subreddit={subreddit}; {notes}"

        return MetricResult(
            "Constructive Counter-View Engagement (CCVE)",
            float(np.mean(values)),
            total_replies,
            notes,
        )

    def ccve_by_subreddit(self) -> Dict[str, MetricResult]:
        results: Dict[str, MetricResult] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.ccve(subreddit=subreddit)
        return results

    def ccve_by_subreddit_topic(self) -> Dict[str, Dict[str, MetricResult]]:
        results: Dict[str, Dict[str, MetricResult]] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.ccve_by_topic(subreddit=subreddit)
        return results

    def _compute_cess_topic_stats(
        self,
        subreddit: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        analyzed_by_comment_id = self._collect_analyzed_comments(subreddit=subreddit)
        topic_to_post_ids = self._resolve_topic_to_post_ids(
            subreddit=subreddit,
            analyzed_by_comment_id=analyzed_by_comment_id,
        )

        post_to_topic: Dict[str, str] = {}
        for topic_label, post_ids in topic_to_post_ids.items():
            for post_id in post_ids:
                post_to_topic[post_id] = topic_label

        def stance_sign(value: Optional[float]) -> int:
            if value is None:
                return 0
            if value > 0:
                return 1
            if value < 0:
                return -1
            return 0

        def resolve_thread_root(comment_id: str) -> Optional[str]:
            current = comment_id
            visited = set()
            while True:
                if current in visited:
                    return None
                visited.add(current)

                meta = self.comment_index.get(current, {})
                parent_id = meta.get("parent_id") if isinstance(meta, dict) else None
                post_id = meta.get("post_id") if isinstance(meta, dict) else None

                if not isinstance(parent_id, str) or not parent_id:
                    return None
                if isinstance(post_id, str) and parent_id == post_id:
                    return current
                current = parent_id

        topic_threads: Dict[str, Dict[str, List[str]]] = {}
        for comment_id in analyzed_by_comment_id:
            meta = self.comment_index.get(comment_id, {})
            post_id = meta.get("post_id") if isinstance(meta, dict) else None
            if not isinstance(post_id, str) or not post_id:
                continue

            topic_label = post_to_topic.get(post_id)
            if not topic_label:
                continue

            thread_root = resolve_thread_root(comment_id)
            if not isinstance(thread_root, str) or not thread_root:
                continue

            topic_threads.setdefault(topic_label, {}).setdefault(thread_root, []).append(comment_id)

        topic_stats: Dict[str, Dict[str, Any]] = {}
        for topic_label in sorted(topic_to_post_ids.keys(), key=lambda item: item.lower()):
            threads = topic_threads.get(topic_label, {})
            thread_count = len(threads)

            deltas: List[float] = []
            excluded_missing_timestamp = 0
            excluded_missing_stance = 0
            excluded_single_comment = 0
            excluded_not_exposed = 0

            for _, comment_ids in threads.items():
                thread_rows: List[Dict[str, Any]] = []
                for cid in comment_ids:
                    entry = analyzed_by_comment_id.get(cid)
                    if not isinstance(entry, dict):
                        continue

                    meta = self.comment_index.get(cid, {})
                    author = meta.get("author") if isinstance(meta, dict) else None
                    ts = self._parse_timestamp(meta.get("created_at") if isinstance(meta, dict) else None)
                    stance = self._extract_stance_value(entry)

                    if not isinstance(author, str) or not author.strip():
                        continue
                    if ts is None:
                        excluded_missing_timestamp += 1
                        continue

                    thread_rows.append(
                        {
                            "comment_id": cid,
                            "author": author,
                            "timestamp": ts,
                            "stance": stance,
                        }
                    )

                if not thread_rows:
                    continue

                thread_rows.sort(key=lambda row: row["timestamp"])
                thread_signs = [stance_sign(row["stance"]) for row in thread_rows if row["stance"] is not None]

                by_author: Dict[str, List[Dict[str, Any]]] = {}
                for row in thread_rows:
                    by_author.setdefault(row["author"], []).append(row)

                for _, author_rows in by_author.items():
                    if len(author_rows) < 2:
                        excluded_single_comment += 1
                        continue

                    author_rows.sort(key=lambda row: row["timestamp"])
                    first_stance = author_rows[0]["stance"]
                    last_stance = author_rows[-1]["stance"]
                    if first_stance is None or last_stance is None:
                        excluded_missing_stance += 1
                        continue

                    first_sign = stance_sign(first_stance)
                    if first_sign == 0:
                        excluded_not_exposed += 1
                        continue

                    exposed = any(sign == (-1 * first_sign) for sign in thread_signs)
                    if not exposed:
                        excluded_not_exposed += 1
                        continue

                    deltas.append(float(last_stance - first_stance))

            value = float(np.mean(deltas)) if deltas else None
            topic_stats[topic_label] = {
                "value": value,
                "sample_size": len(deltas),
                "thread_count": thread_count,
                "excluded_missing_timestamp": excluded_missing_timestamp,
                "excluded_missing_stance": excluded_missing_stance,
                "excluded_single_comment": excluded_single_comment,
                "excluded_not_exposed": excluded_not_exposed,
            }

        return topic_stats

    def cess_by_topic(self, subreddit: Optional[str] = None) -> Dict[str, MetricResult]:
        topic_stats = self._compute_cess_topic_stats(subreddit=subreddit)
        results: Dict[str, MetricResult] = {}
        for topic_label in sorted(topic_stats.keys(), key=lambda item: item.lower()):
            stats = topic_stats[topic_label]
            notes = (
                f"topic={topic_label}; valid_chains={stats['sample_size']}, thread_count={stats['thread_count']}, "
                f"excluded_missing_timestamp={stats['excluded_missing_timestamp']}, "
                f"excluded_missing_stance={stats['excluded_missing_stance']}, "
                f"excluded_single_comment={stats['excluded_single_comment']}, "
                f"excluded_not_exposed={stats['excluded_not_exposed']}; "
                "CESS_s,t=mean(last_stance-first_stance) over exposed user chains; compare topics within the same subreddit"
            )
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"

            results[topic_label] = MetricResult(
                "Counter-Evidence Sentiment Shift (CESS)",
                stats["value"],
                stats["sample_size"],
                notes,
            )

        return results

    def cess(self, subreddit: Optional[str] = None, topic: Optional[str] = None) -> MetricResult:
        by_topic = self.cess_by_topic(subreddit=subreddit)
        if topic:
            cleaned_topic = self._clean_topic_label(topic) or topic
            topic_result = by_topic.get(cleaned_topic)
            if topic_result is None:
                notes = f"No CESS coverage for topic={topic}"
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                return MetricResult("Counter-Evidence Sentiment Shift (CESS)", None, 0, notes)
            return topic_result

        topic_stats = self._compute_cess_topic_stats(subreddit=subreddit)
        weighted_numerator = 0.0
        weighted_denominator = 0
        valid_topics = 0
        total_chains = 0
        for stats in topic_stats.values():
            value = stats.get("value")
            thread_count = int(stats.get("thread_count", 0))
            total_chains += int(stats.get("sample_size", 0))
            if value is None or thread_count <= 0:
                continue

            weighted_numerator += float(value) * thread_count
            weighted_denominator += thread_count
            valid_topics += 1

        if weighted_denominator == 0:
            notes = "No valid topic-level CESS values for subreddit-weighted aggregation"
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"
            return MetricResult("Counter-Evidence Sentiment Shift (CESS)", None, total_chains, notes)

        value = weighted_numerator / float(weighted_denominator)
        notes = (
            f"valid_topics={valid_topics}, weighted_by_threads={weighted_denominator}, total_chains={total_chains}; "
            "CESS_s=sum(alpha_t*CESS_s,t)/sum(alpha_t)"
        )
        if subreddit:
            notes = f"subreddit={subreddit}; {notes}"

        return MetricResult(
            "Counter-Evidence Sentiment Shift (CESS)",
            value,
            total_chains,
            notes,
        )

    def cess_by_subreddit(self) -> Dict[str, MetricResult]:
        results: Dict[str, MetricResult] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.cess(subreddit=subreddit)
        return results

    def cess_by_subreddit_topic(self) -> Dict[str, Dict[str, MetricResult]]:
        results: Dict[str, Dict[str, MetricResult]] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.cess_by_topic(subreddit=subreddit)
        return results

    def eai_by_topic(self, subreddit: Optional[str] = None) -> Dict[str, MetricResult]:
        epsilon = 0.01
        analyzed_by_comment_id = self._collect_analyzed_comments(subreddit=subreddit)
        topic_to_post_ids = self._resolve_topic_to_post_ids(
            subreddit=subreddit,
            analyzed_by_comment_id=analyzed_by_comment_id,
        )

        post_to_topic: Dict[str, str] = {}
        for topic_label, post_ids in topic_to_post_ids.items():
            for post_id in post_ids:
                post_to_topic[post_id] = topic_label

        comments_by_topic: Dict[str, Dict[str, Dict[str, Any]]] = {topic: {} for topic in topic_to_post_ids}
        for comment_id, entry in analyzed_by_comment_id.items():
            meta = self.comment_index.get(comment_id, {})
            post_id = meta.get("post_id") if isinstance(meta, dict) else None
            if not isinstance(post_id, str) or not post_id:
                continue

            topic_label = post_to_topic.get(post_id)
            if not topic_label:
                continue

            parent_id = meta.get("parent_id") if isinstance(meta, dict) else None
            stance = self._extract_stance_value(entry)
            comments_by_topic.setdefault(topic_label, {})[comment_id] = {
                "parent_id": parent_id,
                "stance": stance,
            }

        results: Dict[str, MetricResult] = {}
        for topic_label in sorted(topic_to_post_ids.keys(), key=lambda item: item.lower()):
            topic_comments = comments_by_topic.get(topic_label, {})
            reply_counts: Dict[str, int] = {cid: 0 for cid in topic_comments}
            for _, row in topic_comments.items():
                parent_id = row.get("parent_id")
                if isinstance(parent_id, str) and parent_id in reply_counts:
                    reply_counts[parent_id] += 1

            aligned_ids: List[str] = []
            counter_ids: List[str] = []
            for cid, row in topic_comments.items():
                stance = row.get("stance")
                if isinstance(stance, (int, float)) and stance > 3:
                    aligned_ids.append(cid)
                elif isinstance(stance, (int, float)) and stance < -3:
                    counter_ids.append(cid)

            if not aligned_ids or not counter_ids:
                notes = (
                    f"topic={topic_label}; aligned_n={len(aligned_ids)}, counter_n={len(counter_ids)}; "
                    "requires both groups (stance>3 and stance<-3); neutral [-3,3] excluded"
                )
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                results[topic_label] = MetricResult(
                    "Engagement Asymmetry Index (EAI)",
                    None,
                    len(aligned_ids) + len(counter_ids),
                    notes,
                )
                continue

            engaged_aligned = [1.0 if reply_counts[cid] > 0 else 0.0 for cid in aligned_ids]
            engaged_counter = [1.0 if reply_counts[cid] > 0 else 0.0 for cid in counter_ids]

            engagement_rate_aligned = float(np.mean(engaged_aligned)) if engaged_aligned else 0.0
            engagement_rate_counter = float(np.mean(engaged_counter)) if engaged_counter else 0.0

            denominator = engagement_rate_aligned + engagement_rate_counter + epsilon
            if denominator <= 0:
                notes = (
                    f"topic={topic_label}; aligned_n={len(aligned_ids)}, counter_n={len(counter_ids)}, "
                    f"engagement_rate_aligned={engagement_rate_aligned}, "
                    f"engagement_rate_counter={engagement_rate_counter}, epsilon={epsilon}; denominator near zero"
                )
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                results[topic_label] = MetricResult(
                    "Engagement Asymmetry Index (EAI)",
                    None,
                    len(aligned_ids) + len(counter_ids),
                    notes,
                )
                continue

            value = (engagement_rate_aligned - engagement_rate_counter) / denominator
            notes = (
                f"topic={topic_label}; aligned_n={len(aligned_ids)}, counter_n={len(counter_ids)}, "
                f"engagement_rate_aligned={engagement_rate_aligned}, "
                f"engagement_rate_counter={engagement_rate_counter}, epsilon={epsilon}; "
                "engaged(c)=1 if reply_count(c)>0 else 0; "
                "EAI=(rate_aligned-rate_counter)/(rate_aligned+rate_counter+epsilon); neutral [-3,3] excluded; "
                "compare topics within the same subreddit"
            )
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"

            results[topic_label] = MetricResult(
                "Engagement Asymmetry Index (EAI)",
                value,
                len(aligned_ids) + len(counter_ids),
                notes,
            )

        return results

    def eai(self, subreddit: Optional[str] = None, topic: Optional[str] = None) -> MetricResult:
        by_topic = self.eai_by_topic(subreddit=subreddit)
        if topic:
            cleaned_topic = self._clean_topic_label(topic) or topic
            topic_result = by_topic.get(cleaned_topic)
            if topic_result is None:
                notes = f"No EAI coverage for topic={topic}"
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                return MetricResult("Engagement Asymmetry Index (EAI)", None, 0, notes)
            return topic_result

        if not by_topic:
            detail = f" for subreddit={subreddit}" if subreddit else ""
            return MetricResult("Engagement Asymmetry Index (EAI)", None, 0, f"No topic/comment coverage{detail}")

        weighted_numerator = 0.0
        weighted_denominator = 0
        valid_topics = 0
        total_classified_comments = 0
        for item in by_topic.values():
            total_classified_comments += item.sample_size
            if item.value is None:
                continue
            alpha_t = item.sample_size
            if alpha_t <= 0:
                continue
            weighted_numerator += float(item.value) * alpha_t
            weighted_denominator += alpha_t
            valid_topics += 1

        if weighted_denominator == 0:
            notes = "No valid topic-level EAI values for topic-weighted aggregation"
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"
            return MetricResult("Engagement Asymmetry Index (EAI)", None, total_classified_comments, notes)

        topic_summary = "; ".join(
            f"{topic_name}={result.value:.4f}" if result.value is not None else f"{topic_name}=None"
            for topic_name, result in sorted(by_topic.items(), key=lambda item: item[0].lower())
        )
        notes = (
            f"topics={len(by_topic)}, valid_topics={valid_topics}, "
            f"weighted_by_alpha_t_comments={weighted_denominator}, classified_comments_total={total_classified_comments}; "
            f"topic_values: {topic_summary}; EAI_s=sum(alpha_t*EAI_s,t)/sum(alpha_t), alpha_t=classified comments per topic; "
            "compare topics within the same subreddit"
        )
        if subreddit:
            notes = f"subreddit={subreddit}; {notes}"

        return MetricResult(
            "Engagement Asymmetry Index (EAI)",
            weighted_numerator / float(weighted_denominator),
            total_classified_comments,
            notes,
        )

    def eai_by_subreddit(self) -> Dict[str, MetricResult]:
        results: Dict[str, MetricResult] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.eai(subreddit=subreddit)
        return results

    def eai_by_subreddit_topic(self) -> Dict[str, Dict[str, MetricResult]]:
        results: Dict[str, Dict[str, MetricResult]] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.eai_by_topic(subreddit=subreddit)
        return results

    def csir_by_topic(self, subreddit: Optional[str] = None) -> Dict[str, MetricResult]:
        analyzed_by_comment_id = self._collect_analyzed_comments(subreddit=subreddit)
        topic_to_post_ids = self._resolve_topic_to_post_ids(
            subreddit=subreddit,
            analyzed_by_comment_id=analyzed_by_comment_id,
        )

        post_to_topic: Dict[str, str] = {}
        for topic_label, post_ids in topic_to_post_ids.items():
            for post_id in post_ids:
                post_to_topic[post_id] = topic_label

        comments_by_topic: Dict[str, Dict[str, Dict[str, Any]]] = {topic: {} for topic in topic_to_post_ids}
        for comment_id, entry in analyzed_by_comment_id.items():
            if not isinstance(entry, dict):
                continue

            meta = self.comment_index.get(comment_id, {})
            post_id = meta.get("post_id") if isinstance(meta, dict) else None
            parent_id = meta.get("parent_id") if isinstance(meta, dict) else None
            if not isinstance(post_id, str) or not post_id:
                continue

            topic_label = post_to_topic.get(post_id)
            if not topic_label:
                continue

            comments_by_topic.setdefault(topic_label, {})[comment_id] = {
                "parent_id": parent_id,
                "post_id": post_id,
                "stance": self._extract_stance_value(entry),
            }

        def resolve_thread_root(comment_id: str, topic_comments: Dict[str, Dict[str, Any]]) -> Optional[str]:
            current = comment_id
            visited = set()
            while True:
                if current in visited:
                    return None
                visited.add(current)

                row = topic_comments.get(current)
                if not isinstance(row, dict):
                    return None
                parent_id = row.get("parent_id")
                post_id = row.get("post_id")

                if not isinstance(parent_id, str) or not isinstance(post_id, str):
                    return None
                if parent_id == post_id:
                    return current

                if parent_id not in topic_comments:
                    return None
                current = parent_id

        results: Dict[str, MetricResult] = {}
        for topic_label in sorted(topic_to_post_ids.keys(), key=lambda item: item.lower()):
            topic_comments = comments_by_topic.get(topic_label, {})

            threads: Dict[str, List[str]] = {}
            unresolved_thread = 0
            for comment_id in topic_comments.keys():
                root = resolve_thread_root(comment_id, topic_comments)
                if not isinstance(root, str) or not root:
                    unresolved_thread += 1
                    continue
                threads.setdefault(root, []).append(comment_id)

            thread_csir_values: List[float] = []
            skipped_small_pairs = 0
            skipped_undefined_expected = 0
            excluded_non_strong_or_missing = 0
            eligible_pairs_total = 0
            cross_pairs_total = 0

            for _, thread_comment_ids in threads.items():
                thread_id_set = set(thread_comment_ids)
                eligible_pairs: List[tuple] = []

                for child_id in thread_comment_ids:
                    child_row = topic_comments.get(child_id, {})
                    parent_id = child_row.get("parent_id") if isinstance(child_row, dict) else None
                    if not isinstance(parent_id, str) or parent_id not in thread_id_set:
                        continue

                    parent_row = topic_comments.get(parent_id, {})
                    parent_stance = parent_row.get("stance") if isinstance(parent_row, dict) else None
                    child_stance = child_row.get("stance") if isinstance(child_row, dict) else None

                    if not isinstance(parent_stance, (int, float)) or not isinstance(child_stance, (int, float)):
                        excluded_non_strong_or_missing += 1
                        continue
                    if abs(parent_stance) <= 3 or abs(child_stance) <= 3:
                        excluded_non_strong_or_missing += 1
                        continue

                    eligible_pairs.append((parent_id, child_id))

                if len(eligible_pairs) < 5:
                    skipped_small_pairs += 1
                    continue

                cross_pairs = 0
                eligible_comment_ids = set()
                for parent_id, child_id in eligible_pairs:
                    parent_stance = topic_comments[parent_id]["stance"]
                    child_stance = topic_comments[child_id]["stance"]
                    eligible_comment_ids.add(parent_id)
                    eligible_comment_ids.add(child_id)
                    if (parent_stance * child_stance) < 0:
                        cross_pairs += 1

                observed_rate = cross_pairs / float(len(eligible_pairs))

                aligned_count = 0
                counter_count = 0
                for cid in eligible_comment_ids:
                    stance = topic_comments[cid]["stance"]
                    if stance > 3:
                        aligned_count += 1
                    elif stance < -3:
                        counter_count += 1

                eligible_comment_count = len(eligible_comment_ids)
                if eligible_comment_count == 0:
                    skipped_undefined_expected += 1
                    continue

                p_aligned = aligned_count / float(eligible_comment_count)
                p_counter = counter_count / float(eligible_comment_count)
                expected_rate = 2.0 * p_aligned * p_counter

                if expected_rate <= 0.0 or expected_rate >= 1.0:
                    skipped_undefined_expected += 1
                    continue

                csir_thread = (observed_rate - expected_rate) / (1.0 - expected_rate)
                thread_csir_values.append(float(csir_thread))
                eligible_pairs_total += len(eligible_pairs)
                cross_pairs_total += cross_pairs

            value = float(np.mean(thread_csir_values)) if thread_csir_values else None
            notes = (
                f"topic={topic_label}; valid_threads={len(thread_csir_values)}, total_threads={len(threads)}, "
                f"eligible_pairs_total={eligible_pairs_total}, cross_stance_pairs_total={cross_pairs_total}, "
                f"excluded_non_strong_or_missing={excluded_non_strong_or_missing}, "
                f"skipped_small_pairs(<5)={skipped_small_pairs}, skipped_undefined_expected={skipped_undefined_expected}, "
                f"unresolved_thread_comments={unresolved_thread}; "
                "per-thread CSIR_r=(observed_rate_r-expected_rate_r)/(1-expected_rate_r), expected_rate_r=2*p_aligned*p_counter; "
                "CSIR_s,t=mean(CSIR_r); compare topics within the same subreddit"
            )
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"

            results[topic_label] = MetricResult(
                "Cross-Stance Interaction Rate (CSIR)",
                value,
                len(thread_csir_values),
                notes,
            )

        return results

    def csir(self, subreddit: Optional[str] = None, topic: Optional[str] = None) -> MetricResult:
        by_topic = self.csir_by_topic(subreddit=subreddit)
        if topic:
            cleaned_topic = self._clean_topic_label(topic) or topic
            topic_result = by_topic.get(cleaned_topic)
            if topic_result is None:
                notes = f"No CSIR coverage for topic={topic}"
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                return MetricResult("Cross-Stance Interaction Rate (CSIR)", None, 0, notes)
            return topic_result

        if not by_topic:
            detail = f" for subreddit={subreddit}" if subreddit else ""
            return MetricResult("Cross-Stance Interaction Rate (CSIR)", None, 0, f"No topic/comment coverage{detail}")

        weighted_numerator = 0.0
        weighted_denominator = 0
        valid_topics = 0
        total_valid_threads = 0
        for item in by_topic.values():
            total_valid_threads += item.sample_size
            if item.value is None:
                continue
            alpha_t = item.sample_size
            if alpha_t <= 0:
                continue
            weighted_numerator += float(item.value) * alpha_t
            weighted_denominator += alpha_t
            valid_topics += 1

        if weighted_denominator == 0:
            notes = "No valid thread-normalized CSIR values for topic-weighted aggregation"
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"
            return MetricResult("Cross-Stance Interaction Rate (CSIR)", None, total_valid_threads, notes)

        topic_summary = "; ".join(
            f"{topic_name}={result.value:.4f}" if result.value is not None else f"{topic_name}=None"
            for topic_name, result in sorted(by_topic.items(), key=lambda item: item[0].lower())
        )
        notes = (
            f"topics={len(by_topic)}, valid_topics={valid_topics}, weighted_by_alpha_t_threads={weighted_denominator}, "
            f"total_valid_threads={total_valid_threads}; topic_values: {topic_summary}; "
            "CSIR_s=sum(alpha_t*CSIR_s,t)/sum(alpha_t), alpha_t=valid thread count per topic; "
            "compare topics within the same subreddit"
        )
        if subreddit:
            notes = f"subreddit={subreddit}; {notes}"

        return MetricResult(
            "Cross-Stance Interaction Rate (CSIR)",
            weighted_numerator / float(weighted_denominator),
            total_valid_threads,
            notes,
        )

    def csir_by_subreddit(self) -> Dict[str, MetricResult]:
        results: Dict[str, MetricResult] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.csir(subreddit=subreddit)
        return results

    def csir_by_subreddit_topic(self) -> Dict[str, Dict[str, MetricResult]]:
        results: Dict[str, Dict[str, MetricResult]] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.csir_by_topic(subreddit=subreddit)
        return results

    def voc_by_topic(self, subreddit: Optional[str] = None) -> Dict[str, MetricResult]:
        analyzed_by_comment_id = self._collect_analyzed_comments(subreddit=subreddit)
        topic_to_post_ids = self._resolve_topic_to_post_ids(
            subreddit=subreddit,
            analyzed_by_comment_id=analyzed_by_comment_id,
        )

        post_to_topic: Dict[str, str] = {}
        for topic_label, post_ids in topic_to_post_ids.items():
            for post_id in post_ids:
                post_to_topic[post_id] = topic_label

        topic_rows: Dict[str, List[Dict[str, float]]] = {topic: [] for topic in topic_to_post_ids}
        for comment_id, entry in analyzed_by_comment_id.items():
            if not isinstance(entry, dict):
                continue

            meta = self.comment_index.get(comment_id, {})
            post_id = meta.get("post_id") if isinstance(meta, dict) else None
            if not isinstance(post_id, str) or not post_id:
                continue

            topic_label = post_to_topic.get(post_id)
            if not topic_label:
                continue

            stance = self._extract_stance_value(entry)
            upvotes_raw = meta.get("upvotes") if isinstance(meta, dict) else None
            upvotes = None
            if isinstance(upvotes_raw, (int, float)):
                upvotes = float(upvotes_raw)
            elif isinstance(upvotes_raw, str):
                candidate = upvotes_raw.strip()
                if candidate and candidate.upper() != "N/A":
                    try:
                        upvotes = float(candidate)
                    except ValueError:
                        upvotes = None

            if stance is None or upvotes is None:
                continue

            topic_rows.setdefault(topic_label, []).append(
                {
                    "stance": float(stance),
                    "upvotes": upvotes,
                }
            )

        rng = np.random.default_rng(42)

        def avg_pairwise_distance(stances: List[float]) -> Optional[float]:
            if len(stances) < 2:
                return None

            working = stances
            if len(working) > 500:
                sampled_idx = rng.choice(len(working), size=500, replace=False)
                working = [working[int(i)] for i in sampled_idx]

            arr = np.array(working, dtype=float)
            diff = np.abs(arr[:, None] - arr[None, :]) / 20.0
            iu = np.triu_indices(len(arr), k=1)
            pair_values = diff[iu]
            if pair_values.size == 0:
                return None
            return float(np.mean(pair_values))

        results: Dict[str, MetricResult] = {}
        for topic_label in sorted(topic_to_post_ids.keys(), key=lambda item: item.lower()):
            rows = topic_rows.get(topic_label, [])
            if len(rows) < 2:
                notes = f"topic={topic_label}; C_full={len(rows)}; requires C_full>=2"
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                results[topic_label] = MetricResult("Visible Opinion Compression (VOC)", None, len(rows), notes)
                continue

            threshold = float(np.quantile([row["upvotes"] for row in rows], 0.75))
            visible_rows = [row for row in rows if row["upvotes"] >= threshold]
            if len(visible_rows) < 2:
                notes = (
                    f"topic={topic_label}; C_full={len(rows)}, C_visible={len(visible_rows)}, "
                    f"upvote_p75={threshold}; requires C_visible>=2"
                )
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                results[topic_label] = MetricResult("Visible Opinion Compression (VOC)", None, len(rows), notes)
                continue

            full_dist = avg_pairwise_distance([row["stance"] for row in rows])
            visible_dist = avg_pairwise_distance([row["stance"] for row in visible_rows])
            if full_dist is None or visible_dist is None:
                notes = f"topic={topic_label}; insufficient pairwise distances"
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                results[topic_label] = MetricResult("Visible Opinion Compression (VOC)", None, len(rows), notes)
                continue

            voc_full = 1.0 - full_dist
            voc_visible = 1.0 - visible_dist
            delta_voc = voc_visible - voc_full
            value = {
                "voc_visible": voc_visible,
                "voc_full": voc_full,
                "delta_voc": delta_voc,
            }
            notes = (
                f"topic={topic_label}; C_full={len(rows)}, C_visible={len(visible_rows)}, upvote_p75={threshold}; "
                "pair distance normalized by /20, VOC=1-avg_pairwise_distance"
            )
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"

            results[topic_label] = MetricResult("Visible Opinion Compression (VOC)", value, len(rows), notes)

        return results

    def voc(self, subreddit: Optional[str] = None, topic: Optional[str] = None) -> MetricResult:
        by_topic = self.voc_by_topic(subreddit=subreddit)
        if topic:
            cleaned_topic = self._clean_topic_label(topic) or topic
            topic_result = by_topic.get(cleaned_topic)
            if topic_result is None:
                notes = f"No VOC coverage for topic={topic}"
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                return MetricResult("Visible Opinion Compression (VOC)", None, 0, notes)
            return topic_result

        if not by_topic:
            detail = f" for subreddit={subreddit}" if subreddit else ""
            return MetricResult("Visible Opinion Compression (VOC)", None, 0, f"No topic/comment coverage{detail}")

        vectors = [item.value for item in by_topic.values() if isinstance(item.value, dict)]
        total_valid = sum(item.sample_size for item in by_topic.values())
        if not vectors:
            notes = "No valid topic-level VOC values"
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"
            return MetricResult("Visible Opinion Compression (VOC)", None, total_valid, notes)

        value = {
            "voc_visible": float(np.mean([float(v.get("voc_visible", 0.0)) for v in vectors])),
            "voc_full": float(np.mean([float(v.get("voc_full", 0.0)) for v in vectors])),
            "delta_voc": float(np.mean([float(v.get("delta_voc", 0.0)) for v in vectors])),
        }
        notes = (
            f"topics={len(by_topic)}, valid_comments_total={total_valid}; "
            "value is simple mean of topic-level VOC vectors (voc_visible/voc_full/delta_voc)"
        )
        if subreddit:
            notes = f"subreddit={subreddit}; {notes}"

        return MetricResult("Visible Opinion Compression (VOC)", value, total_valid, notes)

    def voc_by_subreddit(self) -> Dict[str, MetricResult]:
        results: Dict[str, MetricResult] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.voc(subreddit=subreddit)
        return results

    def voc_by_subreddit_topic(self) -> Dict[str, Dict[str, MetricResult]]:
        results: Dict[str, Dict[str, MetricResult]] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.voc_by_topic(subreddit=subreddit)
        return results

    def vsr(self) -> MetricResult:
        return MetricResult("Visibility Suppression Ratio (VSR)", None, len(self.entries), "Pending formula confirmation")

    def lsca_by_topic(self, subreddit: Optional[str] = None) -> Dict[str, MetricResult]:
        analyzed_by_comment_id = self._collect_analyzed_comments(subreddit=subreddit)
        topic_to_post_ids = self._resolve_topic_to_post_ids(
            subreddit=subreddit,
            analyzed_by_comment_id=analyzed_by_comment_id,
        )

        post_to_topic: Dict[str, str] = {}
        for topic_label, post_ids in topic_to_post_ids.items():
            for post_id in post_ids:
                post_to_topic[post_id] = topic_label

        comments_by_topic: Dict[str, Dict[str, Dict[str, Any]]] = {topic: {} for topic in topic_to_post_ids}
        for comment_id, entry in analyzed_by_comment_id.items():
            if not isinstance(entry, dict):
                continue

            meta = self.comment_index.get(comment_id, {})
            post_id = meta.get("post_id") if isinstance(meta, dict) else None
            parent_id = meta.get("parent_id") if isinstance(meta, dict) else None
            if not isinstance(post_id, str) or not post_id:
                continue

            topic_label = post_to_topic.get(post_id)
            if not topic_label:
                continue

            claim_strength = self._extract_epistemic_numeric_field(entry, "claim_strength")
            evidence_quality = self._extract_epistemic_numeric_field(entry, "evidence_quality")
            reasoning_depth = self._extract_epistemic_numeric_field(entry, "reasoning_depth")
            stance = self._extract_stance_value(entry)

            comments_by_topic.setdefault(topic_label, {})[comment_id] = {
                "parent_id": parent_id,
                "claim_strength": claim_strength,
                "evidence_quality": evidence_quality,
                "reasoning_depth": reasoning_depth,
                "stance": stance,
            }

        results: Dict[str, MetricResult] = {}
        for topic_label in sorted(topic_to_post_ids.keys(), key=lambda item: item.lower()):
            topic_comments = comments_by_topic.get(topic_label, {})

            reply_counts: Dict[str, int] = {cid: 0 for cid in topic_comments}
            for _, row in topic_comments.items():
                parent_id = row.get("parent_id")
                if isinstance(parent_id, str) and parent_id in reply_counts:
                    reply_counts[parent_id] += 1

            high_risk_ids: set = set()
            low_risk_ids: set = set()
            counter_evidence_ids: set = set()

            excluded_missing_epistemic = 0
            for cid, row in topic_comments.items():
                claim_strength = row.get("claim_strength")
                evidence_quality = row.get("evidence_quality")
                reasoning_depth = row.get("reasoning_depth")
                stance = row.get("stance")

                if claim_strength is None or evidence_quality is None:
                    excluded_missing_epistemic += 1
                else:
                    if claim_strength >= 7 and evidence_quality <= 3:
                        high_risk_ids.add(cid)
                    if claim_strength <= 3 or evidence_quality >= 6:
                        low_risk_ids.add(cid)

                if (
                    isinstance(stance, (int, float))
                    and stance < 0
                    and isinstance(evidence_quality, (int, float))
                    and isinstance(reasoning_depth, (int, float))
                    and (evidence_quality >= 4 or reasoning_depth >= 4)
                ):
                    counter_evidence_ids.add(cid)

            low_risk_ids = low_risk_ids.union(counter_evidence_ids)

            if len(high_risk_ids) == 0 or len(low_risk_ids) < 5:
                notes = (
                    f"topic={topic_label}; high_risk_n={len(high_risk_ids)}, low_risk_n={len(low_risk_ids)}, "
                    f"counter_evidence_added={len(counter_evidence_ids)}, excluded_missing_epistemic={excluded_missing_epistemic}; "
                    "requires high_risk_n>0 and low_risk_n>=5"
                )
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                results[topic_label] = MetricResult(
                    "Low-Support Claim Amplification (LSCA)",
                    None,
                    len(high_risk_ids) + len(low_risk_ids),
                    notes,
                )
                continue

            avg_eng_high = float(np.mean([reply_counts[cid] for cid in high_risk_ids]))
            avg_eng_low = float(np.mean([reply_counts[cid] for cid in low_risk_ids]))
            if avg_eng_low == 0:
                notes = (
                    f"topic={topic_label}; high_risk_n={len(high_risk_ids)}, low_risk_n={len(low_risk_ids)}, "
                    f"avg_eng_high={avg_eng_high}, avg_eng_low={avg_eng_low}; denominator zero"
                )
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                results[topic_label] = MetricResult(
                    "Low-Support Claim Amplification (LSCA)",
                    None,
                    len(high_risk_ids) + len(low_risk_ids),
                    notes,
                )
                continue

            value = avg_eng_high / avg_eng_low
            notes = (
                f"topic={topic_label}; high_risk_n={len(high_risk_ids)}, low_risk_n={len(low_risk_ids)}, "
                f"counter_evidence_added={len(counter_evidence_ids)}, avg_eng_high={avg_eng_high}, avg_eng_low={avg_eng_low}; "
                "high_risk: claim_strength>=7 and evidence_quality<=3; "
                "low_risk: claim_strength<=3 or evidence_quality>=6 plus counter_evidence union; compare topics within the same subreddit"
            )
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"

            results[topic_label] = MetricResult(
                "Low-Support Claim Amplification (LSCA)",
                value,
                len(high_risk_ids) + len(low_risk_ids),
                notes,
            )

        return results

    def lsca(self, subreddit: Optional[str] = None, topic: Optional[str] = None) -> MetricResult:
        by_topic = self.lsca_by_topic(subreddit=subreddit)
        if topic:
            cleaned_topic = self._clean_topic_label(topic) or topic
            topic_result = by_topic.get(cleaned_topic)
            if topic_result is None:
                notes = f"No LSCA coverage for topic={topic}"
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                return MetricResult("Low-Support Claim Amplification (LSCA)", None, 0, notes)
            return topic_result

        if not by_topic:
            detail = f" for subreddit={subreddit}" if subreddit else ""
            return MetricResult("Low-Support Claim Amplification (LSCA)", None, 0, f"No topic/comment coverage{detail}")

        values = [item.value for item in by_topic.values() if item.value is not None]
        total_classified = sum(item.sample_size for item in by_topic.values())
        if not values:
            notes = "No valid topic-level LSCA values"
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"
            return MetricResult("Low-Support Claim Amplification (LSCA)", None, total_classified, notes)

        topic_summary = "; ".join(
            f"{topic_name}={result.value:.4f}" if result.value is not None else f"{topic_name}=None"
            for topic_name, result in sorted(by_topic.items(), key=lambda item: item[0].lower())
        )
        notes = (
            f"topics={len(by_topic)}, classified_total={total_classified}; "
            f"topic_values: {topic_summary}; compare topics within the same subreddit"
        )
        if subreddit:
            notes = f"subreddit={subreddit}; {notes}"

        return MetricResult(
            "Low-Support Claim Amplification (LSCA)",
            float(np.mean(values)),
            total_classified,
            notes,
        )

    def lsca_by_subreddit(self) -> Dict[str, MetricResult]:
        results: Dict[str, MetricResult] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.lsca(subreddit=subreddit)
        return results

    def lsca_by_subreddit_topic(self) -> Dict[str, Dict[str, MetricResult]]:
        results: Dict[str, Dict[str, MetricResult]] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.lsca_by_topic(subreddit=subreddit)
        return results

    def eas_by_topic(self, subreddit: Optional[str] = None) -> Dict[str, MetricResult]:
        analyzed_by_comment_id = self._collect_analyzed_comments(subreddit=subreddit)
        topic_to_post_ids = self._resolve_topic_to_post_ids(
            subreddit=subreddit,
            analyzed_by_comment_id=analyzed_by_comment_id,
        )

        post_to_topic: Dict[str, str] = {}
        for topic_label, post_ids in topic_to_post_ids.items():
            for post_id in post_ids:
                post_to_topic[post_id] = topic_label

        topic_rows: Dict[str, Dict[str, List[tuple[float, float]]]] = {
            topic: {"anger": [], "anxiety": [], "disgust": []}
            for topic in topic_to_post_ids
        }

        def parse_emotion(entry: Dict[str, Any], field: str) -> Optional[float]:
            if not isinstance(entry, dict):
                return None
            emotion = entry.get("emotion")
            if not isinstance(emotion, dict):
                return None
            raw = emotion.get(field)
            if isinstance(raw, (int, float)):
                return float(raw)
            if isinstance(raw, str):
                candidate = raw.strip()
                if not candidate or candidate.upper() == "N/A":
                    return None
                try:
                    return float(candidate)
                except ValueError:
                    return None
            return None

        def parse_upvotes(meta: Dict[str, Any]) -> Optional[float]:
            raw = meta.get("upvotes") if isinstance(meta, dict) else None
            if isinstance(raw, (int, float)):
                return float(raw)
            if isinstance(raw, str):
                candidate = raw.strip()
                if not candidate or candidate.upper() == "N/A":
                    return None
                try:
                    return float(candidate)
                except ValueError:
                    return None
            return None

        for comment_id, entry in analyzed_by_comment_id.items():
            if not isinstance(entry, dict):
                continue

            meta = self.comment_index.get(comment_id, {})
            post_id = meta.get("post_id") if isinstance(meta, dict) else None
            if not isinstance(post_id, str) or not post_id:
                continue

            topic_label = post_to_topic.get(post_id)
            if not topic_label:
                continue

            upvotes = parse_upvotes(meta)
            if upvotes is None:
                continue

            for emotion in ("anger", "anxiety", "disgust"):
                emotion_value = parse_emotion(entry, emotion)
                if emotion_value is None:
                    continue
                topic_rows.setdefault(topic_label, {"anger": [], "anxiety": [], "disgust": []})[emotion].append(
                    (upvotes, emotion_value)
                )

        results: Dict[str, MetricResult] = {}
        for topic_label in sorted(topic_to_post_ids.keys(), key=lambda item: item.lower()):
            rows_by_emotion = topic_rows.get(topic_label, {"anger": [], "anxiety": [], "disgust": []})

            value: Dict[str, Any] = {}
            n_map: Dict[str, int] = {}
            for emotion in ("anger", "anxiety", "disgust"):
                pairs = rows_by_emotion.get(emotion, [])
                n = len(pairs)
                n_map[emotion] = n

                if n < 10:
                    value[emotion] = None
                    continue

                up = np.array([pair[0] for pair in pairs], dtype=float)
                em = np.array([pair[1] for pair in pairs], dtype=float)
                rho, pvalue = spearmanr(up, em)

                if np.isnan(rho) or np.isnan(pvalue):
                    value[emotion] = None
                    continue

                value[emotion] = {
                    "rho": round(float(rho), 4),
                    "pvalue": round(float(pvalue), 4),
                    "n": n,
                }

            valid_any = any(isinstance(value.get(emotion), dict) for emotion in ("anger", "anxiety", "disgust"))
            if not valid_any:
                value = None

            notes = (
                f"topic={topic_label}; n_by_emotion={n_map}; "
                "EAS uses Spearman correlation between upvotes and each emotion; min_n_per_emotion=10"
            )
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"

            results[topic_label] = MetricResult(
                "Emotional Amplification Score (EAS)",
                value,
                max(n_map.values()) if n_map else 0,
                notes,
            )

        return results

    def eas(self, subreddit: Optional[str] = None, topic: Optional[str] = None) -> MetricResult:
        if topic:
            by_topic = self.eas_by_topic(subreddit=subreddit)
            cleaned_topic = self._clean_topic_label(topic) or topic
            topic_result = by_topic.get(cleaned_topic)
            if topic_result is None:
                notes = f"No EAS coverage for topic={topic}"
                if subreddit:
                    notes = f"subreddit={subreddit}; {notes}"
                return MetricResult("Emotional Amplification Score (EAS)", None, 0, notes)
            return topic_result

        analyzed_by_comment_id = self._collect_analyzed_comments(subreddit=subreddit)
        if not analyzed_by_comment_id:
            detail = f" for subreddit={subreddit}" if subreddit else ""
            return MetricResult("Emotional Amplification Score (EAS)", None, 0, f"No analyzable comments{detail}")

        def parse_emotion(entry: Dict[str, Any], field: str) -> Optional[float]:
            if not isinstance(entry, dict):
                return None
            emotion = entry.get("emotion")
            if not isinstance(emotion, dict):
                return None
            raw = emotion.get(field)
            if isinstance(raw, (int, float)):
                return float(raw)
            if isinstance(raw, str):
                candidate = raw.strip()
                if not candidate or candidate.upper() == "N/A":
                    return None
                try:
                    return float(candidate)
                except ValueError:
                    return None
            return None

        def parse_upvotes(meta: Dict[str, Any]) -> Optional[float]:
            raw = meta.get("upvotes") if isinstance(meta, dict) else None
            if isinstance(raw, (int, float)):
                return float(raw)
            if isinstance(raw, str):
                candidate = raw.strip()
                if not candidate or candidate.upper() == "N/A":
                    return None
                try:
                    return float(candidate)
                except ValueError:
                    return None
            return None

        rows_by_emotion: Dict[str, List[tuple[float, float]]] = {"anger": [], "anxiety": [], "disgust": []}
        for comment_id, entry in analyzed_by_comment_id.items():
            if not isinstance(entry, dict):
                continue
            meta = self.comment_index.get(comment_id, {})
            upvotes = parse_upvotes(meta)
            if upvotes is None:
                continue

            for emotion in ("anger", "anxiety", "disgust"):
                emotion_value = parse_emotion(entry, emotion)
                if emotion_value is None:
                    continue
                rows_by_emotion[emotion].append((upvotes, emotion_value))

        value: Dict[str, Any] = {}
        n_map: Dict[str, int] = {}
        for emotion in ("anger", "anxiety", "disgust"):
            pairs = rows_by_emotion.get(emotion, [])
            n = len(pairs)
            n_map[emotion] = n

            if n < 10:
                value[emotion] = None
                continue

            up = np.array([pair[0] for pair in pairs], dtype=float)
            em = np.array([pair[1] for pair in pairs], dtype=float)
            rho, pvalue = spearmanr(up, em)
            if np.isnan(rho) or np.isnan(pvalue):
                value[emotion] = None
                continue

            value[emotion] = {
                "rho": round(float(rho), 4),
                "pvalue": round(float(pvalue), 4),
                "n": n,
            }

        if not any(isinstance(value.get(emotion), dict) for emotion in ("anger", "anxiety", "disgust")):
            notes = f"No valid EAS correlations with min_n_per_emotion=10; n_by_emotion={n_map}"
            if subreddit:
                notes = f"subreddit={subreddit}; {notes}"
            return MetricResult("Emotional Amplification Score (EAS)", None, max(n_map.values()) if n_map else 0, notes)

        notes = (
            f"n_by_emotion={n_map}; "
            "EAS uses Spearman correlation between upvotes and each emotion at subreddit level; min_n_per_emotion=10"
        )
        if subreddit:
            notes = f"subreddit={subreddit}; {notes}"

        return MetricResult(
            "Emotional Amplification Score (EAS)",
            value,
            max(n_map.values()) if n_map else 0,
            notes,
        )

    def eas_by_subreddit(self) -> Dict[str, MetricResult]:
        results: Dict[str, MetricResult] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.eas(subreddit=subreddit)
        return results

    def eas_by_subreddit_topic(self) -> Dict[str, Dict[str, MetricResult]]:
        results: Dict[str, Dict[str, MetricResult]] = {}
        for subreddit in self.get_subreddits():
            results[subreddit] = self.eas_by_topic(subreddit=subreddit)
        return results
