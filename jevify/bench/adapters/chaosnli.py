"""ChaosNLI: 100 human labels per NLI item. The gold standard for asking whether
a model's *distribution* matches human uncertainty, not just its argmax.

Nie, Zhou & Bansal (EMNLP 2020), https://github.com/easonnie/ChaosNLI (CC BY-SA 4.0).
Not on the Hub, so this adapter downloads the release archive directly.
"""
from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from typing import Any, Sequence

from ..record import BenchRecord, Split
from ._base import Adapter, SourceSpec
from .choice import NLI_CRITERIA

CHAOSNLI_URL = "https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1"
_FILES = {"snli": "chaosNLI_snli.jsonl", "mnli": "chaosNLI_mnli_m.jsonl"}
_LABELS = {"e": "entailment", "n": "neutral", "c": "contradiction"}


class ChaosNLI(Adapter):
    spec = SourceSpec(
        name="chaosnli", hf_id="", primitive="choice", license="cc-by-sa-4.0", domain="nli", task_family="nli",
        k=3, has_soft_labels=True,
        description="SNLI/MNLI items re-annotated by 100 crowdworkers each; soft_label = human vote shares (calibration gold).",
        caps={"train": 0, "validation": 0, "test": 4000},
        notes="Evaluation only. Source URL: " + CHAOSNLI_URL,
    )
    label_column = "majority_label"

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] | None = None

    def _download(self) -> list[dict[str, Any]]:
        req = urllib.request.Request(CHAOSNLI_URL, headers={"User-Agent": "jevify-bench/0.1"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            blob = resp.read()
        rows: list[dict[str, Any]] = []
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            members = {name.split("/")[-1]: name for name in zf.namelist()}
            for subset, fname in _FILES.items():
                with zf.open(members[fname]) as f:
                    for line in io.TextIOWrapper(f, encoding="utf-8"):
                        if not line.strip():
                            continue
                        d = json.loads(line)
                        counter = d["label_counter"]
                        total = sum(counter.values())
                        dist = {_LABELS[k]: counter.get(k, 0) / total for k in ("e", "n", "c")}
                        rows.append({
                            "uid": d["uid"], "subset": subset,
                            "premise": d["example"]["premise"], "hypothesis": d["example"]["hypothesis"],
                            "majority_label": _LABELS[d["majority_label"]], "old_label": _LABELS.get(d.get("old_label"), None),
                            "dist": dist, "n_annotators": total, "entropy": d.get("entropy"),
                        })
        return rows

    def load(self, split: Split) -> Sequence[dict[str, Any]]:
        if split != "test":
            return []
        if self._rows is None:
            self._rows = self._download()
        return self._rows

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        return self.record(split, idx, state={"premise": row["premise"], "hypothesis": row["hypothesis"]},
                           question={"type": "choice",
                                     "instructions": "What is the relationship of `hypothesis` to `premise`?",
                                     "criteria": NLI_CRITERIA},
                           label=row["majority_label"], soft_label=row["dist"],
                           subset=row["subset"], uid=row["uid"], original_label=row["old_label"],
                           n_annotators=row["n_annotators"], human_entropy=row["entropy"])


ADAPTERS = [ChaosNLI]
