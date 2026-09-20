"""ChaosNLI: 100 human labels per NLI item. The gold standard for asking whether
a model's *distribution* matches human uncertainty, not just its argmax.

Nie, Zhou & Bansal (EMNLP 2020), https://github.com/easonnie/ChaosNLI (CC BY-SA 4.0).
The original Dropbox release has been deleted; we use the Hub mirror of the
MNLI portion (``metaeval/chaos-mnli-ambiguity``), which preserves the original
fields (``label_counter``, ``label_dist``, ``old_label``, ``entropy``).
"""
from __future__ import annotations

from typing import Any

from ..record import BenchRecord, Split
from ._base import HFAdapter, SourceSpec
from .choice import NLI_CRITERIA

_LABELS = {"e": "entailment", "n": "neutral", "c": "contradiction"}


class ChaosNLI(HFAdapter):
    spec = SourceSpec(
        name="chaosnli", hf_id="metaeval/chaos-mnli-ambiguity", hf_config="default", primitive="choice",
        license="cc-by-sa-4.0 (ChaosNLI; Hub mirror of the MNLI portion)", domain="nli", task_family="nli",
        k=3, has_soft_labels=True,
        description="MNLI items re-annotated by 100 crowdworkers each; soft_label = human vote shares (calibration gold).",
        caps={"train": 0, "validation": 0, "test": 2000},
        notes="Evaluation only. Original release: https://github.com/easonnie/ChaosNLI",
    )
    label_column = "majority_label"
    split_map = {"test": "train"}   # the mirror ships one split; all of it is evaluation data

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        counter = row["label_counter"]
        total = sum(int(counter[k]) for k in ("e", "n", "c") if counter.get(k) is not None)
        if total == 0:
            return None
        dist = {_LABELS[k]: int(counter.get(k) or 0) / total for k in ("e", "n", "c")}
        majority = _LABELS[row["majority_label"]]
        return self.record(split, idx, state={"premise": row["premise"], "hypothesis": row["hypothesis"]},
                           question={"type": "choice",
                                     "instructions": "What is the relationship of `hypothesis` to `premise`?",
                                     "criteria": NLI_CRITERIA},
                           label=majority, soft_label=dist,
                           uid=row["uid"], original_label=_LABELS.get(row.get("old_label")),
                           n_annotators=total, human_entropy=row.get("entropy"), gini=row.get("gini"))


ADAPTERS = [ChaosNLI]
