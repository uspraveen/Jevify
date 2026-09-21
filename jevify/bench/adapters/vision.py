"""Vision sources: the same three primitives, asked about an image.

Images stay as PIL objects on the record's ``state``; the JSONL writer would not survive
them, so vision configs are built in memory and consumed directly by the vision runner
(see ``jevify.engine.vision``). ``image_dir`` writes them alongside as PNGs when a config
needs to be shipped.
"""
from __future__ import annotations

from typing import Any

from ..record import BenchRecord, Split
from ._base import HFAdapter, SourceSpec


class POPE(HFAdapter):
    """Object hallucination: 'Is there a <object> in the image?' with a yes/no ground truth.

    The canonical VLM hallucination benchmark, and a natural Noul: the question is absolute,
    the answer set is fixed, and the interesting quantity is *how confidently* a model asserts
    an object that is not there. ``category`` splits the negatives into random, popular
    (frequent objects) and adversarial (objects that co-occur with what is present).
    """
    spec = SourceSpec(
        name="pope", hf_id="lmms-lab/POPE", hf_config="default", primitive="noul",
        license="apache-2.0 (POPE; images MS-COCO, CC-BY-4.0)", domain="vision-hallucination",
        task_family="object-presence",
        description="Is the named object present in the image? The standard VLM hallucination probe.",
        caps={"train": 0, "validation": 500, "test": 2000},
        notes="Evaluation only. `category` (random / popular / adversarial) is kept in meta so "
              "hallucination can be read per difficulty. POPE ships one split; a tenth of it is "
              "carved off as validation so the Noul recipe can be fitted without touching test.",
    )
    label_column = "answer"
    split_map = {"test": "test"}
    carve_validation_from = "test"
    carve_frac = 0.1

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        ans = str(row["answer"]).strip().lower()
        if ans not in ("yes", "no"):
            return None
        return self.record(split, idx, state={"image": row["image"], "question": row["question"]},
                           question={"type": "noul",
                                     "instructions": "Based on `image`, is the answer to `question` yes?"},
                           label=int(ans == "yes"), category=row.get("category"),
                           image_source=row.get("image_source"))


class AOKVQA(HFAdapter):
    """A-OKVQA: multiple-choice visual questions needing outside knowledge."""
    spec = SourceSpec(
        name="aokvqa", hf_id="HuggingFaceM4/A-OKVQA", primitive="choice", license="apache-2.0",
        domain="vision-knowledge", task_family="mcq", k=4,
        description="Answer a visual question that also needs world knowledge, from four options.",
        caps={"train": 2000, "validation": 500, "test": 1000},
    )
    label_column = None
    split_map = {"train": "train", "validation": "validation", "test": "validation"}
    carve_validation_from = "validation"
    carve_frac = 0.35

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        choices = list(row["choices"])
        gold = int(row["correct_choice_idx"])
        if not choices or not 0 <= gold < len(choices):
            return None
        letters = ["A", "B", "C", "D", "E"][: len(choices)]
        return self.record(split, idx, state={"image": row["image"], "question": row["question"]},
                           question={"type": "choice",
                                     "instructions": "Which option answers `question` about `image`?",
                                     "criteria": dict(zip(letters, choices))},
                           label=letters[gold], question_id=row.get("question_id"))


class AI2D(HFAdapter):
    """AI2D: grade-school science *diagram* questions — the image is a figure, not a photo."""
    spec = SourceSpec(
        name="ai2d", hf_id="lmms-lab/ai2d", hf_config="default", primitive="choice",
        license="cc-by-sa-4.0 (AI2D)", domain="vision-diagram", task_family="mcq", k=4,
        description="Answer a question about a science diagram from its options.",
        caps={"train": 0, "validation": 500, "test": 1500},
    )
    label_column = "answer"
    split_map = {"test": "test"}
    carve_validation_from = "test"
    carve_frac = 0.25

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        options = list(row["options"])
        try:
            gold = int(row["answer"])
        except (TypeError, ValueError):
            return None
        if not 0 <= gold < len(options):
            return None
        letters = ["A", "B", "C", "D", "E"][: len(options)]
        return self.record(split, idx, state={"image": row["image"], "question": row["question"]},
                           question={"type": "choice",
                                     "instructions": "Which option answers `question` about the diagram in `image`?",
                                     "criteria": dict(zip(letters, options))},
                           label=letters[gold])


ADAPTERS = [POPE, AOKVQA, AI2D]
VISION_SOURCES = [a.spec.name for a in ADAPTERS]
