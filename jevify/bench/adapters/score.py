"""Score sources: place the state on ordered, described levels."""
from __future__ import annotations

import random
from typing import Any, Sequence

from ..record import BenchRecord, Split
from ._base import DEFAULT_SEED, Adapter, HFAdapter, SourceSpec, hf_dataset

SENTIMENT_5 = [
    "Very negative: strongly unfavorable, hostile or dismissive.",
    "Negative: unfavorable overall, with some criticism.",
    "Neutral: mixed or factual, no clear lean.",
    "Positive: favorable overall, with some praise.",
    "Very positive: enthusiastic, strongly favorable.",
]

STARS_5 = [
    "1 star: terrible experience, would not return.",
    "2 stars: poor, more negatives than positives.",
    "3 stars: average, mixed positives and negatives.",
    "4 stars: good, minor complaints at most.",
    "5 stars: excellent, glowing recommendation.",
]

HELPFULNESS_5 = [   # verbatim from the HelpSteer2 paper, Appendix G.3.1
    "Not useful or helpful at all; completely missed the essence of what the user wanted.",
    "Borderline unhelpful: mostly does not capture what the user was looking for, but still usable and helpful in a small way.",
    "Partially helpful: misses the overall goal of the user's query in some way; did not fully satisfy what the user was looking for.",
    "Mostly helpful and mainly aligned with what the user was looking for, but there is still some room for improvement.",
    "Extremely helpful and completely aligned with the spirit of what the prompt was asking for.",
]

VERBOSITY_5 = [   # verbatim from the HelpSteer2 paper, Appendix G.3.1: a LENGTH scale relative to the prompt, not a defect scale
    "Succinct: short, to the point, and the most concise it can be; no additional information beyond what was requested.",
    "Pretty short: on the shorter side, but could still have words, details or text removed before it is at the bare minimum.",
    "Average length: not especially long or short given what the prompt asks; adequate for a full response, neither wordy nor particularly concise.",
    "Moderately long: on the longer side, but could still have more added before it is fully detailed or rambling.",
    "Verbose: particularly lengthy, wordy and/or extensive with extra details given what the prompt requested, whether from repetition or from rich detail.",
]

STS_6 = [
    "Completely dissimilar: the two sentences are about different things.",
    "Not equivalent, but on the same topic.",
    "Not equivalent, but share some details.",
    "Roughly equivalent, but some important information differs or is missing.",
    "Mostly equivalent, but some unimportant details differ.",
    "Completely equivalent: they mean the same thing.",
]

HATE_3 = [
    "No hate speech: the comment does not attack or demean a group.",
    "Unclear or borderline: offensive or hostile, but not clearly hate speech.",
    "Hate speech: attacks, dehumanizes or incites against a group.",
]


class SST5(HFAdapter):
    spec = SourceSpec(
        name="sst5", hf_id="SetFit/sst5", primitive="score", license="unspecified (Stanford Sentiment Treebank)",
        domain="reviews", task_family="sentiment", k=5,
        description="Rate the sentiment of a movie-review sentence on five levels.",
    )
    label_column = "label"

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        return self.record(split, idx, state=row["text"],
                           question={"type": "score", "instructions": "What is the sentiment of this movie review sentence?",
                                     "criteria": SENTIMENT_5},
                           label=int(row["label"]))


class Yelp5(HFAdapter):
    spec = SourceSpec(
        name="yelp5", hf_id="Yelp/yelp_review_full", hf_config="yelp_review_full", primitive="score",
        license="other (Yelp Dataset License, research use)", domain="reviews", task_family="rating", k=5,
        description="Predict the star rating (1–5) a Yelp reviewer gave from the review text.",
    )
    label_column = "label"
    split_map = {"train": "train", "test": "test"}
    carve_validation_from = "train"
    carve_frac = 0.01  # train is 650k rows; 1% is plenty for validation

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        text = row["text"]
        if len(text) > 4000:
            return None
        return self.record(split, idx, state=text,
                           question={"type": "score", "instructions": "How many stars did the reviewer give?",
                                     "criteria": STARS_5},
                           label=int(row["label"]))


class _HelpSteer2(HFAdapter):
    attribute: str
    levels: list[str]
    instructions: str
    split_map = {"train": "train", "test": "validation"}
    carve_validation_from = "train"
    carve_frac = 0.05

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        if len(row["prompt"]) + len(row["response"]) > 12000:
            return None
        return self.record(split, idx, state={"prompt": row["prompt"], "response": row["response"]},
                           question={"type": "score", "instructions": self.instructions, "criteria": self.levels},
                           label=int(row[self.attribute]))


class HelpSteer2Helpfulness(_HelpSteer2):
    spec = SourceSpec(
        name="helpsteer2_helpfulness", hf_id="nvidia/HelpSteer2", primitive="score", license="cc-by-4.0",
        domain="llm-judging", task_family="response-quality", k=5,
        description="Rate how helpful an assistant response is to the user's prompt (HelpSteer2 helpfulness, 0–4).",
    )
    label_column = "helpfulness"
    attribute = "helpfulness"
    levels = HELPFULNESS_5
    instructions = "How helpful is `response` as an answer to `prompt`?"


class HelpSteer2Verbosity(_HelpSteer2):
    spec = SourceSpec(
        name="helpsteer2_verbosity", hf_id="nvidia/HelpSteer2", primitive="score", license="cc-by-4.0",
        domain="llm-judging", task_family="response-quality", k=5,
        description="Rate the length of an assistant response relative to what the prompt asked for (HelpSteer2 verbosity, 0 succinct – 4 verbose).",
        notes="v0.1.1: level descriptions replaced with the paper's verbatim scale; v0.1 wrongly framed 0/1 as 'too short' and 2 as 'appropriate'.",
    )
    label_column = "verbosity"
    attribute = "verbosity"
    levels = VERBOSITY_5
    instructions = "How long is `response` relative to what `prompt` asked for?"


class STSB(HFAdapter):
    spec = SourceSpec(
        name="stsb", hf_id="sentence-transformers/stsb", hf_config="default", primitive="score",
        license="cc-by-sa-4.0 (STS Benchmark)", domain="nli", task_family="semantic-similarity", k=6,
        description="Rate how similar in meaning two sentences are on the 0–5 STS scale.",
        caps={"train": 5749, "validation": 500, "test": 1000},
    )
    label_column = None

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        level = int(round(float(row["score"]) * 5))
        return self.record(split, idx, state={"sentence1": row["sentence1"], "sentence2": row["sentence2"]},
                           question={"type": "score", "instructions": "How similar in meaning are `sentence1` and `sentence2`?",
                                     "criteria": STS_6},
                           label=level, sts_score=float(row["score"]) * 5)


class MeasuringHateSpeech(Adapter):
    """UC Berkeley Measuring Hate Speech: several annotators per comment rate
    ``hatespeech`` in {0, 1, 2}. We aggregate per comment into a label
    distribution (calibration gold) and keep comments with >= 3 annotators."""
    spec = SourceSpec(
        name="measuring_hate_speech", hf_id="ucberkeley-dlab/measuring-hate-speech", hf_config="default", primitive="score",
        license="cc-by-4.0", domain="safety", task_family="hate-speech", k=3, has_soft_labels=True,
        description="Does the comment contain hate speech? Three levels with annotator distributions as soft labels.",
        caps={"train": 8000, "validation": 500, "test": 1000},
        notes="Aggregated from annotator-level rows; soft_label = annotator vote shares over the 3 levels.",
    )
    label_column = "label"

    def __init__(self, min_annotators: int = 3, seed: int = DEFAULT_SEED) -> None:
        self.min_annotators = min_annotators
        self.seed = seed
        self._splits: dict[str, list[dict[str, Any]]] | None = None

    def _aggregate(self) -> None:
        ds = hf_dataset(self.spec.hf_id, self.spec.hf_config, "train")
        df = ds.select_columns(["comment_id", "text", "hatespeech", "hate_speech_score"]).to_pandas()
        df = df.dropna(subset=["hatespeech"])
        df["hatespeech"] = df["hatespeech"].round().astype(int).clip(0, 2)
        rows: list[dict[str, Any]] = []
        for cid, g in df.groupby("comment_id", sort=True):
            n = len(g)
            if n < self.min_annotators:
                continue
            counts = [int((g["hatespeech"] == lvl).sum()) for lvl in range(3)]
            dist = [c / n for c in counts]
            rows.append({"comment_id": int(cid), "text": g["text"].iloc[0], "dist": dist,
                         "label": int(max(range(3), key=lambda i: (dist[i], -abs(i - 1)))),
                         "n_annotators": n, "hate_speech_score": float(g["hate_speech_score"].mean())})
        rng = random.Random(self.seed)
        rng.shuffle(rows)
        n_test = int(0.1 * len(rows))
        n_val = int(0.05 * len(rows))
        self._splits = {"test": rows[:n_test], "validation": rows[n_test:n_test + n_val], "train": rows[n_test + n_val:]}

    def load(self, split: Split) -> Sequence[dict[str, Any]]:
        if self._splits is None:
            self._aggregate()
        assert self._splits is not None
        return self._splits[split]

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        return self.record(split, idx, state=row["text"],
                           question={"type": "score", "instructions": "Does this comment contain hate speech?",
                                     "criteria": HATE_3},
                           label=row["label"], soft_label=row["dist"],
                           n_annotators=row["n_annotators"], hate_speech_score=row["hate_speech_score"],
                           comment_id=row["comment_id"])


ADAPTERS = [SST5, Yelp5, HelpSteer2Helpfulness, HelpSteer2Verbosity, STSB, MeasuringHateSpeech]
