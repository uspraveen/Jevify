"""Choice sources: pick one option from a labeled set."""
from __future__ import annotations

from typing import Any

from ..record import BenchRecord, Split
from ._base import DEFAULT_SEED, Adapter, HFAdapter, SourceSpec, hf_dataset, humanize

NLI_CRITERIA = {
    "entailment": "The hypothesis is definitely true given the premise.",
    "neutral": "The hypothesis might be true; the premise does not settle it.",
    "contradiction": "The hypothesis is definitely false given the premise.",
}


class Banking77(HFAdapter):
    """77 fine-grained banking support intents. Large-K routing with short messages."""
    spec = SourceSpec(
        name="banking77", hf_id="mteb/banking77", primitive="choice", license="cc-by-4.0 (PolyAI; mirror: mit)",
        domain="customer-support", task_family="intent-routing", k=77,
        description="Route a customer's banking message to one of 77 intents.",
    )
    label_column = "label_text"
    split_map = {"train": "train", "test": "test"}
    carve_validation_from = "train"

    def __init__(self) -> None:
        super().__init__()
        self._criteria: dict[str, str] | None = None

    def criteria(self, rows) -> dict[str, str]:
        if self._criteria is None:
            names = sorted(set(rows["label_text"]))
            self._criteria = {n: humanize(n) for n in names}
        return self._criteria

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        criteria = self.criteria(self.load("train"))
        return self.record(split, idx, state=row["text"],
                           question={"type": "choice",
                                     "instructions": "Which banking support intent does the customer's message express?",
                                     "criteria": criteria},
                           label=row["label_text"])


class Clinc150(HFAdapter):
    """150 intents across 10 domains plus an explicit out-of-scope option (Jev's recommended 'other')."""
    spec = SourceSpec(
        name="clinc150", hf_id="clinc/clinc_oos", hf_config="plus", primitive="choice", license="cc-by-3.0",
        domain="virtual-assistant", task_family="intent-routing", k=151,
        description="Classify an assistant query into one of 150 intents or 'oos' (out of scope).",
    )
    label_column = "intent"

    def _names(self) -> list[str]:
        return self._hf_split("train").features["intent"].names

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        names = self._names()
        criteria = {n: ("None of the other intents apply (out of scope)" if n == "oos" else humanize(n)) for n in names}
        label = names[row["intent"]]
        return self.record(split, idx, state=row["text"],
                           question={"type": "choice",
                                     "instructions": "Which intent does the user's query express? Pick 'oos' if none of the listed intents apply.",
                                     "criteria": criteria},
                           label=label, is_oos=(label == "oos"))


class Massive(HFAdapter):
    """Amazon MASSIVE (en-US): 60 assistant intents over 18 scenarios."""
    spec = SourceSpec(
        name="massive", hf_id="mteb/amazon_massive_intent", hf_config="en", primitive="choice", license="cc-by-4.0 (Amazon; mirror: apache-2.0)",
        domain="virtual-assistant", task_family="intent-routing", k=60,
        description="Classify a smart-assistant utterance into one of 60 intents.",
    )
    label_column = "label_text"

    def __init__(self) -> None:
        super().__init__()
        self._criteria: dict[str, str] | None = None

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        if self._criteria is None:
            self._criteria = {n: humanize(n) for n in sorted(set(self._hf_split("train")["label_text"]))}
        return self.record(split, idx, state=row["text"],
                           question={"type": "choice",
                                     "instructions": "Which intent does the user's request to a voice assistant express?",
                                     "criteria": self._criteria},
                           label=row["label_text"])


class Ledgar(HFAdapter):
    """LEDGAR (LexGLUE): contract provisions into 100 categories. Large-K, long-ish legal text."""
    spec = SourceSpec(
        name="ledgar", hf_id="coastalcph/lex_glue", hf_config="ledgar", primitive="choice", license="cc-by-4.0",
        domain="legal", task_family="topic-classification", k=100,
        description="Classify a contract clause into one of 100 provision categories.",
    )
    label_column = "label"

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        names = self._hf_split("train").features["label"].names
        criteria = {n: None for n in names}
        return self.record(split, idx, state=row["text"],
                           question={"type": "choice",
                                     "instructions": "Which contract provision category does this clause belong to?",
                                     "criteria": criteria},
                           label=names[row["label"]])


EMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
            "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
            "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]


class GoEmotions(Adapter):
    """GoEmotions rebuilt from the *raw* per-rater annotations (3–5 raters per
    comment). ``soft_label`` = each emotion's share of all rater votes, so the
    benchmark carries the disagreement instead of hiding it behind one label.
    The hard label is the plurality emotion; ties go to the more frequent
    emotion overall. Comments any rater marked 'very unclear' are dropped."""
    spec = SourceSpec(
        name="go_emotions", hf_id="google-research-datasets/go_emotions", hf_config="raw", primitive="choice", license="apache-2.0",
        domain="social", task_family="emotion", k=28, has_soft_labels=True,
        description="Which emotion does a Reddit comment primarily express (27 emotions + neutral)? soft_label = rater vote shares.",
        caps={"train": 8000, "validation": 500, "test": 1000},
        notes="v0.1.1: rebuilt from the raw config with rater vote shares as soft labels (v0.1 used the single-label 'simplified' subset).",
    )
    label_column = "label"

    def __init__(self, min_raters: int = 3, seed: int = DEFAULT_SEED) -> None:
        self.min_raters = min_raters
        self.seed = seed
        self._splits: dict[str, list[dict[str, Any]]] | None = None

    def _aggregate(self) -> None:
        import random

        ds = hf_dataset(self.spec.hf_id, self.spec.hf_config, "train")
        df = ds.select_columns(["id", "text", "rater_id", "example_very_unclear", *EMOTIONS]).to_pandas()
        rows: list[dict[str, Any]] = []
        overall = df[EMOTIONS].sum()
        for cid, g in df.groupby("id", sort=True):
            if len(g) < self.min_raters or bool(g["example_very_unclear"].any()):
                continue
            votes = g[EMOTIONS].sum()
            total = float(votes.sum())
            if total == 0:
                continue
            dist = {e: float(votes[e]) / total for e in EMOTIONS}
            top = max(EMOTIONS, key=lambda e: (votes[e], overall[e]))
            rows.append({"id": cid, "text": g["text"].iloc[0], "dist": dist, "label": top,
                         "n_raters": int(len(g)), "agreement": float(votes[top]) / len(g)})
        rng = random.Random(self.seed)
        rng.shuffle(rows)
        n_test, n_val = int(0.1 * len(rows)), int(0.05 * len(rows))
        self._splits = {"test": rows[:n_test], "validation": rows[n_test:n_test + n_val], "train": rows[n_test + n_val:]}

    def load(self, split: Split):
        if self._splits is None:
            self._aggregate()
        assert self._splits is not None
        return self._splits[split]

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        return self.record(split, idx, state=row["text"],
                           question={"type": "choice",
                                     "instructions": "Which emotion does the comment primarily express?",
                                     "criteria": {e: None for e in EMOTIONS}},
                           label=row["label"], soft_label=row["dist"],
                           reddit_id=row["id"], n_raters=row["n_raters"], rater_agreement=row["agreement"])


class MMLU(HFAdapter):
    """MMLU: 4-option knowledge questions across 57 subjects. Measures knowledge retention through tiers."""
    spec = SourceSpec(
        name="mmlu", hf_id="cais/mmlu", hf_config="all", primitive="choice", license="mit",
        domain="knowledge", task_family="mcq", k=4,
        description="Answer a multiple-choice exam question (A/B/C/D) across 57 academic subjects.",
        caps={"train": 285, "validation": 500, "test": 1000},
        notes="train = MMLU 'dev' (5-shot examples); no auxiliary_train.",
    )
    label_column = "subject"
    split_map = {"train": "dev", "validation": "validation", "test": "test"}

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        letters = ["A", "B", "C", "D"]
        criteria = {letters[i]: text for i, text in enumerate(row["choices"])}
        return self.record(split, idx,
                           state={"subject": humanize(row["subject"]), "question": row["question"]},
                           question={"type": "choice",
                                     "instructions": "Which option is the correct answer to `question`?",
                                     "criteria": criteria},
                           label=letters[row["answer"]], subject=row["subject"])


class ARCChallenge(HFAdapter):
    """ARC-Challenge: grade-school science questions with 3–5 options."""
    spec = SourceSpec(
        name="arc_challenge", hf_id="allenai/ai2_arc", hf_config="ARC-Challenge", primitive="choice", license="cc-by-sa-4.0",
        domain="knowledge", task_family="mcq",
        description="Answer a science exam question from its lettered options.",
    )
    label_column = "answerKey"

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        if row["answerKey"] not in labels:
            return None
        return self.record(split, idx, state={"question": row["question"]},
                           question={"type": "choice",
                                     "instructions": "Which option is the correct answer to `question`?",
                                     "criteria": dict(zip(labels, texts))},
                           label=row["answerKey"], arc_id=row["id"])


class MNLI(HFAdapter):
    """MultiNLI as a 3-way Choice. test = validation_matched, validation = validation_mismatched."""
    spec = SourceSpec(
        name="mnli", hf_id="nyu-mll/glue", hf_config="mnli", primitive="choice", license="other (GLUE/MultiNLI, research use)",
        domain="nli", task_family="nli", k=3,
        description="Does the hypothesis follow from, contradict, or remain undetermined by the premise?",
    )
    label_column = "label"
    split_map = {"train": "train", "validation": "validation_mismatched", "test": "validation_matched"}

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        names = ["entailment", "neutral", "contradiction"]
        if row["label"] < 0:
            return None
        return self.record(split, idx, state={"premise": row["premise"], "hypothesis": row["hypothesis"]},
                           question={"type": "choice",
                                     "instructions": "What is the relationship of `hypothesis` to `premise`?",
                                     "criteria": NLI_CRITERIA},
                           label=names[row["label"]], glue_idx=row["idx"])


ADAPTERS = [Banking77, Clinc150, Massive, Ledgar, GoEmotions, MMLU, ARCChallenge, MNLI]
