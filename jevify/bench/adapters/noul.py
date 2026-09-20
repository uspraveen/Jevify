"""Noul sources: an absolute yes/no judgment, returned as P(yes)."""
from __future__ import annotations

import re
from typing import Any

from ..record import BenchRecord, Split
from ._base import HFAdapter, SingleSplitHFAdapter, SourceSpec

_WIKI_TOKENS = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-COLON-": ":", "-LCB-": "{", "-RCB-": "}"}
_WIKI_RE = re.compile("|".join(re.escape(k) for k in _WIKI_TOKENS))


def _clean_wiki(text: str) -> str:
    return _WIKI_RE.sub(lambda m: _WIKI_TOKENS[m.group(0)], text).replace(" ,", ",").replace(" .", ".")


class BoolQ(HFAdapter):
    spec = SourceSpec(
        name="boolq", hf_id="google/boolq", hf_config="default", primitive="noul", license="cc-by-sa-3.0",
        domain="reading-comprehension", task_family="grounded-yes-no",
        description="Given a Wikipedia passage, is the answer to the question yes?",
    )
    label_column = "answer"
    split_map = {"train": "train", "test": "validation"}
    carve_validation_from = "train"
    carve_frac = 0.05

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        return self.record(split, idx, state={"passage": row["passage"], "question": row["question"]},
                           question={"type": "noul",
                                     "instructions": "Based only on `passage`, is the answer to `question` yes?"},
                           label=int(bool(row["answer"])))


class FEVERGoldEvidence(HFAdapter):
    spec = SourceSpec(
        name="fever_evidence", hf_id="copenlu/fever_gold_evidence", primitive="noul", license="cc-by-sa-3.0",
        domain="fact-checking", task_family="grounded-yes-no",
        description="Given gold Wikipedia evidence sentences, is the claim supported (yes) or refuted (no)?",
        notes="NOT ENOUGH INFO rows are dropped so the question is a clean yes/no.",
    )
    label_column = "label"

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        if row["label"] not in ("SUPPORTS", "REFUTES"):
            return None
        evidence = [_clean_wiki(e[2]) for e in row["evidence"] if len(e) >= 3 and e[2]]
        if not evidence:
            return None
        return self.record(split, idx, state={"claim": row["claim"], "evidence": evidence},
                           question={"type": "noul",
                                     "instructions": "Does `evidence` support `claim`?",
                                     "criteria": {"true": "The evidence establishes the claim is true.",
                                                  "false": "The evidence shows the claim is false."}},
                           label=int(row["label"] == "SUPPORTS"), fever_id=row.get("original_id"))


class PAWS(HFAdapter):
    spec = SourceSpec(
        name="paws", hf_id="google-research-datasets/paws", hf_config="labeled_final", primitive="noul",
        license="other (Google PAWS, free for research and commercial use)", domain="nli", task_family="paraphrase",
        description="Do two sentences with high lexical overlap actually mean the same thing?",
    )
    label_column = "label"

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        return self.record(split, idx, state={"sentence1": row["sentence1"], "sentence2": row["sentence2"]},
                           question={"type": "noul",
                                     "instructions": "Do `sentence1` and `sentence2` have the same meaning?"},
                           label=int(row["label"]), paws_id=row["id"])


class CivilCommentsToxicity(HFAdapter):
    """Jigsaw Civil Comments: ``toxicity`` is the fraction of annotators who
    flagged the comment, which makes it a calibration-gold soft label."""
    spec = SourceSpec(
        name="civil_comments", hf_id="google/civil_comments", hf_config="default", primitive="noul", license="cc0-1.0",
        domain="safety", task_family="toxicity", has_soft_labels=True,
        description="Is this online comment toxic? Soft label = share of annotators who said yes.",
        caps={"train": 8000, "validation": 500, "test": 2000},
        notes="Natural class balance (~8% toxic); test cap raised to 2000 so positives are not too thin.",
    )
    label_column = None  # keep the natural distribution

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        text = row["text"]
        if not text or len(text) > 3000:
            return None
        tox = float(row["toxicity"])
        return self.record(split, idx, state=text,
                           question={"type": "noul",
                                     "instructions": "Is this comment toxic?",
                                     "criteria": {"true": "Rude, disrespectful, or likely to make someone leave the discussion.",
                                                  "false": "Civil, even if critical or blunt."}},
                           label=int(tox >= 0.5), soft_label=tox, severe_toxicity=float(row["severe_toxicity"]))


class SMSSpam(SingleSplitHFAdapter):
    spec = SourceSpec(
        name="sms_spam", hf_id="ucirvine/sms_spam", hf_config="plain_text", primitive="noul",
        license="unknown (UCI SMS Spam Collection, public)", domain="messaging", task_family="spam",
        description="Is this SMS message spam?",
        caps={"train": 4000, "validation": 300, "test": 800},
    )
    label_column = "label"

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        return self.record(split, idx, state=row["sms"],
                           question={"type": "noul", "instructions": "Is this SMS message spam?"},
                           label=int(row["label"]))


class _StrategyQA(HFAdapter):
    split_map = {"train": "train", "test": "test"}
    carve_validation_from = "train"
    carve_frac = 0.1
    grounded: bool

    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        if self.grounded:
            state: Any = {"facts": row["facts"], "question": row["question"]}
            instructions = "Given `facts`, is the answer to `question` yes?"
        else:
            state = {"question": row["question"]}
            instructions = "Is the answer to `question` yes?"
        return self.record(split, idx, state=state,
                           question={"type": "noul", "instructions": instructions},
                           label=int(bool(row["answer"])), qid=row["qid"], term=row["term"])


class StrategyQAClosed(_StrategyQA):
    """Question only: needs world knowledge plus implicit multi-step reasoning. A
    deliberate 'System Two' probe (Jev's jaggedness page predicts weakness)."""
    spec = SourceSpec(
        name="strategyqa_closed", hf_id="ChilleD/StrategyQA", primitive="noul", license="mit",
        domain="knowledge", task_family="implicit-reasoning",
        description="Answer an implicit multi-hop yes/no question from world knowledge alone.",
        caps={"train": 1400, "validation": 160, "test": 687},
    )
    label_column = "answer"
    grounded = False


class StrategyQAGrounded(_StrategyQA):
    spec = SourceSpec(
        name="strategyqa_grounded", hf_id="ChilleD/StrategyQA", primitive="noul", license="mit",
        domain="knowledge", task_family="grounded-yes-no",
        description="Same questions with the supporting facts supplied in the state.",
        caps={"train": 1400, "validation": 160, "test": 687},
    )
    label_column = "answer"
    grounded = True


ADAPTERS = [BoolQ, FEVERGoldEvidence, PAWS, CivilCommentsToxicity, SMSSpam, StrategyQAClosed, StrategyQAGrounded]
