"""Shared machinery for turning a Hugging Face dataset into BenchRecords.

An adapter declares a ``SourceSpec`` (where the data comes from, its license,
which primitive it maps to) and implements two things: ``load(split)`` which
returns the rows for a bench split, and ``convert(row, split, idx)`` which
turns one row into a ``BenchRecord`` (or ``None`` to skip it).

Sampling is the builder's job (see ``build.py``): it asks the adapter for the
label of each row via ``label_column`` so it can sample stratified when the
label space is small, and uniformly otherwise.
"""
from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..record import BenchRecord, Primitive, Split

DEFAULT_SEED = 20260920


@dataclass(frozen=True)
class SourceSpec:
    name: str                     # bench config name, e.g. "banking77"
    hf_id: str                    # Hugging Face dataset id ("" for non-HF sources)
    primitive: Primitive
    license: str                  # SPDX-ish id or a short note, copied into the manifest
    domain: str                   # support / legal / knowledge / safety / reviews / nli / ...
    task_family: str              # intent-routing / topic / mcq / sentiment / toxicity / ...
    description: str
    hf_config: str | None = None
    k: int | None = None          # number of options / levels when fixed
    has_soft_labels: bool = False
    caps: dict[str, int] = field(default_factory=lambda: {"train": 8000, "validation": 500, "test": 1000})
    notes: str = ""


class Adapter(ABC):
    spec: SourceSpec
    label_column: str | None = None   # column to stratify on, if any

    @abstractmethod
    def load(self, split: Split) -> Sequence[dict[str, Any]]:
        """Rows for a bench split. Must be indexable (a ``datasets.Dataset`` is fine)."""

    @abstractmethod
    def convert(self, row: dict[str, Any], split: Split, idx: int) -> BenchRecord | None:
        ...

    # ---- helpers for subclasses ------------------------------------------------------
    def record(self, split: Split, idx: int, *, state: Any, question: dict[str, Any],
               label: str | int, soft_label: Any = None, **meta: Any) -> BenchRecord:
        base_meta = {"hf_id": self.spec.hf_id, "license": self.spec.license,
                     "domain": self.spec.domain, "task_family": self.spec.task_family}
        if self.spec.hf_config:
            base_meta["hf_config"] = self.spec.hf_config
        base_meta.update(meta)
        return BenchRecord(
            id=f"{self.spec.name}/{split}/{idx}",
            source=self.spec.name,
            primitive=self.spec.primitive,
            split=split,
            state=state,
            question=question,
            label=label,
            soft_label=soft_label,
            meta=base_meta,
        )


# --------------------------------------------------------------------------- HF loading

def hf_dataset(hf_id: str, config: str | None, split: str):
    from datasets import load_dataset  # imported lazily: only the builder needs it
    return load_dataset(hf_id, config, split=split)


def carve(ds, *, frac: float, seed: int = DEFAULT_SEED):
    """Split an HF Dataset into (rest, held) deterministically. Used to make a
    validation split for sources that ship only train/test."""
    parts = ds.train_test_split(test_size=frac, seed=seed, shuffle=True)
    return parts["train"], parts["test"]


class HFAdapter(Adapter):
    """Adapter whose bench splits map onto HF splits, with an optional carve.

    ``split_map`` maps bench split -> HF split name. If ``carve_validation_from``
    is set (an HF split name), a ``carve_frac`` slice of that split is held out
    as the bench ``validation`` split and removed from the bench split that
    uses it.
    """
    split_map: dict[str, str] = {"train": "train", "validation": "validation", "test": "test"}
    carve_validation_from: str | None = None
    carve_frac: float = 0.1
    _cache: dict[str, Any]

    def __init__(self) -> None:
        self._cache = {}

    def _hf_split(self, name: str):
        if name not in self._cache:
            self._cache[name] = hf_dataset(self.spec.hf_id, self.spec.hf_config, name)
        return self._cache[name]

    def load(self, split: Split):
        if self.carve_validation_from:
            src = self.carve_validation_from
            key = f"__carved__{src}"
            if key not in self._cache:
                self._cache[key] = carve(self._hf_split(src), frac=self.carve_frac)
            rest, held = self._cache[key]
            if split == "validation":
                return held
            if self.split_map.get(split) == src:
                return rest
        hf_split = self.split_map.get(split)
        if hf_split is None:
            return []
        return self._hf_split(hf_split)


class SingleSplitHFAdapter(HFAdapter):
    """For sources that ship one split only: hold out ``test_frac`` as test, then
    ``validation_frac`` of the remainder as validation; the rest is train."""
    source_split: str = "train"
    test_frac: float = 0.15
    validation_frac: float = 0.05

    def load(self, split: Split):
        key = "__single_split__"
        if key not in self._cache:
            rest, test = carve(self._hf_split(self.source_split), frac=self.test_frac)
            train, val = carve(rest, frac=self.validation_frac, seed=DEFAULT_SEED + 1)
            self._cache[key] = {"train": train, "validation": val, "test": test}
        return self._cache[key][split]


# --------------------------------------------------------------------------- text helpers

_WORD = re.compile(r"[_\-]+")


def humanize(label: str) -> str:
    """'card_arrival' -> 'Card arrival'; 'transfer_money' -> 'Transfer money'."""
    s = _WORD.sub(" ", str(label)).strip()
    return s[:1].upper() + s[1:] if s else s


def stratified_order(labels: Sequence[Any], seed: int) -> list[int]:
    """Indices ordered so that taking any prefix gives a near-balanced sample:
    round-robin over label groups, each group shuffled."""
    rng = random.Random(seed)
    groups: dict[Any, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)
    for g in groups.values():
        rng.shuffle(g)
    keys = list(groups)
    rng.shuffle(keys)
    out: list[int] = []
    pos = 0
    while len(out) < len(labels):
        progressed = False
        for k in keys:
            g = groups[k]
            if pos < len(g):
                out.append(g[pos])
                progressed = True
        if not progressed:
            break
        pos += 1
    return out


def uniform_order(n: int, seed: int) -> list[int]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx
