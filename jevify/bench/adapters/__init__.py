"""Registry of jev-bench sources."""
from __future__ import annotations

from . import chaosnli, choice, noul, score
from ._base import Adapter, SourceSpec

ALL_ADAPTERS: list[type[Adapter]] = [*choice.ADAPTERS, *score.ADAPTERS, *noul.ADAPTERS, *chaosnli.ADAPTERS]

REGISTRY: dict[str, type[Adapter]] = {a.spec.name: a for a in ALL_ADAPTERS}


def specs() -> list[SourceSpec]:
    return [a.spec for a in ALL_ADAPTERS]


__all__ = ["ALL_ADAPTERS", "REGISTRY", "Adapter", "SourceSpec", "specs"]
