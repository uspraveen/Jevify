"""Registry of jev-bench sources."""
from __future__ import annotations

from . import chaosnli, choice, noul, score, vision
from ._base import Adapter, SourceSpec

ALL_ADAPTERS: list[type[Adapter]] = [*choice.ADAPTERS, *score.ADAPTERS, *noul.ADAPTERS, *chaosnli.ADAPTERS]
VISION_ADAPTERS: list[type[Adapter]] = list(vision.ADAPTERS)

REGISTRY: dict[str, type[Adapter]] = {a.spec.name: a for a in ALL_ADAPTERS}
VISION_REGISTRY: dict[str, type[Adapter]] = {a.spec.name: a for a in VISION_ADAPTERS}


def specs() -> list[SourceSpec]:
    return [a.spec for a in ALL_ADAPTERS]


def vision_specs() -> list[SourceSpec]:
    return [a.spec for a in VISION_ADAPTERS]


__all__ = ["ALL_ADAPTERS", "VISION_ADAPTERS", "REGISTRY", "VISION_REGISTRY", "Adapter", "SourceSpec",
           "specs", "vision_specs"]
