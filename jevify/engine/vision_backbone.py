"""Direct control over the vision half of a VLM.

Jevifying a text model leaves one structural choice: which layer to read. A VLM has a
second, larger one. Before any decision head sees anything, a vision tower has already
decided what survives — at what resolution, into how many tokens, through a projector
trained for one particular decoder. That stage is usually an opaque prefix. Here it is
something you can inspect, budget, freeze and adapt on its own terms.

Three capabilities, each of which works across VLM families rather than for one model:

``describe``      what the tower actually is — class, depth, parameter share, and the
                  projector that bridges it to the decoder.
``pixel_budget``  how much image survives preprocessing. This is the honest knob for
                  visual signal: it changes the number of image tokens the decoder sees,
                  which is measurable (``image_tokens``) rather than nominal.
``lora_pattern``  a regex that confines LoRA to the vision tower, the decoder, or both.
                  Suffix targeting cannot do this: ``q_proj`` exists on both sides of a
                  VLM, so a plain suffix list silently adapts the whole model when you
                  asked for the encoder.

What this module deliberately does not do is swap one vision tower for another. The
projector is trained against a specific encoder's output geometry and semantics; a
different encoder produces embeddings the decoder was never aligned to, so the swap
"works" mechanically and destroys the model. Doing it honestly means retraining the
projector, which is a different (and much larger) job than Jevification.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

# Names VLM families use for the two stages. Matched against module paths, longest first,
# so a nested `model.vision_tower.vision_model` resolves to the outermost real tower.
TOWER_HINTS = ("vision_tower", "vision_model", "visual", "image_encoder", "image_tower", "vit")
PROJECTOR_HINTS = ("multi_modal_projector", "mm_projector", "merger", "connector",
                   "vision_projection", "image_projection", "projector")

# Linear names worth adapting inside a ViT-style encoder (families differ in which exist).
VISION_LINEARS = ("q_proj", "k_proj", "v_proj", "out_proj", "o_proj", "qkv", "proj", "fc1", "fc2")
DECODER_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _named_modules(model) -> list[tuple[str, Any]]:
    return list(model.named_modules())


def _find(model, hints: Sequence[str]) -> tuple[str, Any] | tuple[None, None]:
    """Shallowest module whose own name matches a hint, preferring bigger candidates."""
    best: tuple[int, int, str, Any] | None = None
    for name, mod in _named_modules(model):
        if not name:
            continue
        leaf = name.split(".")[-1]
        if not any(h == leaf or h in leaf for h in hints):
            continue
        depth = name.count(".")
        size = sum(p.numel() for p in mod.parameters())
        if size == 0:
            continue
        cand = (-depth, size, name, mod)
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    return (best[2], best[3]) if best else (None, None)


def find_vision_tower(model) -> tuple[str | None, Any]:
    """The image encoder, or (None, None) for a text-only model."""
    return _find(model, TOWER_HINTS)


def find_projector(model) -> tuple[str | None, Any]:
    """The module mapping encoder output into the decoder's embedding space."""
    return _find(model, PROJECTOR_HINTS)


def _count(module) -> int:
    return sum(p.numel() for p in module.parameters()) if module is not None else 0


def _depth(module) -> int:
    """Number of blocks in the module's longest ModuleList — a ViT's layer count."""
    import torch.nn as nn

    best = 0
    if module is None:
        return 0
    for _n, m in module.named_modules():
        if isinstance(m, nn.ModuleList):
            best = max(best, len(m))
    return best


def describe(model) -> dict[str, Any]:
    """An inventory of the vision stage: what it is and how much of the model it is."""
    t_name, tower = find_vision_tower(model)
    p_name, proj = find_projector(model)
    total = sum(p.numel() for p in model.parameters())
    n_tower, n_proj = _count(tower), _count(proj)
    info: dict[str, Any] = {
        "is_vlm": tower is not None,
        "total_params": total,
        "vision_tower": None if tower is None else {
            "path": t_name, "class": type(tower).__name__, "params": n_tower,
            "share_of_model": round(n_tower / total, 4) if total else 0.0, "layers": _depth(tower),
            "frozen": all(not p.requires_grad for p in tower.parameters()),
        },
        "projector": None if proj is None else {
            "path": p_name, "class": type(proj).__name__, "params": n_proj,
            "share_of_model": round(n_proj / total, 4) if total else 0.0,
        },
        "decoder_params": total - n_tower - n_proj,
    }
    cfg = getattr(model, "config", None)
    vcfg = getattr(cfg, "vision_config", None) if cfg is not None else None
    if vcfg is not None:
        info["vision_config"] = {k: getattr(vcfg, k) for k in
                                 ("hidden_size", "image_size", "patch_size", "num_hidden_layers",
                                  "num_attention_heads", "spatial_merge_size")
                                 if getattr(vcfg, k, None) is not None}
    return info


# --------------------------------------------------------------------------- freezing
def set_trainable(module, trainable: bool) -> int:
    """Flip requires_grad across a module; returns how many parameters changed."""
    n = 0
    if module is None:
        return 0
    for p in module.parameters():
        if p.requires_grad != trainable:
            p.requires_grad_(trainable)
            n += p.numel()
    return n


def freeze_vision(model, frozen: bool = True) -> int:
    tower = find_vision_tower(model)[1]
    return set_trainable(tower, not frozen)


# --------------------------------------------------------------------------- LoRA scoping
def lora_pattern(model, where: str = "vision", linears: Iterable[str] | None = None) -> str:
    """A regex for ``LoraConfig(target_modules=...)`` confined to one half of the model.

    peft matches a plain suffix list against the *end* of a module path, and ``q_proj``
    is a suffix on both the encoder and the decoder — so asking for the vision tower by
    suffix silently adapts everything. A regex over the full path is what actually scopes
    it, and peft accepts one in place of the list.
    """
    t_name = find_vision_tower(model)[0]
    if where == "vision" and not t_name:
        raise ValueError("this model has no vision tower to target")
    names = {n for n, _ in _named_modules(model)}

    def present(prefix: str | None, want: Iterable[str]) -> list[str]:
        out = []
        for leaf in want:
            if any(n.split(".")[-1] == leaf and (prefix is None or n.startswith(prefix + "."))
                   for n in names):
                out.append(leaf)
        return sorted(set(out))

    if where == "vision":
        leaves = present(t_name, linears or VISION_LINEARS)
        if not leaves:
            raise ValueError(f"no adaptable linear layers found under {t_name}")
        return rf"^{re.escape(t_name)}\..*\.({'|'.join(leaves)})$"
    if where == "decoder":
        leaves = present(None, linears or DECODER_LINEARS)
        body = f"({'|'.join(leaves)})"
        if t_name:                      # everything except the tower
            return rf"^(?!{re.escape(t_name)}\.).*\.{body}$"
        return rf"^.*\.{body}$"
    if where == "both":
        leaves = present(None, set(linears or ()) or set(VISION_LINEARS) | set(DECODER_LINEARS))
        return rf"^.*\.({'|'.join(leaves)})$"
    raise ValueError(f"where must be vision, decoder or both, not {where!r}")


# --------------------------------------------------------------------------- pixel budget
def _image_processor(processor):
    return getattr(processor, "image_processor", processor)


def pixel_budget(processor, max_pixels: int | None = None, min_pixels: int | None = None) -> dict[str, Any]:
    """Set how much image survives preprocessing; returns what was actually applied.

    Returning the applied values matters: processors silently ignore attributes they do
    not have, so a budget you *set* is not necessarily a budget that *took*.
    """
    ip = _image_processor(processor)
    applied: dict[str, Any] = {}
    for key, val in (("max_pixels", max_pixels), ("min_pixels", min_pixels)):
        if val is None:
            continue
        if hasattr(ip, key):
            setattr(ip, key, int(val))
            applied[key] = int(val)
        # newer processors nest the same budget under `size`
        size = getattr(ip, "size", None)
        if isinstance(size, dict) and key in size:
            size[key] = int(val)
            applied[key] = int(val)
    applied["effective"] = {k: getattr(ip, k, None) for k in ("max_pixels", "min_pixels")}
    if isinstance(getattr(ip, "size", None), dict):
        applied["size"] = dict(ip.size)
    return applied


def image_tokens(processor, image) -> int:
    """How many tokens this image becomes under the current budget.

    The measured cost of the current setting, rather than the nominal one: merging and
    padding mean the token count is not a simple function of pixels.
    """
    enc = _image_processor(processor)(images=[image], return_tensors="pt")
    grid = enc.get("image_grid_thw")
    if grid is not None:
        merge = getattr(_image_processor(processor), "merge_size", 1) or 1
        return int(grid.prod(dim=-1).sum()) // (merge * merge)
    values = enc.get("pixel_values")
    if values is None:
        return 0
    if values.ndim == 4:                       # (n, c, h, w) — patchified by the tower
        patch = getattr(_image_processor(processor), "patch_size", None) or 14
        return int(values.shape[-2] // patch) * int(values.shape[-1] // patch) * int(values.shape[0])
    return int(values.shape[0])
