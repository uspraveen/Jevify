"""Jevify for vision-language models: the same typed questions, about images.

A System One question over an image is the same object as one over text — a `state`, a
question, a fixed answer set — so nothing about the wire format changes. What changes is
that the state carries images, which have to reach the model through its processor rather
than its tokenizer.

Tier 0 needs only the logits at the answer position, so the whole readout is: render the
text with image placeholders, let the processor interleave the real image tokens, run one
forward pass, and gather the candidate tokens from the final position. Candidates are
single-token by the same identifier machinery the text engine uses, so no per-option slot
positions are needed — which matters, because image tokens expand and would break any
character-offset mapping into the prompt.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .template import CUE, Rendered, default_identifiers, render

IMAGE_KEYS = ("image", "images", "image_url", "photo", "picture", "figure", "diagram")


def _is_image(x: Any) -> bool:
    try:
        from PIL.Image import Image

        return isinstance(x, Image)
    except Exception:
        return False


def load_image(x: Any):
    """Accept a PIL image, a local path, an http(s) URL, a data URI, or raw bytes."""
    from PIL import Image

    if _is_image(x):
        return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")
    if isinstance(x, str):
        if x.startswith("data:"):
            return Image.open(io.BytesIO(base64.b64decode(x.split(",", 1)[1]))).convert("RGB")
        if x.startswith(("http://", "https://")):
            import urllib.request

            with urllib.request.urlopen(x, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        return Image.open(x).convert("RGB")
    raise TypeError(f"cannot read an image from {type(x).__name__}")


def split_images(state: Any) -> tuple[Any, list[Any]]:
    """Pull every image out of a state, leaving a placeholder in its place.

    Returns (text-only state, images in the order they appear). Placeholders read
    ``<image 1>`` so the model is told where each one sat in the structure.
    """
    images: list[Any] = []

    def walk(node: Any, key: str | None = None) -> Any:
        if _is_image(node) or (isinstance(node, (bytes, bytearray))):
            images.append(load_image(node))
            return f"<image {len(images)}>"
        if isinstance(node, str) and key and key.lower() in IMAGE_KEYS and _looks_like_image_ref(node):
            images.append(load_image(node))
            return f"<image {len(images)}>"
        if isinstance(node, dict):
            return {k: walk(v, k) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, key) for v in node]
        return node

    return walk(state), images


def _looks_like_image_ref(s: str) -> bool:
    return s.startswith(("http://", "https://", "data:")) or s.lower().endswith(
        (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))


@dataclass
class VisionItem:
    prefix: str
    candidates: list[str]
    images: list[Any]


class VisionScorer:
    """Teacher-forced candidate scoring for a VLM: one forward pass, logits at the cue."""

    def __init__(self, model_id: str, *, device: str | None = None, dtype: torch.dtype | None = None,
                 batch_size: int = 4, max_pixels: int | None = None, trust_remote_code: bool = False,
                 hf_token: str | None = None) -> None:
        from transformers import AutoProcessor

        # transformers renamed the vision-language auto class; accept either
        try:
            from transformers import AutoModelForImageTextToText as _AutoVLM
        except ImportError:                                  # pragma: no cover - older transformers
            from transformers import AutoModelForVision2Seq as _AutoVLM

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        kw: dict[str, Any] = {"trust_remote_code": trust_remote_code, "token": hf_token}
        if max_pixels:
            kw["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_id, **kw)
        self.tokenizer = getattr(self.processor, "tokenizer", None) or self.processor
        self.model = _AutoVLM.from_pretrained(
            model_id, dtype=self.dtype, trust_remote_code=trust_remote_code, token=hf_token).to(self.device).eval()
        self.batch_size = batch_size
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else (
            self.tokenizer.eos_token_id or 0)
        self._identifiers: list[str] | None = None

    # ------------------------------------------------------------------ identifiers
    def identifiers(self, k: int = 255) -> list[str]:
        """Single-token option identifiers for this processor's tokenizer."""
        if self._identifiers is None:
            import itertools
            import string

            probe = "Allowed answers: A, B" + chr(10) + CUE
            base = self._encode(probe)
            ok: list[str] = []
            for cand in itertools.chain(string.ascii_uppercase,
                                        ("".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2))):
                ids = self._encode(probe + cand)
                if len(ids) - _common(base, ids) == 1:
                    ok.append(cand)
                if len(ok) >= 255:
                    break
            self._identifiers = ok if len(ok) >= 26 else default_identifiers(255)
        return self._identifiers[:k]

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    # ------------------------------------------------------------------ prompting
    def build_prompt(self, rendered: Rendered, n_images: int) -> str:
        """Wrap the rendered text in the model's chat template with image slots."""
        user, _cue = rendered.prefix.rsplit(chr(10) + CUE, 1)
        content: list[dict[str, Any]] = [{"type": "image"} for _ in range(n_images)]
        content.append({"type": "text", "text": user})
        messages = [{"role": "user", "content": content}]
        try:
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception:                                   # processors without a chat template
            text = ("<image>" * n_images) + chr(10) + user + chr(10)
        return text + CUE

    def item(self, state: Any, question: dict[str, Any], *, mode: str = "index") -> tuple[VisionItem, Rendered]:
        """Build the scoring item and return it with the rendering (for the answer keys)."""
        text_state, images = split_images(state)
        rd = render(text_state, question, mode=mode, identifiers=self.identifiers())
        return VisionItem(self.build_prompt(rd, len(images)), rd.candidates, images), rd

    # ------------------------------------------------------------------ scoring
    @torch.inference_mode()
    def score_many(self, items: Sequence[VisionItem]) -> list[list[float]]:
        """Summed log-prob of each candidate, gathered at the answer position."""
        out: list[list[float]] = []
        for start in range(0, len(items), self.batch_size):
            chunk = list(items[start:start + self.batch_size])
            texts = [it.prefix for it in chunk]
            images = [it.images for it in chunk]
            kwargs: dict[str, Any] = {"text": texts, "return_tensors": "pt", "padding": True, "padding_side": "left"}
            if any(images):
                kwargs["images"] = images
            enc = self.processor(**kwargs)
            enc = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in enc.items()}
            logits = self.model(**enc).logits[:, -1].float()
            logp = F.log_softmax(logits, dim=-1)
            for row, it in enumerate(chunk):
                toks = [self._candidate_token(it.prefix, c) for c in it.candidates]
                out.append([float(logp[row, t]) for t in toks])
        return out

    def _candidate_token(self, prefix: str, candidate: str) -> int:
        """The single token a candidate contributes after the answer cue."""
        tail = prefix[prefix.rfind(chr(10)) + 1:]
        base = self._encode(tail)
        ids = self._encode(tail + candidate)
        n = _common(base, ids)
        return ids[n] if n < len(ids) else ids[-1]


def _common(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n
