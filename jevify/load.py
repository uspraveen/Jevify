"""Load a published Jevified model and ask it questions.

    from jevify import load_jevified

    model = load_jevified("Praveenrajus/jevify-qwen3.5-4b")
    model.ask(state, questions)          # -> {question id: wire answer}

The published repo carries only the decision heads and a config; the backbone is pulled
from its own Hub repo at load time. A Tier 0 repo (no heads) works too — it is just the
recipe, and ``ask`` then uses the logit readout alone.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .engine.predict import Recipe, Tier0Engine, finalize
from .engine.readout import HFScorer
from .wire import SystemOneRequest, parse_question


class JevifiedModel:
    """A System One model: state + typed questions in, calibrated distributions out."""

    def __init__(self, scorer: HFScorer, config: dict[str, Any], heads=None) -> None:
        self.config = config
        self.backbone = config["backbone"]
        self.heads = heads
        recipe_cfg = config.get("recipe") or {}
        self.engine = Tier0Engine(scorer, Recipe(
            mode=recipe_cfg.get("mode", "index"),
            chat=config.get("chat", True),
            permutations=recipe_cfg.get("permutations", 1),
            prior_weight=recipe_cfg.get("prior_weight", 0.0),
            temperature=recipe_cfg.get("temperature") or {"choice": 1.0, "score": 1.0, "noul": 1.0},
            bias=recipe_cfg.get("bias") or {"noul": 0.0},
            state_last=bool(recipe_cfg.get("state_last", False)),
        ))
        self._extractor = None
        # a vision-language scorer renders its own prompts (images go through the processor,
        # not the tokenizer), so `ask` takes a different path to the same wire format
        self.vision = config.get("modality") == "vision" or type(scorer).__name__ == "VisionScorer"
        if heads is not None:
            from .engine.features import FeatureExtractor

            self._extractor = FeatureExtractor(scorer, layer=config.get("layer", -1), chat=config.get("chat", True))

    # ------------------------------------------------------------------ inference
    def ask(self, state: Any, questions: dict[str, Any], *, model_name: str | None = None) -> dict[str, Any]:
        """Answer every question about one state. Returns the wire-format answers."""
        req = SystemOneRequest.model_validate({"state": state, "model": model_name or self.backbone,
                                               "questions": questions})
        qs = {k: q.model_dump(exclude_none=True) for k, q in req.questions.items()}
        if self.vision:
            return self._ask_vision(state, qs)
        if self.heads is None:
            return self._ask_tier0(state, qs)
        return self._ask_heads(state, qs)

    def _ask_vision(self, state: Any, qs: dict[str, Any]) -> dict[str, Any]:
        """Images in `state` (PIL, path, URL, data URI or bytes) reach the model through its
        processor; everything after the logits is the text path's recipe and wire format."""
        engine = self.engine
        items, plan = [], []
        for qid, qd in qs.items():
            it, rd = engine.scorer.item(state, qd, mode=engine.recipe.mode)
            items.append(it)
            plan.append((qid, qd, rd))
        scores = engine.scorer.score_many(items)
        out = {}
        for (qid, qd, rd), sc in zip(plan, scores):
            pred = finalize(qd["type"], qd, {"mode": rd.mode, "runs": [{"keys": rd.keys, "logscores": sc}], "prior": None},
                            engine.recipe)
            out[qid] = _to_wire(qd, pred)
        return out

    def _ask_tier0(self, state: Any, qs: dict[str, Any]) -> dict[str, Any]:
        engine = self.engine
        items, plan = [], []
        for qid, qd in qs.items():
            rds = engine._renderings(state, qd)
            for rd in rds:
                items.append((engine._prefix(rd), rd.candidates))
            plan.append((qid, qd, rds))
        scores = engine.scorer.score_many(items)
        out, k = {}, 0
        for qid, qd, rds in plan:
            runs = []
            for rd in rds:
                runs.append({"keys": rd.keys, "logscores": scores[k]})
                k += 1
            pred = finalize(qd["type"], qd, {"mode": rds[0].mode, "runs": runs, "prior": None}, engine.recipe)
            out[qid] = _to_wire(qd, pred)
        return out

    def _ask_heads(self, state: Any, qs: dict[str, Any]) -> dict[str, Any]:
        from .bench.record import BenchRecord
        from .engine.heads import predict_rows

        recs, keys_of = [], {}
        for qid, qd in qs.items():
            q = parse_question(qd)
            placeholder = (list(qd["criteria"].keys())[0] if qd["type"] == "choice"
                           else ("0" if qd["type"] == "score" else 0))
            recs.append(BenchRecord(id=qid, source="live", primitive=qd["type"], split="test",
                                    state=state, question=qd, label=placeholder))
            keys_of[qid] = qd
        rows = self._extractor.extract(recs, batch_size=max(1, min(8, len(recs))))
        # the heads live on the scorer's device; collating to cpu would mix devices
        dists = predict_rows(self.heads, rows, device=self.engine.scorer.device)
        out = {}
        for qid, qd in keys_of.items():
            dist = dists[qid]
            out[qid] = _to_wire(qd, None, dist)
        return out

    def __repr__(self) -> str:
        tier = "Tier 2" if self.config.get("lora") else ("Tier 1" if self.heads is not None else "Tier 0")
        tier += " vision" if self.vision else ""
        return f"JevifiedModel({self.backbone!r}, {tier})"


def _to_wire(qd: dict[str, Any], pred, dist: dict[str, float] | None = None) -> dict[str, Any]:
    from .wire import choice_confidence, round_probabilities, score_confidence, score_expectation

    if dist is None:
        if qd["type"] == "noul":
            return {"type": "noul", "noul": round(pred.p_yes, 4)}
        if qd["type"] == "choice":
            return {"type": "choice", "choice": pred.answer, "confidence": round(pred.confidence, 4),
                    "probabilities": pred.probabilities}
        return {"type": "score", "score": round(pred.answer, 4), "confidence": round(pred.confidence, 4),
                "legend": {str(i): c for i, c in enumerate(qd["criteria"])}, "probabilities": pred.probabilities}
    if qd["type"] == "noul":
        return {"type": "noul", "noul": round(float(dist.get("1", 0.0)), 4)}
    keys = list(qd["criteria"].keys()) if qd["type"] == "choice" else [str(i) for i in range(len(qd["criteria"]))]
    vals = round_probabilities([float(dist.get(k, 0.0)) for k in keys], 4)
    pm = dict(zip(keys, vals))
    if qd["type"] == "choice":
        return {"type": "choice", "choice": max(pm, key=pm.get), "confidence": round(choice_confidence(vals), 4),
                "probabilities": pm}
    return {"type": "score", "score": round(score_expectation(vals), 4), "confidence": round(score_confidence(vals), 4),
            "legend": {str(i): c for i, c in enumerate(qd["criteria"])}, "probabilities": pm}


def load_jevified(repo_or_path: str, *, device: str | None = None, hf_token: str | None = None,
                  trust_remote_code: bool = False, engine: str = "hf") -> JevifiedModel:
    """Load a Jevified model from the Hub or a local directory.

    ``engine="vllm"`` serves the Tier 0 readout (or a merged Tier 2 backbone) through vLLM
    with prefix caching -- the production path. Heads (Tier 1) need hidden states and stay
    on the Hugging Face path.
    """
    import os

    path = Path(repo_or_path)
    if not path.exists():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_or_path, token=hf_token or os.environ.get("HF_TOKEN")))
    config = _config_for(path)
    if config.get("modality") == "vision":
        from .engine.vision import VisionScorer

        scorer = VisionScorer(config["backbone"], trust_remote_code=trust_remote_code or config.get("trust_remote_code", False),
                              hf_token=hf_token or os.environ.get("HF_TOKEN"),
                              max_pixels=config.get("max_pixels") or None)
        return JevifiedModel(scorer, config, None)
    if engine == "vllm":
        from .engine.vllm_readout import VLLMScorer

        if (path / "lora" / "adapter_config.json").exists() or (path / "heads" / "heads.pt").exists():
            raise ValueError("engine='vllm' serves the Tier 0 readout of a backbone; merge a LoRA into a checkpoint first, "
                             "and Tier 1 heads need the Hugging Face path")
        scorer = VLLMScorer(config["backbone"], trust_remote_code=trust_remote_code or config.get("trust_remote_code", False),
                            hf_token=hf_token or os.environ.get("HF_TOKEN"))
        return JevifiedModel(scorer, config, None)
    scorer = HFScorer(config["backbone"], device=device, hf_token=hf_token or os.environ.get("HF_TOKEN"),
                      trust_remote_code=trust_remote_code or config.get("trust_remote_code", False))
    if (path / "lora" / "adapter_config.json").exists():
        # Tier 2: the adapter is merged into the weights at load, so serving a LoRA-trained model
        # costs exactly what serving its backbone costs -- no adapter matmuls at inference
        from peft import PeftModel

        scorer.model = PeftModel.from_pretrained(scorer.model, str(path / "lora")).merge_and_unload().eval()
        config = {**config, "lora": True}
    heads = None
    if (path / "heads" / "heads.pt").exists():
        from .engine.heads import load_heads

        heads = load_heads(path / "heads", device=scorer.device)
    return JevifiedModel(scorer, config, heads)


def _config_for(path: Path) -> dict[str, Any]:
    """A published repo carries jevify_config.json; a raw training output directory (what
    ``python -m jevify.train`` writes) carries run.json. Either loads, so a model can be
    tried the moment training finishes, before anything is published."""
    cfg = path / "jevify_config.json"
    if cfg.exists():
        return json.loads(cfg.read_text(encoding="utf-8"))
    run = path / "run.json"
    if not run.exists():
        raise FileNotFoundError(f"{path}: neither jevify_config.json nor run.json found")
    meta = json.loads(run.read_text(encoding="utf-8"))
    recipe = meta.get("recipe") or {}
    return {"backbone": meta["model_id"], "chat": meta.get("chat_applied", True), "tier": meta.get("tier", 0),
            "modality": meta.get("modality", "text"), "max_pixels": meta.get("max_pixels"),
            "recipe": {"mode": recipe.get("mode", "index"), "permutations": recipe.get("permutations", 1),
                       "prior_weight": recipe.get("prior_weight", 0.0), "temperature": recipe.get("temperature"),
                       "bias": recipe.get("bias")},
            "trust_remote_code": meta.get("trust_remote_code", False)}
