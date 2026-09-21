"""A Jevified model behind TypeSafe's wire protocol.

    jevify-serve --model Qwen/Qwen3.5-0.8B [--port 8000] [--mode index] [--temperature choice=1.3,...]

    TYPESAFE_BASE_URL=http://localhost:8000 python -c "from typesafe_sdk import *; ..."

Every question in a request is rendered and scored in one batched call (the
state is shared, questions are independent by construction). ``usage`` counts
prefix tokens as input and scored candidate tokens as output, which is the
honest analogue of Jev's accounting.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .engine.predict import Recipe, Tier0Engine, finalize
from .engine.template import render
from .wire import SystemOneRequest, SystemOneResponse

MODEL_ALIAS = "jevify-latest"


class SystemOneService:
    def __init__(self, engine: Tier0Engine, model_name: str) -> None:
        self.engine = engine
        self.model_name = model_name
        self.released = dt.datetime.now(dt.timezone.utc).date().isoformat()

    jevified = None

    def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        engine = self.engine
        if self.jevified is not None and self.jevified.heads is not None:
            qs = {k: q.model_dump(exclude_none=True) for k, q in req.questions.items()}
            answers = self.jevified.ask(req.state, qs, model_name=self.model_name)
            tok = sum(len(engine.scorer.tokenize(engine._prefix(rd), rd.candidates).prefix_ids)
                      for qid, qd in qs.items() for rd in engine._renderings(req.state, qd))
            return SystemOneResponse.model_validate({"model": self.model_name, "answers": answers,
                                                     "usage": {"input_tokens": tok, "output_tokens": len(qs)}})
        items: list[tuple[str, list[str]]] = []
        plan: list[tuple[str, Any, list[Any]]] = []
        for qid, q in req.questions.items():
            qd = q.model_dump(exclude_none=True)
            rds = engine._renderings(req.state, qd)
            for rd in rds:
                items.append((engine._prefix(rd), rd.candidates))
            plan.append((qid, qd, rds))
        scores = engine.scorer.score_many(items)
        in_tok = sum(len(engine.scorer.tokenize(p, c).prefix_ids) for p, c in items[: len(items)])
        out_tok = sum(sum(len(t) for t in engine.scorer.tokenize(p, c).cand_ids) for p, c in items)
        answers: dict[str, Any] = {}
        k = 0
        for qid, qd, rds in plan:
            runs = []
            for rd in rds:
                runs.append({"keys": rd.keys, "logscores": scores[k]})
                k += 1
            extra = {"mode": rds[0].mode, "runs": runs, "prior": None}
            pred = finalize(qd["type"], qd, extra, engine.recipe)
            if qd["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": round(pred.p_yes, 4)}
            elif qd["type"] == "choice":
                answers[qid] = {"type": "choice", "choice": pred.answer, "confidence": round(pred.confidence, 4),
                                "probabilities": pred.probabilities}
            else:
                legend = {str(i): c for i, c in enumerate(qd["criteria"])}
                answers[qid] = {"type": "score", "score": round(pred.answer, 4), "confidence": round(pred.confidence, 4),
                                "legend": legend, "probabilities": pred.probabilities}
        return SystemOneResponse.model_validate({"model": self.model_name, "answers": answers,
                                                 "usage": {"input_tokens": in_tok, "output_tokens": out_tok}})


def build_app(service: SystemOneService) -> FastAPI:
    app = FastAPI(title="jevify", version=__version__)

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        body = await request.json()
        try:
            req = SystemOneRequest.model_validate(body)
        except Exception as e:  # mirror TypeSafe's 422 with a detail list
            raise HTTPException(status_code=422, detail=[{"loc": ["body"], "msg": str(e), "type": "value_error"}])
        t0 = time.perf_counter()
        resp = service.answer(req)
        out = JSONResponse(resp.model_dump())
        out.headers["x-jevify-latency-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        out.headers["x-jevify-request-id"] = os.urandom(8).hex()
        return out

    @app.get("/v1/models")
    async def models():
        tier = "Tier 1" if getattr(service.jevified, "heads", None) is not None else "Tier 0"
        return {"models": [{"name": MODEL_ALIAS, "description": f"Jevified {service.model_name} ({tier})", "release_date": service.released},
                           {"name": service.model_name, "description": "The underlying checkpoint", "release_date": service.released}]}

    @app.get("/healthz")
    async def health():
        return {"ok": True, "model": service.model_name}

    return app


def make_service(model_id: str, *, mode: str = "index", chat: bool = True, permutations: int = 1,
                 temperature: dict[str, float] | None = None, device: str | None = None) -> SystemOneService:
    """Serve either a plain HF checkpoint (Tier 0) or a published Jevified repo (Tier 0 or 1)."""
    from .engine.readout import HFScorer

    jevified = _try_load_jevified(model_id, device)
    if jevified is not None:
        svc = SystemOneService(jevified.engine, model_name=model_id)
        svc.jevified = jevified
        return svc
    scorer = HFScorer(model_id, device=device, hf_token=os.environ.get("HF_TOKEN"))
    recipe = Recipe(mode=mode, chat=chat, permutations=permutations, temperature=temperature or Recipe().temperature)
    return SystemOneService(Tier0Engine(scorer, recipe), model_name=model_id)


def _try_load_jevified(model_id: str, device: str | None):
    """Return a JevifiedModel if model_id names a published Jevify repo, else None."""
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(model_id, "jevify_config.json", token=os.environ.get("HF_TOKEN"))
    except Exception:
        if not (Path(model_id) / "jevify_config.json").exists():
            return None
    from .load import load_jevified

    return load_jevified(model_id, device=device)


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(prog="jevify-serve")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mode", default="index", choices=["index", "label"])
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--permutations", type=int, default=1)
    ap.add_argument("--temperature", default="", help="e.g. choice=1.4,score=1.2,noul=1.1")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    temps = {k: float(v) for k, v in (kv.split("=") for kv in a.temperature.split(",") if kv)} or None
    svc = make_service(a.model, mode=a.mode, chat=not a.no_chat, permutations=a.permutations, temperature=temps, device=a.device)
    uvicorn.run(build_app(svc), host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
