"""Run bench records through a System One HTTP API (TypeSafe's Jev, or a Jevified server).

One request per record (each record has its own state). Concurrency-limited,
retries 429/529 with backoff honoring ``retry-after``. Reads the key from
``TYPESAFE_API_KEY`` and the base URL from ``TYPESAFE_BASE_URL`` — the same
variables the official SDK uses.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Iterable, Iterator

import httpx

from ..bench.record import BenchRecord
from .base import Prediction, Runner

DEFAULT_BASE_URL = "https://api.typesafe.ai"
RETRY_STATUSES = {429, 529, 502, 503}


class SystemOneAPIRunner(Runner):
    name = "systemone-api"

    def __init__(self, *, model: str = "jev-latest", base_url: str | None = None, api_key: str | None = None,
                 concurrency: int = 8, max_retries: int = 6, timeout: float = 30.0) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.timeout = timeout

    def predict(self, records: Iterable[BenchRecord]) -> Iterator[Prediction]:
        """Yields predictions in completion order so callers can persist incrementally."""
        recs = list(records)
        queue: "asyncio.Queue[Prediction | None]" = asyncio.Queue()
        loop = asyncio.new_event_loop()
        producer = loop.create_task(self._run(recs, queue))
        try:
            while True:
                item = loop.run_until_complete(queue.get())
                if item is None:
                    break
                yield item
            loop.run_until_complete(producer)
        finally:
            loop.close()

    async def _run(self, recs: list[BenchRecord], queue: "asyncio.Queue[Prediction | None]") -> None:
        sem = asyncio.Semaphore(self.concurrency)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=self.timeout) as client:
            async def one(r: BenchRecord) -> None:
                async with sem:
                    pred = await self._predict_one(client, r)
                await queue.put(pred)
            await asyncio.gather(*(one(r) for r in recs))
        await queue.put(None)

    async def _predict_one(self, client: httpx.AsyncClient, rec: BenchRecord) -> Prediction:
        body = {"state": rec.state, "model": self.model, "questions": {"q": rec.question}}
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = await client.post("/v1/systemone", json=body)
            except httpx.HTTPError as e:
                if attempt == self.max_retries:
                    return Prediction(id=rec.id, primitive=rec.primitive, model=self.model, error=f"transport: {e}")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            latency = (time.perf_counter() - t0) * 1000
            if resp.status_code in RETRY_STATUSES and attempt < self.max_retries:
                ra = resp.headers.get("retry-after-ms") or resp.headers.get("retry-after")
                wait = (float(ra) / 1000 if "retry-after-ms" in resp.headers else float(ra)) if ra else delay
                await asyncio.sleep(min(max(wait, 0.5), 60))
                delay = min(delay * 2, 30)
                continue
            if resp.status_code != 200:
                return Prediction(id=rec.id, primitive=rec.primitive, model=self.model,
                                  error=f"http {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            return self._to_prediction(rec, data, latency)
        return Prediction(id=rec.id, primitive=rec.primitive, model=self.model, error="exhausted retries")

    def _to_prediction(self, rec: BenchRecord, data: dict, latency: float) -> Prediction:
        ans = data["answers"]["q"]
        usage = data.get("usage")
        model = data.get("model", self.model)
        if rec.primitive == "noul":
            return Prediction(id=rec.id, primitive="noul", p_yes=float(ans["noul"]), answer=float(ans["noul"]),
                              model=model, latency_ms=latency, usage=usage)
        if rec.primitive == "choice":
            probs = {k: float(v) for k, v in ans["probabilities"].items()}
            return Prediction(id=rec.id, primitive="choice", probabilities=probs, answer=ans["choice"],
                              confidence=float(ans["confidence"]), model=model, latency_ms=latency, usage=usage)
        probs = {str(k): float(v) for k, v in ans["probabilities"].items()}
        return Prediction(id=rec.id, primitive="score", probabilities=probs, answer=float(ans["score"]),
                          confidence=float(ans["confidence"]), model=model, latency_ms=latency, usage=usage)
