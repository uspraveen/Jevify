"""Where does Jev's time go? Controlled latency probes against the API.

Jev's round trip barely moves with the number of options (186 ms at K=4, 219 ms at
K=151 in our baseline), while the same options cost our engine 1,300 tokens of prefill.
The baseline cannot say why: every request for a config carried the *identical* option
set, it ran eight requests in flight, and it never varied one thing at a time. These
probes do. Each block changes exactly one variable, every request is sequential on one
keep-alive connection, and conditions are interleaved round-robin so drift in the API
cannot masquerade as an effect. Every response's ``usage`` is kept: the API reports how
many input tokens it charged for, which is the token count *it* saw.

    A  caching        K=151, same state: identical options | a nonce inside every option's
                      text | same options, shuffled order | identical options, nonce in state
    B  input cost     K=4, state length 50 .. 6400 tokens (nonce per request)
    C  option cost    K=2 .. 128, every option text unique per request, state fixed
    D  output size    K=32, one-token labels vs ten-token labels (output tokens scale with
                      label length, so this isolates the cost of a bigger answer template)
    E  question count 1 vs 8 questions in one request, same state
    F  window         state length 400 .. 30,000 tokens: a fixed compute window shows as a cliff
    G  max K          K = 2, 64, 255 (the documented maximum), unique one-token labels
    H  batching       the same small request alone vs eight in flight; a batching wait makes a
                      lone client pay the whole wait and *drops* under load

    TYPESAFE_API_KEY=... python scripts/probe_jev_latency.py --n 35 --out results/jev-latency-probe
"""
from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
MODEL = os.environ.get("JEV_MODEL", "jev-1.13.0")

FILLER = ("The customer wrote to us this morning about their account. They explained that they had recently "
          "moved apartments, changed their phone number, and wanted to make sure that statements and alerts "
          "would still reach them. They also asked whether the fee on last month's statement could be reviewed. ")


def nonce(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def state_of_tokens(approx_tokens: int, tag: str) -> str:
    """A natural-text state of roughly the requested length (FILLER is ~60 tokens)."""
    reps = max(1, round(approx_tokens / 60))
    return f"[{tag}] " + (FILLER * reps).strip()


def clinc_criteria() -> dict[str, str]:
    """The real 151-intent option set from jev-bench, so block A uses what the baseline used."""
    from jevify.bench.record import read_jsonl
    recs = read_jsonl(ROOT / "data" / "jev-bench" / "data" / "clinc150" / "test.jsonl", limit=1)
    return dict(next(iter(recs)).question["criteria"])


def synthetic_criteria(k: int, label_tokens: int, unique: str | None) -> dict[str, str]:
    """K options with labels of a chosen length and short descriptions; ``unique`` salts the text."""
    out = {}
    for i in range(k):
        label = f"opt{i:03d}" if label_tokens <= 1 else " ".join([f"option{i:03d}"] + [f"word{j}" for j in range(label_tokens - 1)])
        desc = f"the customer wants outcome number {i} handled" + (f" ref {unique}{i}" if unique else "")
        out[label] = desc
    return out


class Probe:
    def __init__(self, n: int) -> None:
        self.n = n
        self.client = httpx.Client(base_url=BASE, timeout=60.0,
                                  headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
                                           "Content-Type": "application/json"})
        self.rows: list[dict] = []
        self.first_headers: dict | None = None

    def call(self, block: str, cond: str, state: str, questions: dict, meta: dict | None = None) -> dict:
        body = {"state": state, "model": MODEL, "questions": questions}
        t0 = time.perf_counter()
        r = self.client.post("/v1/systemone", json=body)
        ms = (time.perf_counter() - t0) * 1000
        if self.first_headers is None:
            self.first_headers = {k: v for k, v in r.headers.items()
                                  if k.lower() in ("server", "via", "x-request-id", "cf-ray", "x-served-by",
                                                   "server-timing", "x-cache", "x-runtime", "x-process-time",
                                                   "x-envoy-upstream-service-time", "fly-request-id", "x-vercel-id")}
        row = {"block": block, "cond": cond, "ms": round(ms, 1), "status": r.status_code, **(meta or {})}
        if r.headers.get("x-envoy-upstream-service-time"):
            row["server_ms"] = float(r.headers["x-envoy-upstream-service-time"])   # the server's own clock
        if r.status_code == 200:
            u = r.json().get("usage", {})
            row["input_tokens"] = u.get("input_tokens"); row["output_tokens"] = u.get("output_tokens")
            st = r.headers.get("server-timing")
            if st:
                row["server_timing"] = st
        else:
            row["error"] = r.text[:160]
            if r.status_code in (429, 529):
                time.sleep(float(r.headers.get("retry-after", 2)))
        self.rows.append(row)
        return row

    # ------------------------------------------------------------------ blocks
    def block_a(self, crit: dict[str, str]):
        base_state = state_of_tokens(120, "acct")
        keys = list(crit)
        conds = {
            "identical": lambda: (base_state, crit),
            "nonce_in_options": lambda: (base_state, {k: f"{v} ref {nonce()}" for k, v in crit.items()}),
            "shuffled_order": lambda: (base_state, {k: crit[k] for k in random.sample(keys, len(keys))}),
            "nonce_in_state": lambda: (f"[{nonce()}] " + base_state, crit),
        }
        for _ in range(self.n):
            for cond, make in conds.items():
                s, c = make()
                self.call("A", cond, s, {"q": {"type": "choice", "instructions": "Which intent does the message express?",
                                                "criteria": c}}, {"k": len(c)})

    def block_b(self):
        q = {"q": {"type": "choice", "instructions": "What is the customer's main request?",
                   "criteria": {"address": "update contact details", "fee": "review a fee",
                                "card": "replace a card", "other": "something else"}}}
        for _ in range(self.n):
            for toks in (50, 400, 1600, 6400):
                self.call("B", f"state_{toks}", state_of_tokens(toks, nonce()), q, {"target_tokens": toks})

    def block_c(self):
        s = state_of_tokens(120, "acct")
        for _ in range(self.n):
            for k in (2, 8, 32, 128):
                crit = synthetic_criteria(k, 1, nonce())
                self.call("C", f"k_{k}", s, {"q": {"type": "choice", "instructions": "Which outcome fits?", "criteria": crit}},
                          {"k": k})

    def block_d(self):
        s = state_of_tokens(120, "acct")
        for _ in range(self.n):
            for name, lt in (("labels_1tok", 1), ("labels_10tok", 10)):
                crit = synthetic_criteria(32, lt, nonce())
                self.call("D", name, s, {"q": {"type": "choice", "instructions": "Which outcome fits?", "criteria": crit}},
                          {"k": 32, "label_tokens": lt})

    def block_e(self):
        s = state_of_tokens(120, "acct")
        for _ in range(self.n):
            for nq in (1, 8):
                qs = {f"q{i}": {"type": "choice", "instructions": f"Question {i} {nonce()}: which outcome fits?",
                                "criteria": synthetic_criteria(4, 1, nonce())} for i in range(nq)}
                self.call("E", f"questions_{nq}", s, qs, {"n_questions": nq})


    def block_f(self):
        """State length up to the documented 32K limit: a fixed compute window shows as a cliff."""
        q = {"q": {"type": "noul", "instructions": "Does the customer mention a fee?"}}
        for _ in range(self.n):
            for toks in (400, 3000, 8000, 16000, 24000, 30000):
                self.call("F", f"state_{toks}", state_of_tokens(toks, nonce()), q, {"target_tokens": toks})

    def block_g(self):
        """K up to the documented maximum of 255 with one-token labels: the per-option cost."""
        s = state_of_tokens(120, "acct")
        for _ in range(self.n):
            for k in (2, 64, 255):
                crit = synthetic_criteria(k, 1, nonce())
                self.call("G", f"k_{k}", s, {"q": {"type": "choice", "instructions": "Which outcome fits?", "criteria": crit}}, {"k": k})

    def block_h(self):
        """The same small request alone vs eight in flight. A server that waits to fill a batch
        makes a lone client pay the whole wait; under load the batch fills and latency *drops*."""
        import concurrent.futures as cf
        q = {"q": {"type": "noul", "instructions": "Does the customer mention a fee?"}}
        for _ in range(self.n):
            self.call("H", "alone", state_of_tokens(120, nonce()), q)
        clients = [httpx.Client(base_url=BASE, timeout=60.0, headers=dict(self.client.headers)) for _ in range(8)]

        def one(c):
            body = {"state": state_of_tokens(120, nonce()), "model": MODEL, "questions": q}
            t0 = time.perf_counter(); r = c.post("/v1/systemone", json=body); ms = (time.perf_counter() - t0) * 1000
            row = {"block": "H", "cond": "8_in_flight", "ms": round(ms, 1), "status": r.status_code}
            if r.headers.get("x-envoy-upstream-service-time"):
                row["server_ms"] = float(r.headers["x-envoy-upstream-service-time"])
            if r.status_code == 200:
                u = r.json().get("usage", {}); row["input_tokens"] = u.get("input_tokens"); row["output_tokens"] = u.get("output_tokens")
            return row
        for _ in range(max(1, self.n // 8)):
            with cf.ThreadPoolExecutor(8) as ex:
                self.rows.extend(ex.map(one, clients))
        for c in clients:
            c.close()


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["block"], r["cond"]) for r in rows})
    for b, c in keys:
        rs = [r for r in rows if r["block"] == b and r["cond"] == c and r["status"] == 200]
        if not rs:
            out.append({"block": b, "cond": c, "n": 0}); continue
        ms = np.array([r["ms"] for r in rs])
        sv = [r["server_ms"] for r in rs if r.get("server_ms") is not None]
        it = [r["input_tokens"] for r in rs if r.get("input_tokens") is not None]
        ot = [r["output_tokens"] for r in rs if r.get("output_tokens") is not None]
        out.append({"block": b, "cond": c, "n": len(rs), "p50_ms": float(np.median(ms)), "p10_ms": float(np.percentile(ms, 10)),
                    "p90_ms": float(np.percentile(ms, 90)), "server_p50_ms": float(np.median(sv)) if sv else None,
                    "input_tokens": float(np.median(it)) if it else None,
                    "output_tokens": float(np.median(ot)) if ot else None,
                    "errors": sum(1 for r in rows if r["block"] == b and r["cond"] == c and r["status"] != 200)})
    return out


def markdown(summary: list[dict], headers: dict | None) -> str:
    lines = ["# Jev latency probes — where does the time go?", "",
             f"Model `{MODEL}`, sequential requests on one keep-alive connection, conditions interleaved round-robin. "
             "`input_tokens` / `output_tokens` are the API's own accounting (medians).", "",
             "| block | condition | n | p10 ms | p50 ms | p90 ms | server p50 ms | input tok | output tok | errors |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        if s["n"] == 0:
            lines.append(f"| {s['block']} | {s['cond']} | 0 | — | — | — | — | — | — | — |"); continue
        srv = f"{s['server_p50_ms']:.0f}" if s.get("server_p50_ms") is not None else "—"
        lines.append(f"| {s['block']} | {s['cond']} | {s['n']} | {s['p10_ms']:.0f} | {s['p50_ms']:.0f} | {s['p90_ms']:.0f} | {srv} | "
                     f"{s['input_tokens']:.0f} | {s['output_tokens']:.0f} | {s['errors']} |")
    if headers:
        lines += ["", "Response headers of note: " + ", ".join(f"`{k}: {v}`" for k, v in headers.items())]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=35, help="requests per condition")
    ap.add_argument("--blocks", default="A,B,C,D,E")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "jev-latency-probe")
    ap.add_argument("--tag", default="", help="suffix for the output files, to keep probe rounds apart")
    a = ap.parse_args()
    random.seed(20260921)
    p = Probe(a.n)
    # warm the connection so TLS setup is not attributed to the first condition
    p.call("warm", "warm", state_of_tokens(50, "w"), {"q": {"type": "noul", "instructions": "Is this a complaint?"}})
    blocks = {"A": lambda: p.block_a(clinc_criteria()), "B": p.block_b, "C": p.block_c, "D": p.block_d, "E": p.block_e,
              "F": p.block_f, "G": p.block_g, "H": p.block_h}
    for b in [x for x in a.blocks.split(",") if x]:
        t0 = time.time(); blocks[b](); print(f"block {b}: {time.time()-t0:.0f}s, {len(p.rows)} requests so far", flush=True)
    summary = summarize([r for r in p.rows if r["block"] != "warm"])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"probe{a.tag}.json").write_text(json.dumps({"model": MODEL, "base": BASE, "n": a.n, "summary": summary,
                                                  "headers": p.first_headers, "rows": p.rows}, indent=1), encoding="utf-8")
    md = markdown(summary, p.first_headers)
    (a.out / f"README{a.tag}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
