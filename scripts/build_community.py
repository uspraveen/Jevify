"""Build three public, third-party Jev benchmarks as jev-bench-format records.

    python scripts/build_community.py --out <dir> [--together-inputs <tev1 repo>/evaluation/public-third-party/inputs.json]

Every number jev-bench produces is on a benchmark this project wrote. These three were written by
other people, to test Jev, before any Jevified model existed:

- ``phishnchips_*``  anisselbd/jev-phishing-bench: 2,000 PhishNChips v5.2 emails (1,000 phishing
  around real malicious URLs, 1,000 legitimate), and the question set its author sent to Jev.
  Four of those questions have a gold answer, so four configs: the verdict (Choice), the same
  question as a Noul, and the author's two rewordings of the verdict. The five "signal"
  questions have no labels and are not scored.
- ``tool_risk``      themsquared/jev-benchmark: 60 hand-labelled agent tool calls, four risk
  classes, tagged clear / ambiguous / adversarial.
- ``ticket_routing`` WallerChen/jev-measured: 27 support tickets with one decisive signal each,
  plus 6 deliberately ambiguous tickets with *no* label (split ``unlabelled``), which exist to
  see whether a model's confidence drops when the ticket genuinely is ambiguous.

The questions, criteria and labels are read from each repository at a pinned commit (parsed
from its source, not retyped) and the PhishNChips file is checked against the SHA-256 its
benchmark pins. Nothing from these repositories is committed here or uploaded anywhere: this
script fetches them, and the published results carry ids and scores only.

``--together-inputs`` compares the result with the frozen inputs Together AI used for the same
three benchmarks (their tev1 repository), so the two evaluations can be checked to be the same
records.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevify.bench.record import BenchRecord, write_jsonl  # noqa: E402

PINS = {
    "jev-phishing-bench": ("anisselbd/jev-phishing-bench", "1d56e8c64d029a9554a0874e2ef2901ed196e230"),
    "jev-benchmark": ("themsquared/jev-benchmark", "d699d44558e27c9071caef8d5cea51615d5a3131"),
    "jev-measured": ("WallerChen/jev-measured", "4a12dfb3e59fe368760af44607745c97b65e3fa9"),
}
PHISHNCHIPS_CSV = "https://huggingface.co/datasets/AreLit/PhishNChips/resolve/main/core_emails.csv"
PHISHNCHIPS_SHA256 = "cebb407ff8630491a97400e37464b8db8dfc4299164fca51fcb4ac7eec8204ef"   # pinned by prepare_data.py
EMAIL_FIELDS = ["sender", "from", "subject", "body", "link_display_text", "link_url"]  # prepare_data.EMAIL_FIELDS

# config -> (question id in run_jev.QUESTIONS, primitive, label for a phishing email, label for a legitimate one)
PHISHING_CONFIGS = {
    "phishnchips_verdict": ("verdict", "choice", "phishing", "legitimate"),
    "phishnchips_noul": ("is_phishing", "noul", 1, 0),
    "phishnchips_click": ("verdict_alt_click", "choice", "do_not_click", "click"),
    "phishnchips_minimal": ("verdict_alt_minimal", "choice", "phishing", "legitimate"),
}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "jevify-build-community"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def raw(repo_key: str, path: str) -> tuple[bytes, str]:
    repo, commit = PINS[repo_key]
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    return fetch(url), url


def literals(source: str, *names: str) -> dict:
    """Module-level constants of a Python file, evaluated as literals (never executed)."""
    out = {}
    for node in ast.parse(source).body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            targets = [node.target.id]
        for t in targets:
            if t in names:
                out[t] = ast.literal_eval(node.value)
    missing = set(names) - set(out)
    if missing:
        raise SystemExit(f"could not find {sorted(missing)} in the pinned source")
    return out


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_phishing(files: dict) -> dict[str, list[BenchRecord]]:
    src, url = raw("jev-phishing-bench", "run_jev.py")
    files[url] = sha256(src)
    questions = literals(src.decode(), "QUESTIONS")["QUESTIONS"]
    data = fetch(PHISHNCHIPS_CSV)
    digest = sha256(data)
    if digest != PHISHNCHIPS_SHA256:
        raise SystemExit(f"PhishNChips core_emails.csv changed upstream: {digest}")
    files[PHISHNCHIPS_CSV] = digest
    csv.field_size_limit(1 << 30)
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    if len(rows) != 2000:
        raise SystemExit(f"expected 2000 emails, got {len(rows)}")
    out: dict[str, list[BenchRecord]] = {c: [] for c in PHISHING_CONFIGS}
    skipped = 0
    for row in rows:
        try:
            email = json.loads(row["email_content"])
        except json.JSONDecodeError:
            skipped += 1
            continue
        if any(k not in email for k in EMAIL_FIELDS):
            skipped += 1
            continue
        state = {k: email[k] for k in EMAIL_FIELDS}
        phish = int(row["phish_label"]) == 1
        for config, (qid, prim, yes_label, no_label) in PHISHING_CONFIGS.items():
            out[config].append(BenchRecord(
                id=f"{config}/test/{row['id']}", source=config, primitive=prim, split="test", state=state,
                question=questions[qid], label=yes_label if phish else no_label,
                meta={"origin": "anisselbd/jev-phishing-bench + AreLit/PhishNChips v5.2", "email_id": row["id"],
                      "url_category": row["url_category"], "strategy": row["strategy"],
                      "datasource": row["datasource"]}))
    if skipped:
        raise SystemExit(f"{skipped} emails failed to parse; the source benchmark keeps all 2,000")
    return out


def build_tool_risk(files: dict) -> list[BenchRecord]:
    src, url = raw("jev-benchmark", "bench.py")
    files[url] = sha256(src)
    consts = literals(src.decode(), "INSTRUCTIONS", "CRITERIA")
    tasks_b, turl = raw("jev-benchmark", "tasks.jsonl")
    files[turl] = sha256(tasks_b)
    question = {"type": "choice", "instructions": consts["INSTRUCTIONS"], "criteria": consts["CRITERIA"]}
    recs = []
    for line in tasks_b.decode().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        assert t["label"] in consts["CRITERIA"], t
        recs.append(BenchRecord(id=f"tool_risk/test/{t['id']}", source="tool_risk", primitive="choice", split="test",
                                state=t["state"], question=question, label=t["label"],
                                meta={"origin": "themsquared/jev-benchmark", "task_id": t["id"],
                                      "difficulty": t["difficulty"]}))
    return recs


def build_tickets(files: dict) -> tuple[list[BenchRecord], list[BenchRecord]]:
    src, url = raw("jev-measured", "bench/labelled.py")
    files[url] = sha256(src)
    text = src.decode()
    consts = literals(text, "QUEUES", "UNAMBIGUOUS", "AMBIGUOUS")
    instructions = "Which team should handle this ticket?"   # choice_question() in the same file
    if f'"instructions": "{instructions}"' not in text:
        raise SystemExit("choice_question() changed in the pinned source")
    question = {"type": "choice", "instructions": instructions, "criteria": consts["QUEUES"]}
    labelled = [BenchRecord(id=f"ticket_routing/test/{i}", source="ticket_routing", primitive="choice", split="test",
                            state=t, question=question, label=gold,
                            meta={"origin": "WallerChen/jev-measured", "index": i})
                for i, (t, gold) in enumerate(consts["UNAMBIGUOUS"])]
    first = next(iter(consts["QUEUES"]))
    # no gold answer exists for these by design; the label is a placeholder that nothing scores
    unlabelled = [BenchRecord(id=f"ticket_routing/unlabelled/{i}", source="ticket_routing", primitive="choice",
                              split="unlabelled", state=t, question=question, label=first,
                              meta={"origin": "WallerChen/jev-measured", "index": i, "unlabelled": True})
                  for i, t in enumerate(consts["AMBIGUOUS"])]
    return labelled, unlabelled


def compare_with_together(path: Path, built: dict[str, list[BenchRecord]]) -> dict:
    """Same records as Together's frozen inputs? (state, question text, option keys and
    descriptions in order, gold). Differences are reported, not silently reconciled."""
    theirs = json.loads(path.read_text(encoding="utf-8"))
    by_suite = {
        "phishing": {r.meta["email_id"]: r for r in built["phishnchips_verdict"]},
        "tool_risk": {r.meta["task_id"]: r for r in built["tool_risk"]},
        "ticket_routing": {str(r.meta["index"]): r for r in built["ticket_routing"]},
    }
    report: dict = {"compared": 0, "missing": [], "diffs": Counter(), "examples": []}
    for row in theirs:
        suite = row["suite"]
        key = row["id"].split(":", 1)[1]        # "phishing:<email id>", "tool_risk:<task id>", "ticket_routing:<index>"
        r = by_suite[suite].get(key)
        if r is None:
            report["missing"].append(row["id"])
            continue
        report["compared"] += 1
        t = row["task"]
        checks = {
            "state": t["state"] == r.state,
            "question": t["question"] == r.question["instructions"],
            "options": [(o["key"], o["description"]) for o in t["options"]]
                       == [(k, v if v else k) for k, v in r.question["criteria"].items()],
            "gold": row["gold"] == r.label,
        }
        for name, ok in checks.items():
            if not ok:
                report["diffs"][f"{suite}:{name}"] += 1
                if len(report["examples"]) < 6:
                    report["examples"].append({"id": row["id"], "field": name,
                                               "together": (t["state"] if name == "state" else t.get(name)) if name != "gold" else row["gold"],
                                               "source": r.state if name == "state" else None})
    report["diffs"] = dict(report["diffs"])
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--together-inputs", type=Path, default=None)
    a = ap.parse_args()
    files: dict[str, str] = {}
    built: dict[str, list[BenchRecord]] = {}
    built.update(build_phishing(files))
    built["tool_risk"] = build_tool_risk(files)
    labelled, unlabelled = build_tickets(files)
    built["ticket_routing"] = labelled
    counts = {}
    for config, recs in built.items():
        n = write_jsonl(a.out / "data" / config / "test.jsonl", recs)
        counts[config] = {"test": n, "labels": dict(Counter(str(r.label) for r in recs))}
    counts["ticket_routing"]["unlabelled"] = write_jsonl(a.out / "data" / "ticket_routing" / "unlabelled.jsonl", unlabelled)
    manifest = {"name": "jev-community", "pins": {k: {"repo": v[0], "commit": v[1]} for k, v in PINS.items()},
                "phishnchips": {"url": PHISHNCHIPS_CSV, "sha256": PHISHNCHIPS_SHA256}, "fetched_sha256": files,
                "configs": counts}
    if a.together_inputs:
        manifest["together_inputs_check"] = compare_with_together(a.together_inputs, built)
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
