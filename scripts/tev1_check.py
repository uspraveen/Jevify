"""Pre-flight for scoring Together AI's Tev1-4B-experimental in its own prompt format.

    python scripts/tev1_check.py --model togethercomputer/Tev1-4B-experimental --revision <sha> --base Qwen/Qwen3.5-4B

A comparison is only as good as its harness, so before any benchmark run this checks, and exits
non-zero on the first failure:

1. loading: Tev1 loads through the same text-only class as its base with the *same* missing and
   unused weight sets -- a decoder weight re-initialised at load would score noise, plausibly;
2. it is a different model: a decoder projection differs from the base checkpoint's;
3. the prompt: the chat template ends in the empty non-thinking assistant turn Tev1 was trained to
   continue, and every answer letter A-X is exactly one token after it (the scorer's joint
   tokenization agrees with the plain token id);
4. behaviour: Together's four published examples (examples/*.json in their repository),
   answered by the readout, with the full distribution printed.
"""
from __future__ import annotations

import argparse
import json
import string
import sys

import torch

EXAMPLES = {   # github.com/togethercomputer/tev1 examples/, transcribed as typed questions
    "charge-dispute": ("Customer message: Hi, I checked my statement and your company charged my card twice for the October "
                       "subscription. The amounts are both $19.99 on the same day. I have not changed my plan.",
                       {"type": "choice", "instructions": "Which listed support intent best matches this customer's message?",
                        "criteria": {"duplicate_charge": "The customer reports being charged more than once.",
                                     "cancel_subscription": "The customer wants to end or downgrade a subscription.",
                                     "card_declined": "The customer reports a payment that failed or was declined.",
                                     "none": "None of the listed intents matches."}}),
    "yes-no": ("The museum is open Tuesday through Sunday from 10 a.m. to 6 p.m. It is closed on Mondays.",
               {"type": "choice", "instructions": "Is the museum open on Monday?", "criteria": {"yes": "Yes.", "no": "No."}}),
}


def load_info(model_id: str, revision: str | None) -> tuple[set, set, torch.nn.Module]:
    from transformers import AutoModelForCausalLM

    m, info = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16,
                                                   output_loading_info=True)
    return set(info["missing_keys"]), set(info["unexpected_keys"]), m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--base", required=True)
    ap.add_argument("--examples", default="", help="directory holding Together's examples/*.json (optional)")
    a = ap.parse_args()
    ok = True

    # 1-2. loading, and that it is not the base model
    miss_t, unexp_t, mt = load_info(a.model, a.revision)
    miss_b, unexp_b, mb = load_info(a.base, None)
    print(f"[load] {a.model}: {len(miss_t)} missing, {len(unexp_t)} unused; {a.base}: {len(miss_b)} missing, {len(unexp_b)} unused")
    print("[load] missing (tev1):", sorted(miss_t)[:8], "| unused sample:", sorted(unexp_t)[:3])
    if miss_t != miss_b or unexp_t != unexp_b:
        print("[load] FAIL: the two checkpoints load differently"); ok = False
    if any("language_model" in k or k.startswith("model.layers") or "lm_head" in k for k in miss_t):
        print("[load] FAIL: a decoder weight was not loaded"); ok = False
    sd_t, sd_b = mt.state_dict(), mb.state_dict()
    diffs = {}
    for name in sd_t:
        if name.endswith(("q_proj.weight", "in_proj_qkv.weight", "gate_proj.weight")) and name in sd_b:
            diffs[name] = (sd_t[name].float() - sd_b[name].float()).abs().mean().item()
    changed = sum(v > 0 for v in diffs.values())
    print(f"[weights] {changed}/{len(diffs)} projection matrices differ from the base; "
          f"mean |dW| {sum(diffs.values()) / max(len(diffs), 1):.2e}")
    if changed == 0:
        print("[weights] FAIL: identical to the base checkpoint"); ok = False
    del mt, mb, sd_t, sd_b

    # 3. the prompt and the answer tokens, through the real scorer
    from jevify.engine.predict import Recipe, Tier0Engine
    from jevify.engine.readout import HFScorer
    from jevify.engine.template import TEV1_ASSISTANT_PREFIX, render

    sc = HFScorer(a.model, revision=a.revision, dtype=torch.bfloat16, batch_size=8)
    eng = Tier0Engine(sc, Recipe(prompt="tev1", permutations=1))
    state, q = EXAMPLES["charge-dispute"]
    rd = render(state, q, fmt="tev1", identifiers=sc.identifiers())
    prefix = eng._prefix(rd)
    print("[prompt] tail:", repr(prefix[-120:]))
    if not prefix.endswith(TEV1_ASSISTANT_PREFIX):
        print("[prompt] FAIL: not the non-thinking assistant turn"); ok = False
    letters = list(string.ascii_uppercase[:24])
    tok = sc.tokenize(prefix, letters)
    plain = [sc.tokenizer.convert_tokens_to_ids(c) for c in letters]
    single = tok.single_token and [c[0] for c in tok.cand_ids] == plain
    print(f"[prompt] A-X single tokens after the prefix, equal to the plain ids: {single}")
    if not single:
        print("[prompt] FAIL: answer letters are not the trained tokens"); ok = False
    big = {"type": "choice", "instructions": "Pick one.", "criteria": {f"k{i}": f"option {i}" for i in range(151)}}
    rbig = render("x", big, fmt="tev1", identifiers=sc.identifiers())
    tb = sc.tokenize(eng._prefix(rbig), rbig.candidates)
    print(f"[prompt] K=151: {sum(len(c) == 1 for c in tb.cand_ids)}/151 identifiers single-token "
          f"(beyond X Tev1 never saw a label; reported separately)")

    # 4. Together's examples
    import glob
    import os

    cases = dict(EXAMPLES)
    if a.examples:
        for path in sorted(glob.glob(os.path.join(a.examples, "*.json"))):
            d = json.load(open(path, encoding="utf-8"))
            name = os.path.basename(path)[:-5]
            cases[name] = (d["state"], {"type": "choice", "instructions": d["question"],
                                         "criteria": {o["key"]: o["description"] for o in d["options"]}})
    from jevify.bench.record import BenchRecord

    recs = [BenchRecord(id=f"ex/{k}", source="ex", primitive="choice", split="test", state=s, question=qq,
                        label=next(iter(qq["criteria"]))) for k, (s, qq) in cases.items()]
    for r, p in zip(recs, eng.score_records(recs, batch=8, want_prior=False)):
        top = sorted(p.probabilities.items(), key=lambda kv: -kv[1])
        print(f"[example] {r.id}: {p.answer}  " + ", ".join(f"{k} {v:.3f}" for k, v in top))
    print("CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
