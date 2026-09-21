"""Jevify playground: ask a Jevified open model typed questions and see the distributions.

Runs on HF Spaces (ZeroGPU when available, CPU otherwise). The model is loaded lazily and
cached, so the first question pays the download and the rest are fast.
"""
from __future__ import annotations

import json
import os
import time

import gradio as gr

MODELS = {
    "Qwen3.5-4B · Tier 1 (best)": "Praveenrajus/jevify-qwen3.5-4b",
    "Qwen3.5-2B · Tier 1": "Praveenrajus/jevify-qwen3.5-2b",
}
DEFAULT = list(MODELS)[1]

EXAMPLES = [
    [
        json.dumps({"ticket": "I was charged twice for order A-104. Please refund the duplicate.",
                    "order": {"id": "A-104", "charges": [{"amount_usd": 49}, {"amount_usd": 49}]}}, indent=2),
        json.dumps({
            "department": {"type": "choice", "instructions": "Which team should handle `ticket`?",
                           "criteria": {"billing": "Payments, refunds, invoices",
                                        "shipping": "Delivery problems",
                                        "technical": "Bugs and integrations",
                                        "other": "None of the above"}},
            "refund_requested": {"type": "noul", "instructions": "Does `ticket` request a refund?"},
            "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                            "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry"]},
        }, indent=2),
    ],
    [
        json.dumps({"premise": "A button on the Chatterbox page will make this easy, so please do join in.",
                    "hypothesis": "They wanted to make the site very user friendly."}, indent=2),
        json.dumps({"relation": {"type": "choice",
                                 "instructions": "What is the relationship of `hypothesis` to `premise`?",
                                 "criteria": {"entailment": "The hypothesis is definitely true given the premise.",
                                              "neutral": "It might be true; the premise does not settle it.",
                                              "contradiction": "It is definitely false given the premise."}}}, indent=2),
    ],
    [
        "Your comments lack dignity, logic and reason. You have shown you would rather insult than think.",
        json.dumps({"toxic": {"type": "noul", "instructions": "Is this comment toxic?",
                              "criteria": {"true": "Rude, disrespectful, or likely to make someone leave the discussion.",
                                           "false": "Civil, even if critical or blunt."}}}, indent=2),
    ],
]

_cache: dict[str, object] = {}


def _load(repo: str):
    if repo not in _cache:
        from jevify import load_jevified

        _cache[repo] = load_jevified(repo, hf_token=os.environ.get("HF_TOKEN"))
    return _cache[repo]


def _parse_state(text: str):
    text = (text or "").strip()
    if not text:
        raise gr.Error("State is empty — give the model something to read.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text          # a plain string is a valid state


def _render(answers: dict, elapsed_ms: float) -> tuple[str, str]:
    blocks = []
    for qid, a in answers.items():
        if a["type"] == "noul":
            p = a["noul"]
            bar = _bar("yes", p) + _bar("no", 1 - p)
            blocks.append(f"### `{qid}` — noul\n\n**P(yes) = {p:.3f}**\n\n{bar}")
        elif a["type"] == "choice":
            ranked = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])
            bars = "".join(_bar(k, v) for k, v in ranked)
            blocks.append(f"### `{qid}` — choice\n\n**{a['choice']}** · confidence {a['confidence']:.3f}\n\n{bars}")
        else:
            ranked = [(f"{i} · {str(a['legend'][str(i)])[:48]}", a["probabilities"][str(i)])
                      for i in range(len(a["legend"]))]
            bars = "".join(_bar(k, v) for k, v in ranked)
            blocks.append(f"### `{qid}` — score\n\n**{a['score']:.2f}** · confidence {a['confidence']:.3f}\n\n{bars}")
    note = f"\n\n<sub>{elapsed_ms:.0f} ms · probabilities are the model's own, not a parsed string</sub>"
    return "\n\n".join(blocks) + note, json.dumps(answers, indent=2)


def _bar(label: str, p: float) -> str:
    filled = int(round(p * 28))
    return f"`{'█' * filled}{'░' * (28 - filled)}` {p:.3f}  {label}\n\n"


def run(model_key: str, state_text: str, questions_text: str):
    try:
        questions = json.loads(questions_text)
    except json.JSONDecodeError as e:
        raise gr.Error(f"Questions must be a JSON object: {e}")
    if not isinstance(questions, dict) or not questions:
        raise gr.Error("Give at least one question, keyed by a name you choose.")
    model = _load(MODELS[model_key])
    t0 = time.perf_counter()
    answers = model.ask(_parse_state(state_text), questions)
    return _render(answers, (time.perf_counter() - t0) * 1000)


try:                                    # ZeroGPU when the Space has it, plain function otherwise
    import spaces

    run = spaces.GPU(duration=90)(run)
except Exception:
    pass

with gr.Blocks(title="Jevify playground", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Jevify playground\n"
        "A **System One** model doesn't write text. It reads a `state`, answers typed questions, and returns "
        "**calibrated probability distributions** your code can branch on — `choice` (pick one of K), "
        "`score` (place on ordered levels), `noul` (P(yes)).\n\n"
        "These are open models given that interface by [Jevify](https://github.com/uspraveen/Jevify) and scored "
        "against TypeSafe's Jev on [jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench). "
        "The probabilities below are read from the model's own logits — nothing is parsed out of generated text."
    )
    with gr.Row():
        model_key = gr.Dropdown(list(MODELS), value=DEFAULT, label="Model")
    with gr.Row():
        with gr.Column(scale=1):
            state = gr.Code(label="State — JSON object, array, or plain text", language="json", lines=12)
            questions = gr.Code(label="Questions — a JSON map of id → typed question", language="json", lines=14)
            go = gr.Button("Ask", variant="primary")
        with gr.Column(scale=1):
            out_md = gr.Markdown(label="Answers")
            out_json = gr.Code(label="Wire response", language="json", lines=14)
    gr.Examples(EXAMPLES, inputs=[state, questions], label="Try one")
    gr.Markdown(
        "---\n"
        "**Same wire format as TypeSafe's API.** `jevify-serve --model <repo>` exposes `/v1/systemone`, and the "
        "official `typesafe-sdk` works against it by setting `TYPESAFE_BASE_URL`.\n\n"
        "What these models are and how they were measured: "
        "[findings](https://github.com/uspraveen/Jevify/blob/main/docs/FINDINGS.md) · "
        "[benchmark](https://huggingface.co/datasets/Praveenrajus/jev-bench)"
    )
    go.click(run, [model_key, state, questions], [out_md, out_json])

if __name__ == "__main__":
    demo.launch()
