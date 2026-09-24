# CLM-v0.1-8B on jev-bench

[CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) (Stanford / NVIDIA, released 23
September 2026) is a bi-encoder decision model: a frozen Qwen3-8B embeds the state and every candidate
answer separately, a 20M-parameter projection head maps both into one space, and the answer is a
softmax over cosine similarities. It serves the same System One wire format as Jev, so it was scored
here exactly as Jev is — through its own server, as its authors ship it, nothing fitted.

| | accuracy | ECE | Brier | TVD→human |
|---|---|---|---|---|
| CLM-v0.1-8B, jev-bench macro (22 configs, 22,773 records) | 0.340 | 0.338 | 0.814 | 0.669 |
| Jev 1.13.0 | 0.733 | 0.113 | 0.349 | 0.432 |

Per config: [`test_report.md`](test_report.md); against Jev: [`vs_jev/compare.md`](vs_jev/compare.md).
Near chance on the knowledge and many-option configs (MMLU 0.252 with four options; LEDGAR 0.011 with
100); closest to Jev on FEVER (0.881), SMS spam (0.834) and PAWS (0.797). The community benchmarks are
in [`../community/`](../community/README.md) and the discussion in FINDINGS §13.

## How it was served

```bash
git clone https://github.com/Contrastive-LM/CLM && git -C CLM checkout cca045ffdb07b3ebcfe6938537cdeac5e14899c9
pip install --no-deps -e CLM            # into an environment that has vLLM (0.29 here)
export CLM_CKPT_DIR=<dir holding CLM_v0.1-8B.pt>
vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling --enforce-eager \
    --enable-prefix-caching --max-model-len 2048 --gpu-memory-utilization 0.6 --host 127.0.0.1 --port 18090
clm-serve --host 127.0.0.1 --port 18700 --emb-url http://127.0.0.1:18090/v1/embeddings --no-ui

TYPESAFE_API_KEY=local jevify-run api --records <jev-bench root> --out test_predictions.jsonl \
    --model clm-latest --base-url http://127.0.0.1:18700 --concurrency 8
```

Both servers default to every network interface; they were bound to localhost here. One A40; 22,773
requests, 0 errors, median 130 ms client-side at concurrency 8. The request is the documented one:
`{"model": "clm-latest", "state": ..., "questions": {"q": {"type", "instructions", "criteria"}}}`.

## Checked by hand

Because the numbers are far from the release's own claims, they were probed directly on the same
server before being reported:

- `Agent tool call: kubectl get pods -n prod` (read-only) and `Agent tool call: DROP TABLE orders`
  (destructive) both return "privileged" at p = 0.9999 with the default head; the no-head ablation
  `clm-raw` also answers "privileged" (0.57).
- Six MMLU test items: 0 of 6 with the documented request, 1 of 6 with the question text moved into
  the instructions, 0 of 6 with `clm-raw`.
- It answers easy cases: the README's own example ("my invoice was charged twice…") is billing at
  0.985, a CORS error is technical at 0.82.

Its authors evaluate on computer use, games, tool calls and best-of-N verification, none of which
jev-bench covers; this is a measurement of the v0.1 release as a general System One model, not of
the approach.
