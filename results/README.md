# results/

Everything a claim in this project rests on, in one tree. The same tree is published under
[`Praveenrajus/jev-bench`](https://huggingface.co/datasets/Praveenrajus/jev-bench/tree/main/results);
the two are identical except that the raw `test_predictions.jsonl` files (16–70 MB each) live only
on the Hub. **Each figure exists in exactly one place: next to the data that produced it.**

| path | what it is | produced by |
|---|---|---|
| `jev-1.13.0/` | The Jev baseline: predictions on all 22,773 test records, per-config metrics, [label audit](jev-1.13.0/README.md), six diagnostic figures | `jevify-run api` → `report` |
| `jev-1.13.0/probes/` | [Behavioral probes](jev-1.13.0/probes/README.md) (cardinality, order, opaque keys, distractors, primitive swap), 14,800 requests, and `probe_cardinality.png` | `jevify-bench probe` |
| `<run>/` | One Jevified checkpoint: `run.json` (what ran), `recipe.json` (calibration fitted on validation only), `test_predictions.jsonl`, `test_metrics.json`, `test_report.md`, `ablation.md` | `scripts/process_run.py` / `process_tier1.py` |
| `<run>/figures/` | That model's six diagnostics: `calibration_map`, `reliability`, `risk_coverage`, `human_vs_model`, `vs_cardinality`, `latency` (Tier 0 only) | `jevify.bench.figures.make_all` |
| `<run>/vs_jev/` | That model against Jev, config by config: `compare.md` + accuracy / ECE bar charts | `jevify-run compare` |
| `leaderboard/` | Every model on one table (`leaderboard.md/json`), one map (`models_map.png`), and per-config heatmaps of accuracy and ECE across all models | `scripts/leaderboard.py` |
| `figures/` | Cross-cutting findings that aggregate *across* runs: `tier1_story`, `tier1_per_source`, `instruct_vs_base`, `recipe_ladder`, `confidence_vs_agreement` | `scripts/make_figures.py` |
| `qwen3vl-2b/` | Vision Tier 0 (POPE / A-OKVQA / AI2D, 4,244 records), recipe-fitted on validation splits (`recipe.json`, `ablation.md`), with a text-only records manifest since images do not round-trip through the per-source layout; published as `Praveenrajus/jevify-qwen3-vl-2b` | `python -m jevify.train vision`, `jevify-run recipe`, `scripts/publish_recipe.py` |
| `latency/` | Single-request latency vs answer-set size and batched throughput for four sizes × three tiers on one A40, against Jev's measured round trip | `scripts/latency.py` |
| `jev-latency-probe/` | Controlled probes of the Jev API with the server's own clock: fixed floor + per-token cost; options, questions and caching isolated | `scripts/probe_jev_latency.py` |
| `latency/latency_vision.*` | Vision serving latency on one A40: per-source p50/p90, input and image tokens, batched throughput; `latency_vision_multi.json` times several questions about one image | `scripts/latency_vision.py` |
| `vision-budget/` | The encoder budget sweep: `results.json`, table, `vision_budget.png` | `scripts/vision_budget.py` |

Run ids: `<model>` is Tier 0; `<model>-t1` Tier 1 with heads replacing the LM score; `<model>-t1r`
Tier 1 residual; `<model>-t2` Tier 2 (LoRA + residual heads); `qwen3-vl-2b-px<N>` a vision run
at a pixel budget of N 28×28 patches.

Every figure carries its model, the bench version and its generation date in the footer, so a
screenshot can always be traced back here.
