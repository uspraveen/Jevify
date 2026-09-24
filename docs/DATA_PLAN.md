# Training data for the held-out gap

*Research note, 24 September 2026. Nothing here has been trained yet; this is the selection and the
rules, written before the experiment so the experiment cannot quietly move them.*

## The gap this is for

On jev-bench our best balanced model (Qwen3.5-4B, Tier 2, LoRA lr 3e-5) ties Jev on macro accuracy
and beats it on calibration, but trails it on the six sources no model here was trained on: 0.765
against 0.835 (FINDINGS 7.7). The losses concentrate in two places: **knowledge** (ARC-Challenge,
MMLU) and **ordinal scales the model never saw**. And the Tier 2 runs are data-starved: 5,885
training records, overfitting after one or two passes (FINDINGS 7.1, 7.6).

Two levers, and data is only one of them. What a model *knows* comes mostly from its base; the MoE
bake-off (a bigger "brain" at small-model latency) is the other lever and is not replaced by this
plan. What data can do is teach the **decision skill across many task types and answer formats**,
and give the model practice *using* the knowledge it has. That is what the selection below targets.

## What "high quality" means here

1. **Human labels, or labels correct by construction.** Model-labelled data is allowed only for
   breadth (Tier 2 below) and never as a calibration target: a frontier model's confidence is the
   overconfident profile we measured in Jev (FINDINGS 1.2), and training toward it imports it.
2. **A license that permits training and releasing the model**: Apache-2.0, MIT, CC-BY, CC-BY-SA,
   ODC-BY. Excluded: non-commercial, "research only", and anything with no license asserted.
   Licenses were read from the Hub card *and*, where the card was empty, the source repository.
3. **Decontaminated** against every jev-bench test and validation split and the three community
   benchmarks (normalized-text hash + 13-gram overlap), with the removed counts reported.
4. **Diversity over volume**: many task types, option counts from 2 to 100+, ordinal scales of
   different lengths and orientations; a cap per task so no single source dominates.
5. **Held-out integrity**: nothing from the families of the six held-out sources (science
   multiple choice like ARC, fact verification like FEVER, StrategyQA, intent routing like CLINC,
   star ratings like Yelp, hate-speech intensity) unless the held-out split is redesigned first.
6. **Natural label distributions**: no class rebalancing (FINDINGS 3.4).

## Recommended sources

### Tier 1: human-labelled core

| source | what it teaches | size | license (where verified) | notes |
|---|---|---|---|---|
| **Super-NaturalInstructions** | decisions across *task types*: 1,616 tasks, 76 task categories, each with a human-written definition | ~5M instances | Apache-2.0 (github.com/allenai/natural-instructions) | Built to measure generalization to unseen tasks; part of FLAN and Tülu. Use tasks with a finite label set, converted to Choice/Noul with the task's own label names and definition. Its official held-out categories give a principled unseen-task evaluation. **The single best lever for "unseen question types".** |
| **tasksource** | 500+ harmonized classification tasks (NLI, stance, sentiment, emotion, …) | millions | CC-BY-4.0 aggregate (github.com/sileod/tasksource); per-task licenses vary, filter per task | The collection behind the widely used zero-shot `deberta-v3-*-tasksource-nli` classifiers. |
| CommonsenseQA | commonsense multiple choice, K=5 | 9.7k train | MIT | |
| QASC | multi-fact science reasoning, K=8 | 8.1k | CC-BY-4.0 (card), Apache-2.0 (repo) | science family overlaps ARC in *domain*; decide with the held-out redesign |
| CosmosQA | reading comprehension with commonsense, K=4 | 25k | CC-BY-4.0 | |
| OpenBookQA | elementary science + an open book, K=4 | 5k | Apache-2.0 (github.com/allenai/OpenBookQA) | science family, as QASC |
| WinoGrande | pronoun resolution, K=2, adversarially filtered | 40k | Apache-2.0 (github.com/allenai/winogrande) | |
| MedMCQA | medical knowledge, K=4 | 183k | Apache-2.0 | cap hard (≤ 5k) |
| MMLU `auxiliary_train` | knowledge MCQ | 100k | MIT | it is ARC + OBQA + RACE + MCTest: drop the ARC part (held out) and RACE (non-commercial) |
| SNLI | NLI with **five annotator labels** on dev/test and a validated share of train | 570k | CC-BY-SA-4.0 | soft labels for the multi-annotated share; not in jev-bench |
| WANLI | NLI, human-revised worker-AI collaboration | 108k | CC-BY-4.0 | |
| **HelpSteer3** | graded human preference, **strength −3…+3**, multiple annotators | ~40k | CC-BY-4.0 | an *unseen ordinal scale* of exactly the kind we lose on |
| Arena human preference 140k | which of two responses a human preferred | 140k | CC-BY-4.0 | Noul / two-option Choice; long states, cap |

### Tier 2: breadth at scale, model-labelled (knowledge only)

| source | size | license | use |
|---|---|---|---|
| Nemotron-CrossThink | 10M+ | CC-BY-4.0 | multi-domain QA from web text, used by NVIDIA for RL; take its multiple-choice part, cap |
| OpenScienceReasoning-2 | 100k+ | CC-BY-4.0 | science MCQ with model-agreed answers; cap |

Rule for Tier 2: never used to fit a recipe, never in an evaluation, and a run that uses it must
not degrade TVD to human distributions (ChaosNLI, Measuring Hate Speech).

### Excluded, and why

| source | reason |
|---|---|
| SciQ, ANLI, WebLINX | non-commercial licenses |
| RACE | "non-commercial research purposes only" |
| OpenAI summarize-from-feedback | no license asserted in the repository |
| PIQA, Social IQa, HellaSwag | no license on the Hub card; would need the upstream terms checked first |
| MMLU-Pro | contains MMLU test questions: contaminates jev-bench's `mmlu` |
| ARC, FEVER / VitaminC, StrategyQA, CLINC, Yelp, Measuring Hate Speech | families of the held-out sources |
| Nemotron-CC "diverse QA" (the CLM pretraining corpus) | gated, NVIDIA's own license; generative QA pairs, not decisions |

## How it would be used

- **Mixture**: Tier 1 at ≤ 2,000 records per task/source after decontamination (≈ 40–60k), Tier 2
  ≤ 20k, the rule-based synthetic data of plan B ≤ 10k, and replay of the current jev-bench train
  split so nothing that works now is forgotten (the CLM report measured replay preventing most
  forgetting: 69.0 → 68.5 with replay, 56.2 without).
- **Evaluation first**: redesign the held-out split to hold out whole task families (Super-NI's
  official held-out categories plus jev-bench's six), so a gain cannot come from seeing a
  neighbour of the test set.
- **Acceptance**: held-out accuracy up by more than the seed spread; ECE, TVD to human
  distributions and ChaosNLI calibration not worse; the community benchmarks not worse, and in
  particular no collapse of the phishing flag rate (Tev1 answered "legitimate" on 1,987 of 2,000
  emails).

## Where this sits against other efforts

- **Together's Tev1**: 37,840 records, 58% generated by code (rule application, routing), plus
  MNLI / BoolQ / Banking77 / AG News / SST-5. Narrow in task type; its rule generators are worth
  reusing (plan B), its public-data share is what this plan broadens.
- **Stanford/NVIDIA CLM**: 60M Nemotron question-answer pairs, then 30M generated hard negatives,
  then agent trajectories with 40% replay, all into a 20M-parameter head on a frozen 8B backbone.
  Volume over curation, for a bi-encoder; the replay and "hard negatives after broad
  pre-training" results carry over.
