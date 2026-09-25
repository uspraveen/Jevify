# Post-training study: what each training stage does to a decision readout, and readout fine-tuning

The results behind [FINDINGS §17–18](../../docs/FINDINGS.md#17-what-post-training-does-to-a-decision-readout):
five model families scored at every published training stage, and readout fine-tunes (LoRA and full)
with and without a coherence penalty — every model measured by the same battery.

## Files

| file | contents |
|---|---|
| `pt.md` | every table: RQ1–2 (accuracy, ECE, Brier, held-out, TVD to human labels, fitted temperatures), RQ3 (order, primitive, distractor and tag invariance, cardinality), RQ4 (coherence), out of distribution, seed means |
| `pt.json` | the same numbers, one object per model (`label`, `family`, `size`, `stage` and every metric) |
| `models/<model>/` | per model: `test_metrics.json` (per jev-bench source), `recipe.json` (the fitted Tier 0 recipe), `coherence.json`, `probes.json`, `tags.json` where measured, and `entry.json` (label, family, stage) |

Predictions are not in git (they are large); every published fine-tune carries its own `results/` on the Hub.

## The battery

Each model is read as a Tier 0 decision model (one forward pass, the probability of every allowed answer at the
answer position), and a recipe — option-order permutations, a temperature per primitive, a Noul bias — is fitted on
jev-bench validation (200 records per source) and applied to everything below.

- **jev-bench** — 22 configs, 22,773 test records; six held-out sources (`clinc150`, `arc_challenge`, `yelp5`,
  `measuring_hate_speech`, `fever_evidence`, `strategyqa_grounded`) never trained on by any fine-tune here.
- **Coherence** — 4,749 automatically built question families (19,595 questions: the options of a Choice as yes/no
  questions, the negation of a Noul, the thresholds of a Score), 150 test records per source plus all of ChaosNLI.
  Sure loss = squared distance of the family's answers to the nearest coherent set (de Finetti); 0 is coherent.
- **Jev's probe suite** (FINDINGS §2) — option order, distractors, the same question as Noul vs Choice or Score vs
  Choice, opaque keys, cardinality from K=2 to the source's maximum.
- **Tag test** — option tags A–J replaced by shifted letters, lower case, digits (0- and 1-based) and symbols;
  TVD and top-answer flips against the default tags.
- **Three new tests** (FINDINGS §14) and **three community benchmarks** (FINDINGS §12).

## Models

| family | checkpoints (commit) | prompt |
|---|---|---|
| Qwen3.5 | 2B-Base b1485b2f, 2B 15852e8c, 4B-Base 1001bb4d, 4B 851bf6e8, 9B-Base 68c46c4b, 9B c2022362 | the instruct template (the base checkpoints ship it) |
| Qwen3-4B | Base 906bfd4b, Instruct-2507 cdbee75f, Thinking-2507 768f209d | own template |
| Gemma-4 | E2B d29ff6b4, E2B-it 3e22461f, E4B 411aa17b, E4B-it ee0ef602 | no template for all four (the bases have none); the instruct models also in their template |
| Llama-3.1 → Tülu 3 | Llama-3.1-8B via `unsloth/Meta-Llama-3.1-8B` e9a141a2 (safetensors sha256-identical to the gated `meta-llama/Llama-3.1-8B` d04e592b), Tülu-3-8B-SFT f2a0b46b, -DPO a7beb67e, Tülu-3-8B (RLVR) 66694379 | base and SFT with no template; SFT, DPO, RLVR in Tülu's template |
| SmolLM3-3B | Base d78a42f7; `SmolLM3-3B-checkpoints` it-mid-training 0485ec16, it-SFT f6ddaa5f, it-soup-APO cfb32d50; SmolLM3-3B a07cc9a0 | no template for all five; mid-training, SFT, APO and final also in their template |
| references | Jev 1.13.0 (TypeSafe API); Tev1-4B-experimental 0b7becf0 in its own prompt; the published Jevify 4B Tier 2 | — |

## Readout fine-tuning

Trained on the model's own decision readout with the primitive's proper scoring rule, options shuffled per family,
on the train splits of the 16 non-held-out jev-bench sources (5,885 families, at most 400 records per source);
LoRA rank 16 (Gemma: decoder layers only), lr 3e-5, two epochs, best epoch by validation loss. The **coherence
arm** adds the family's sure loss (weight 1). The **full fine-tune** trains every weight (fp32 master weights, bf16
autocast, gradient checkpointing) with the learning rate chosen on validation from 1e-5, 3e-6, 1e-6 and 3e-7 (1e-6).
Seeds: three at 2B and two at 4B for the jev-bench and coherence numbers; one for everything else.

Each fine-tune is published as a loadable model with its recipe, its adapter (merged at load) or weights, a
`results/` folder and a reproduction check (FINDINGS §18, *Published models*).

## Where it ran

Tier 0 jev-bench scoring for Tülu 3, Qwen3.5-9B-Base, Llama-3.1-8B, Tülu-3-8B-SFT (no template) and the SmolLM3
ladder on rented A100-80GB GPUs (Gemma-4-E2B-it without a template on an L4); everything else — every other
battery part and every fine-tune — on L40S and A40 GPUs. One run (SmolLM3 APO, no template) was interrupted after
10,208 of 22,773 test records and resumed on an L40S with the same code, skipping the records already scored.
A re-run of the 2B supervised LoRA (seed 0) reproduced the original to 0.001 accuracy.
