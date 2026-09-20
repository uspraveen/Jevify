# What we verified about TypeSafe's Jev API (jev-1.13.0, probed 2026-09-19)

Jevify mirrors this contract so a Jevified model is a drop-in. Everything below was
measured against the live API unless marked *documented*.

## Contract

- `POST https://api.typesafe.ai/v1/systemone` with `{state, model, questions}`; `state` is a
  string, JSON object or array (text only); `questions` is a map of typed questions.
- `choice`: 2–255 labeled options → `choice`, `probabilities` (sum to 1), `confidence`.
- `score`: 2–10 ordered levels → `score` (probability-weighted level index), `legend`,
  `probabilities`, `confidence`.
- `noul`: → `noul` = P(yes). No confidence field.
- `instructions` is optional on the wire (the SDK's schema allows `None`).
- *Documented:* 64K tokens per request, 32K for state + longest question; $0.042 / Mtok
  input, output free; 250K tok/s and 1200 rpm rate limits; English-first.
- The official `typesafe-sdk` reads `TYPESAFE_BASE_URL`, so a compatible server needs no
  client changes. Its wire schema (v0.7.0) matches the docs; there are no hidden parameters.

## Behaviour

- Probabilities are rounded to 0.01 and every distribution sums to exactly 1.00.
- **Not deterministic**: identical requests jitter by ±0.01–0.02.
- **Choice confidence is exactly `(p_max − 1/K) / (1 − 1/K)`** on every sample we took —
  a rescaled max-probability, not normalized entropy.
- **Score confidence is not reproducible** from the published distribution by any standard
  statistic (a bimodal `[0.54, 0.15, 0.31]` scored 0.0; a K=10 answer with p_max 0.52 scored
  0.80). Jevify defines its own (`jevify.wire.score_confidence`) and says so.
- **Noul is absolute; Choice is relative.** TypeSafe's own docs show the same question as a
  Noul returning 0.22 while the yes/no Choice returns 0.01/0.99, and `P(x) + P(¬x)` summing
  to 1.19. These are separate readouts, and Tier 0 logit readout cannot reproduce Noul's
  semantics without a dedicated head.
- **Questions are isolated**: adding leading questions does not move other answers.
- **Latency is flat in the number of questions**: ~330–530 ms per call from a laptop for
  1–12 questions; 184 ms median in a 160-request run.
- `output_tokens` ≈ 14 + ~8.4 per option for Choice and scales with option *label* length
  only; a constant 17 for Score (any K) and 20 for Noul. Nothing free-form is generated.
  Whether slot values are emitted as tokens or read from a head is not knowable from outside;
  the constant-size Score output and exact-1.00 sums lean toward a head-style readout.

## Known limitations (their "jaggedness" page)

Literal reading; no counting, arithmetic or date comparison; accuracy falls with irrelevant
state ("context rot"); adversarial state can steer answers; no generation. This profile is
what jev-bench's harder configs (StrategyQA closed-book, long-state sources) are there to
measure.

## Training, as publicly described

"Reinforcement learning for calibrated decisions" against proper scoring rules; the model is
**trained entirely on synthetic data** built by a lab that "owns … statistically well-understood
synthetic data" (Almeida to TechCrunch, 2026-09-18). No paper, weights or data released.
