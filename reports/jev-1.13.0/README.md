# Baseline: TypeSafe Jev 1.13.0 on jev-bench (test splits)

Run 2026-09-20 via `POST /v1/systemone` (`jev-latest` → `jev-1.13.0`), one request per
record, 22,773 records, 0 errors, median latency 192 ms from a 2-core sandbox in us-east.
Predictions and full metrics (including reliability bins) are archived in the dataset repo
under `results/jev-1.13.0/`; the command was

```bash
jevify-run api    --records data/jev-bench --out preds/jev-1.13.0_test.jsonl --split test --concurrency 4
jevify-run report --records data/jev-bench --preds preds/jev-1.13.0_test.jsonl --md test_report.md --json test_metrics.json
```

## Results

See [`test_report.md`](test_report.md) for the full table. Columns: accuracy of the argmax
(or the 0.5 threshold for noul), top-label ECE (15 equal-width bins), Brier, NLL, selective
accuracy on the most-confident 90% / 50%, AURC, ranked-probability score and MAE (ordinal),
AUROC (noul), and total variation distance to the human label distribution where one exists.

Within-one-level accuracy for the ordinal (score) configs, which the table does not show:
helpsteer2_helpfulness 0.81, helpsteer2_verbosity 0.84, measuring_hate_speech 0.79,
sst5 0.95, stsb 0.95, yelp5 0.99.

## Figures

**Reliability diagrams** — observed accuracy vs. stated confidence, one panel per config (dot size = records in bin).
Diagonal = perfectly calibrated. Flat lines (`go_emotions`, `helpsteer2_verbosity`) mean confidence carries no information.

![reliability](figures/reliability.png)

**Calibration map** — where each config lands on accuracy vs. ECE.

![calibration map](figures/calibration_map.png)

**Risk–coverage** — error rate if you act only on the most-confident fraction of records. This is the curve a
confidence-gated router actually lives on.

![risk coverage](figures/risk_coverage.png)

**Model vs. human probability** on the three calibration-gold configs. Dots are (item, option) pairs; the line is the
binned mean. On ChaosNLI the *average* tracks humans but individual answers are pinned near 0/1; on Civil Comments the
model says ~0.29 "toxic" when zero annotators did; on hate speech the model barely moves as human consensus goes 0→1.

![human vs model](figures/human_vs_model.png)

**Latency** — client-observed per request from a 2-core sandbox (us-east), one question per request.

![latency](figures/latency.png)

## Reading

1. **Crisp tasks are excellent and well calibrated.** Knowledge MCQ (ARC 97.9%, MMLU 92.3%),
   grounded yes/no (FEVER-with-evidence 97.2%, BoolQ 91.7%, StrategyQA-grounded 95.6%) and
   NLI (88.3%) all land with ECE ≤ 0.06 and selective accuracy that rises cleanly with
   confidence. As a router or a guard on well-specified questions, the confidence signal is
   usable as advertised.
2. **Where humans disagree, the probabilities do not track human uncertainty.** On ChaosNLI
   (100 annotators per item) the TVD to the human distribution is 0.33 and ECE is 0.22; on
   Measuring Hate Speech vote shares TVD is 0.43. Jev is overconfident on genuinely ambiguous
   inputs. This is the axis an open System One model should be judged on, because it is the
   one that "calibrated" is supposed to mean.
3. **The System Two gap is measurable**: 78.5% closed-book vs 95.6% grounded on the same
   StrategyQA questions. TypeSafe's jaggedness page predicts exactly this.
4. **LLM-response judging and fine-grained emotion are weak.** HelpSteer2 helpfulness /
   verbosity: 36% / 34% exact level (81% / 84% within one). GoEmotions: 32% on 28 classes with
   ECE 0.35 — confidently wrong.
5. **Civil Comments: 72.9% accuracy on a ~92%-negative set** means Jev flags far more comments
   as toxic than Jigsaw's annotators did (AUROC 0.83: the ranking is fine, the operating point
   is not). Absolute Noul semantics plus a broad `criteria.true` string moved the threshold;
   worth remembering when porting any threshold from one question wording to another.
6. **Large-K routing degrades gracefully** (clinc150 89%, massive 81%, banking77 80%,
   ledgar 75% over 100 legal categories) with ECE around 0.1.

## Caveats

- The API rounds probabilities to 0.01, so a true label reported at 0.00 contributes
  −ln(1e-12) to NLL. NLL is therefore inflated on high-K configs (GoEmotions 9.1); use Brier
  and ECE as the proper scores for rounded outputs.
- Choice `confidence` from the API is the rescaled max-probability; Score `confidence` is
  TypeSafe's undisclosed statistic. Selective-accuracy columns use whatever the API returned.
- Single run; the API is not deterministic (±0.01–0.02 per probability), so treat the third
  decimal as noise.
- Label noise is part of the benchmark by design: SST-5 and Yelp-5 human agreement is
  itself limited, and GoEmotions single-label rows are the least ambiguous subset only.
