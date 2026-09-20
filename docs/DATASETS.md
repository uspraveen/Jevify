# jev-bench: source selection and sampling policy

## What we optimized for

1. **Clean human labels**, ideally with a known annotation process.
2. **Human label distributions where they exist** (ChaosNLI, Civil Comments, Measuring Hate
   Speech): the only way to test whether a model's *probabilities* match human uncertainty
   rather than merely whether its argmax is right.
3. **Coverage of the three primitives across K and domains**: routing with 60–151 options,
   3–5 option MCQ, 3–6 level ordinal scales, absolute yes/no with and without grounding.
4. **Real System One workloads**: support/assistant intent routing, moderation, legal clause
   classification, LLM-response judging, RAG-style grounded yes/no.
5. **Deliberate hard cases**: StrategyQA (closed-book) probes the "System Two" weaknesses
   TypeSafe documents on Jev's jaggedness page; the grounded variant isolates knowledge from
   reasoning.
6. **Permissive, documented licenses**. A few classics (SST, Yelp, GLUE/MNLI) carry
   research-use terms; they are marked in the manifest and can be excluded by config.

## Sources

| config | primitive | K | domain | why it's here | license |
|---|---|---|---|---|---|
| `banking77` | choice | 77 | support | canonical large-K intent routing, short messages | cc-by-4.0 |
| `clinc150` | choice | 151 | assistant | 150 intents **plus an explicit out-of-scope option** (Jev's recommended `other`) | cc-by-3.0 |
| `massive` | choice | 60 | assistant | voice-assistant intents, Amazon | cc-by-4.0 |
| `ledgar` | choice | 100 | legal | contract clauses → 100 provision types; long-tail, domain text | cc-by-4.0 |
| `go_emotions` | choice | 28 | social | fine-grained emotion, single-label rows only | apache-2.0 |
| `mmlu` | choice | 4 | knowledge | knowledge retention across tiers | mit |
| `arc_challenge` | choice | 3–5 | knowledge | variable option count per item | cc-by-sa-4.0 |
| `mnli` | choice | 3 | nli | the classic 3-way judgment; test = validation_matched | other (research) |
| `chaosnli` | choice | 3 | nli | **100 annotators per item — calibration gold** (eval only) | cc-by-sa-4.0 |
| `sst5` | score | 5 | reviews | fine-grained sentiment | unspecified (SST) |
| `yelp5` | score | 5 | reviews | star ratings; naturally noisy ordinal labels | other (Yelp research) |
| `helpsteer2_helpfulness` | score | 5 | llm-judging | rating assistant responses (0–4 Likert) | cc-by-4.0 |
| `helpsteer2_verbosity` | score | 5 | llm-judging | a more objective ordinal attribute | cc-by-4.0 |
| `stsb` | score | 6 | nli | semantic similarity on the official 0–5 scale | cc-by-sa-4.0 |
| `measuring_hate_speech` | score | 3 | safety | **annotator vote shares over 3 levels — calibration gold** | cc-by-4.0 |
| `boolq` | noul | 2 | reading | grounded yes/no over a passage | cc-by-sa-3.0 |
| `fever_evidence` | noul | 2 | fact-checking | claim vs gold evidence; NEI dropped | cc-by-sa-3.0 |
| `paws` | noul | 2 | nli | paraphrase with adversarial lexical overlap | other (free use) |
| `civil_comments` | noul | 2 | safety | **toxicity = share of annotators — calibration gold**; natural ~8% positive | cc0-1.0 |
| `sms_spam` | noul | 2 | messaging | tiny, easy anchor | unknown (UCI) |
| `strategyqa_closed` | noul | 2 | knowledge | implicit multi-hop from world knowledge — a System Two probe | mit |
| `strategyqa_grounded` | noul | 2 | knowledge | same questions with the facts supplied | mit |

## Sampling policy

- **Natural label distributions.** Splits are uniform random samples of the source split
  (seed `20260920`). We deliberately do *not* rebalance: a calibration benchmark that shifts
  base rates makes a calibrated model look miscalibrated. `--stratify` exists for building
  training mixes, never for evaluation.
- **Caps** per source: test ≤ 1000 (2000 for civil_comments, all 3113 for chaosnli),
  validation ≤ 500, train ≤ 8000. Train/validation splits exist so Tier 1/2 recipes and
  temperature scaling have in-distribution data without touching test.
- **Carving.** Sources without a validation split get one carved from train (seeded);
  single-split sources (sms_spam) get test and validation carved.
- **Question wording** follows TypeSafe's guidance: a literal question in `instructions`,
  backtick field paths when the state is an object, descriptive criteria, an explicit
  out-of-scope option where the source has one.

## What is *not* here yet

- TypeSafe's cookbook tasks (skill ranking, RAG passage relevance, date extraction,
  hierarchical taxonomies): a public sample of what their users actually ask. Planned as
  an extra config once mined.
- Non-English data. Jev is English-first; so is v0.1 of the benchmark.
- Any synthetic data. jev-bench is the yardstick no LLM wrote; synthetic data is for training.
