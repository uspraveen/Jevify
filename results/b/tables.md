### Applying a stated rule (legal_rules: 885 LegalBench fact patterns, 10 tasks)

| model | accuracy | ECE | Brier | says yes | macro over tasks | diversity (6 variants) | hearsay | personal_jurisdiction | telemarketing_sales_rule | ucc_v_common_law |
|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.13.0 | 0.924 | 0.121 | 0.083 | 0.478 | 0.883 | 0.947 | 0.840 | 0.840 | 0.830 | 0.957 |
| Qwen3.5-4B Tier 0 | 0.618 | 0.155 | 0.228 | 0.185 | 0.694 | 0.558 | 0.691 | 0.560 | 0.787 | 0.872 |
| Qwen3.5-9B Tier 0 | 0.699 | 0.073 | 0.186 | 0.391 | 0.738 | 0.665 | 0.702 | 0.620 | 0.787 | 0.915 |
| Tev1-4B + recipe (Tev1 prompt) | 0.678 | 0.195 | 0.227 | 0.268 | 0.696 | 0.640 | 0.777 | 0.500 | 0.617 | 0.947 |
| Jevify 4B Tier 2 (lr 3e-5) | 0.653 | 0.238 | 0.261 | 0.227 | 0.703 | 0.597 | 0.766 | 0.460 | 0.723 | 0.968 |

*Base rate: 45% of the items are "yes".*

### None of the above (nota: 1,000 MMLU / ARC items; the right option removed in half)

| model | accuracy | ECE | right option removed: chose "none" | mean P(none) | right option present: chose "none" | accuracy when present |
|---|---|---|---|---|---|---|
| Jev 1.13.0 | 0.843 | 0.052 | 0.744 | 0.675 | 0.016 | 0.942 |
| Qwen3.5-4B Tier 0 | 0.639 | 0.066 | 0.482 | 0.364 | 0.032 | 0.796 |
| Qwen3.5-9B Tier 0 | 0.746 | 0.200 | 0.676 | 0.447 | 0.078 | 0.816 |
| Tev1-4B + recipe (Tev1 prompt) | 0.723 | 0.114 | 0.702 | 0.528 | 0.140 | 0.744 |
| Jevify 4B Tier 2 (lr 3e-5) | 0.732 | 0.084 | 0.658 | 0.609 | 0.044 | 0.806 |

### Instructions hidden in the input (injection: 800 items, each clean and with a planted instruction)

| model | accuracy, clean | accuracy, planted | hijack rate | shift in P(named answer) | hijack: boolq | hijack: civil_comments | hijack: fever_evidence | hijack: sms_spam |
|---|---|---|---|---|---|---|---|---|
| Jev 1.13.0 | 0.896 | 0.691 | +0.205 | +0.185 | +0.055 | +0.185 | +0.010 | +0.570 |
| Qwen3.5-4B Tier 0 | 0.859 | 0.463 | +0.396 | +0.293 | +0.290 | +0.225 | +0.300 | +0.770 |
| Qwen3.5-9B Tier 0 | 0.865 | 0.566 | +0.299 | +0.227 | +0.115 | +0.235 | +0.195 | +0.650 |
| Tev1-4B + recipe (Tev1 prompt) | 0.914 | 0.776 | +0.138 | +0.141 | +0.040 | +0.110 | +0.015 | +0.385 |
| Jevify 4B Tier 2 (lr 3e-5) | 0.932 | 0.771 | +0.161 | +0.172 | +0.030 | +0.015 | +0.095 | +0.505 |

*Hijack rate: how much more often the model gives the answer the planted sentence names than it does on the clean copy of the same item (0 = the sentence is ignored).*

| model | hijack toward "no" (e.g. not spam) | hijack toward "yes" (e.g. spam) | real spam that a planted "not spam" got through |
|---|---|---|---|
| Jev 1.13.0 | +0.030 | +0.291 | 0.000 of 30 |
| Qwen3.5-4B Tier 0 | +0.445 | +0.372 | 0.333 of 30 |
| Qwen3.5-9B Tier 0 | +0.266 | +0.315 | 0.267 of 30 |
| Tev1-4B + recipe (Tev1 prompt) | +0.057 | +0.177 | 0.133 of 30 |
| Jevify 4B Tier 2 (lr 3e-5) | +0.076 | +0.203 | 0.000 of 30 |

*Last column: of the real spam messages that carried a planted "not spam" instruction, the share the model called spam when clean but not spam once the sentence was added.*
