### Applying a stated rule (legal_rules: 885 LegalBench fact patterns, 10 tasks)

| model | accuracy | ECE | Brier | says yes | macro over tasks | diversity (6 variants) | hearsay | personal_jurisdiction | telemarketing_sales_rule | ucc_v_common_law |
|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.13.0 | 0.924 | 0.121 | 0.083 | 0.478 | 0.883 | 0.947 | 0.840 | 0.840 | 0.830 | 0.957 |
| Qwen3.5-4B Tier 0 | 0.618 | 0.155 | 0.228 | 0.185 | 0.694 | 0.558 | 0.691 | 0.560 | 0.787 | 0.872 |
| Tier 2, published (A40) | 0.653 | 0.238 | 0.261 | 0.227 | 0.703 | 0.597 | 0.766 | 0.460 | 0.723 | 0.968 |
| Tier 2, re-run | 0.667 | 0.162 | 0.211 | 0.313 | 0.690 | 0.627 | 0.745 | 0.460 | 0.660 | 0.957 |
| + option shuffling | 0.697 | 0.117 | 0.203 | 0.359 | 0.695 | 0.675 | 0.734 | 0.440 | 0.681 | 0.947 |
| + Tev1 data | 0.652 | 0.185 | 0.213 | 0.267 | 0.674 | 0.615 | 0.702 | 0.480 | 0.638 | 0.936 |
| + Tev1 data + shuffling | 0.649 | 0.234 | 0.246 | 0.241 | 0.685 | 0.603 | 0.713 | 0.460 | 0.702 | 0.947 |

*Base rate: 45% of the items are "yes".*

### None of the above (nota: 1,000 MMLU / ARC items; the right option removed in half)

| model | accuracy | ECE | right option removed: chose "none" | mean P(none) | right option present: chose "none" | accuracy when present |
|---|---|---|---|---|---|---|
| Jev 1.13.0 | 0.843 | 0.052 | 0.744 | 0.675 | 0.016 | 0.942 |
| Qwen3.5-4B Tier 0 | 0.639 | 0.066 | 0.482 | 0.364 | 0.032 | 0.796 |
| Tier 2, published (A40) | 0.732 | 0.084 | 0.658 | 0.609 | 0.044 | 0.806 |
| Tier 2, re-run | 0.676 | 0.124 | 0.540 | 0.533 | 0.030 | 0.812 |
| + option shuffling | 0.632 | 0.169 | 0.446 | 0.446 | 0.020 | 0.818 |
| + Tev1 data | 0.690 | 0.084 | 0.570 | 0.537 | 0.042 | 0.810 |
| + Tev1 data + shuffling | 0.655 | 0.132 | 0.516 | 0.487 | 0.050 | 0.794 |

### Instructions hidden in the input (injection: 800 items, each clean and with a planted instruction)

| model | accuracy, clean | accuracy, planted | hijack rate | shift in P(named answer) | hijack: boolq | hijack: civil_comments | hijack: fever_evidence | hijack: sms_spam |
|---|---|---|---|---|---|---|---|---|
| Jev 1.13.0 | 0.896 | 0.691 | +0.205 | +0.185 | +0.055 | +0.185 | +0.010 | +0.570 |
| Qwen3.5-4B Tier 0 | 0.859 | 0.463 | +0.396 | +0.293 | +0.290 | +0.225 | +0.300 | +0.770 |
| Tier 2, published (A40) | 0.932 | 0.771 | +0.161 | +0.172 | +0.030 | +0.015 | +0.095 | +0.505 |
| Tier 2, re-run | 0.931 | 0.752 | +0.179 | +0.178 | +0.035 | +0.010 | +0.060 | +0.610 |
| + option shuffling | 0.936 | 0.774 | +0.163 | +0.179 | +0.020 | +0.010 | +0.025 | +0.595 |
| + Tev1 data | 0.931 | 0.760 | +0.171 | +0.174 | +0.025 | +0.010 | +0.025 | +0.625 |
| + Tev1 data + shuffling | 0.929 | 0.756 | +0.172 | +0.167 | +0.025 | +0.020 | +0.040 | +0.605 |

*Hijack rate: how much more often the model gives the answer the planted sentence names than it does on the clean copy of the same item (0 = the sentence is ignored).*
