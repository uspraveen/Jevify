| model | tier | macro acc | macro ECE | macro Brier | sel@90 | choice acc | score acc | noul acc | TVD→human | GPU | test cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Jev 1.13.0 (TypeSafe API)** | API | 0.733 | 0.113 | 0.349 | 0.760 | 0.770 | 0.503 | 0.881 | 0.432 |  |  |
| Qwen/Qwen3.5-9B (readout LoRA + coherence) | Readout FT | 0.763 | 0.056 | 0.299 | 0.791 | 0.791 | 0.554 | 0.906 | 0.301 |  |  |
| Qwen/Qwen3.5-9B (readout LoRA) | Readout FT | 0.759 | 0.052 | 0.302 | 0.787 | 0.790 | 0.541 | 0.905 | 0.314 |  |  |
| Qwen/Qwen3.5-4B (readout LoRA + coherence) | Readout FT | 0.751 | 0.058 | 0.316 | 0.779 | 0.771 | 0.555 | 0.893 | 0.303 |  |  |
| Qwen/Qwen3.5-4B-Base (readout LoRA + coherence) | Readout FT | 0.747 | 0.058 | 0.318 | 0.777 | 0.771 | 0.535 | 0.897 | 0.314 |  |  |
| Qwen/Qwen3.5-4B (Tier 2 residual) | Tier 2 residual | 0.747 | 0.110 | 0.342 | 0.774 | 0.770 | 0.529 | 0.903 | 0.347 | A40 |  |
| Qwen/Qwen3.5-4B (readout LoRA) | Readout FT | 0.743 | 0.059 | 0.321 | 0.772 | 0.766 | 0.530 | 0.896 | 0.326 |  |  |
| google/gemma-4-E4B-it (readout LoRA + coherence) | Readout FT | 0.742 | 0.053 | 0.322 | 0.772 | 0.759 | 0.546 | 0.889 | 0.297 |  |  |
| Qwen/Qwen3.5-4B-Base (readout LoRA) | Readout FT | 0.741 | 0.064 | 0.324 | 0.770 | 0.768 | 0.522 | 0.893 | 0.324 |  |  |
| google/gemma-4-E4B-it (readout LoRA) | Readout FT | 0.737 | 0.057 | 0.330 | 0.766 | 0.754 | 0.531 | 0.892 | 0.316 |  |  |
| Qwen/Qwen3.5-4B (Tier 2 residual, lr 3e-05) | Tier 2 residual | 0.734 | 0.096 | 0.342 | 0.762 | 0.761 | 0.518 | 0.884 | 0.337 | A40 |  |
| google/gemma-4-12B-it | Tier 0 | 0.716 | 0.086 | 0.359 | 0.740 | 0.749 | 0.507 | 0.855 | 0.405 | A40 |  |
| google/gemma-4-E4B (readout LoRA) | Readout FT | 0.708 | 0.061 | 0.359 | 0.736 | 0.717 | 0.502 | 0.874 | 0.335 |  |  |
| Qwen/Qwen3.5-2B (readout full fine-tune + coherence) | Readout FT | 0.704 | 0.056 | 0.358 | 0.735 | 0.718 | 0.510 | 0.853 | 0.307 |  |  |
| togethercomputer/Tev1-4B-experimental (Tev1 prompt) | External | 0.703 | 0.086 | 0.361 | 0.732 | 0.717 | 0.512 | 0.850 | 0.394 | A40 |  |
| Qwen/Qwen3.5-2B (readout full fine-tune) | Readout FT | 0.703 | 0.056 | 0.358 | 0.734 | 0.715 | 0.509 | 0.855 | 0.315 |  |  |
| Qwen/Qwen3.5-2B-Base (readout LoRA + coherence) | Readout FT | 0.702 | 0.051 | 0.365 | 0.731 | 0.714 | 0.509 | 0.852 | 0.320 |  |  |
| Qwen/Qwen3.5-2B (readout LoRA + coherence) | Readout FT | 0.702 | 0.054 | 0.364 | 0.729 | 0.718 | 0.496 | 0.858 | 0.315 |  |  |
| Qwen/Qwen3.5-2B (readout LoRA) | Readout FT | 0.701 | 0.051 | 0.362 | 0.730 | 0.713 | 0.508 | 0.849 | 0.324 |  |  |
| Qwen/Qwen3.5-4B (Tier 1 residual) | Tier 1 residual | 0.698 | 0.089 | 0.378 | 0.729 | 0.699 | 0.507 | 0.862 | 0.360 | A100-80GB | $1.28 |
| Qwen/Qwen3.5-2B-Base (readout LoRA) | Readout FT | 0.696 | 0.053 | 0.365 | 0.725 | 0.714 | 0.487 | 0.852 | 0.342 |  |  |
| Qwen/Qwen3.5-2B (Tier 2 residual, lr 3e-05, soft labels) | Tier 2 residual | 0.694 | 0.105 | 0.390 | 0.724 | 0.697 | 0.497 | 0.859 | 0.364 | A40 |  |
| togethercomputer/Tev1-4B-experimental (Jevify prompt) | External | 0.690 | 0.088 | 0.369 | 0.718 | 0.714 | 0.487 | 0.834 | 0.408 | A40 |  |
| Qwen/Qwen3.5-2B (Tier 2 residual, lr 3e-05) | Tier 2 residual | 0.690 | 0.103 | 0.389 | 0.720 | 0.693 | 0.510 | 0.840 | 0.332 | A40 |  |
| Qwen/Qwen3.5-9B | Tier 0 | 0.689 | 0.092 | 0.379 | 0.717 | 0.716 | 0.503 | 0.815 | 0.423 | A40 |  |
| Qwen/Qwen3.5-2B (Tier 2 residual) | Tier 2 residual | 0.685 | 0.116 | 0.410 | 0.713 | 0.671 | 0.508 | 0.854 | 0.370 | A40 |  |
| Qwen/Qwen3.5-2B (Tier 2 residual, soft labels) | Tier 2 residual | 0.674 | 0.116 | 0.412 | 0.703 | 0.682 | 0.471 | 0.837 | 0.433 | A40 |  |
| Qwen/Qwen3.5-4B (Tev1 prompt) | Tier 0 | 0.673 | 0.104 | 0.402 | 0.702 | 0.687 | 0.499 | 0.805 | 0.421 | A40 |  |
| Qwen/Qwen3.5-4B | Tier 0 | 0.662 | 0.090 | 0.401 | 0.689 | 0.687 | 0.468 | 0.796 | 0.438 | A100-80GB | $0.96 |
| google/gemma-4-E4B-it | Tier 0 | 0.658 | 0.092 | 0.398 | 0.682 | 0.700 | 0.432 | 0.798 | 0.434 | A100-80GB | $1.00 |
| Qwen/Qwen3.5-2B (Tier 1 residual) | Tier 1 residual | 0.632 | 0.069 | 0.445 | 0.657 | 0.596 | 0.449 | 0.835 | 0.374 | A100-80GB | $0.67 |
| IFM/K2-Horizon-7B | Tier 0 | 0.623 | 0.093 | 0.436 | 0.647 | 0.617 | 0.409 | 0.813 | 0.434 | A40 |  |
| Qwen/Qwen3.5-2B (Tier 1 replace) | Tier 1 replace | 0.599 | 0.083 | 0.475 | 0.621 | 0.567 | 0.378 | 0.828 | 0.416 | A100-80GB | $0.62 |
| google/gemma-4-E2B-it | Tier 0 | 0.591 | 0.123 | 0.484 | 0.612 | 0.594 | 0.440 | 0.716 | 0.492 | L4 | $0.75 |
| Qwen/Qwen3.5-2B | Tier 0 | 0.577 | 0.089 | 0.485 | 0.599 | 0.545 | 0.424 | 0.750 | 0.458 | A100-80GB | $0.66 |
| HuggingFaceTB/SmolLM3-3B | Tier 0 | 0.552 | 0.111 | 0.519 | 0.570 | 0.504 | 0.401 | 0.744 | 0.460 | A100-80GB | $0.64 |
| Qwen/Qwen3.5-0.8B | Tier 0 | 0.526 | 0.112 | 0.542 | 0.544 | 0.436 | 0.388 | 0.759 | 0.440 | L4 | $0.55 |
| Qwen/Qwen3.5-0.8B-Base | Tier 0 | 0.466 | 0.105 | 0.582 | 0.478 | 0.326 | 0.412 | 0.692 | 0.452 | L4 | $0.50 |
| IFM/K2-Horizon-0.9B | Tier 0 | 0.448 | 0.119 | 0.589 | 0.461 | 0.344 | 0.343 | 0.670 | 0.479 | L4 | $0.35 |
| google/gemma-4-E2B | Tier 0 | 0.396 | 0.115 | 0.609 | 0.404 | 0.247 | 0.382 | 0.600 | 0.496 | L4 | $0.69 |
| CLM-v0.1-8B (Contrastive-LM, self-hosted) | External | 0.340 | 0.338 | 0.814 | 0.349 | 0.213 | 0.234 | 0.593 | 0.669 |  |  |
