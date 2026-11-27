# Oxta-Decision-115M (`NSOS-Jev-LiquidM3-v2.2`) — Jevify Benchmark Entry

**External Self-Hosted System One Model (`115.3M` params, trained 100% from scratch on physical TPU `v6e-1`)**

- **Training Data:** `975,430` real public examples across 21 `Praveenrajus/jev-bench` `train` splits (`chaosnli` 100% zero-shot held-out) + 15 deduplicated Hugging Face decision datasets (`48,704` val+test fingerprints excluded; zero synthetic LLM data, zero pretrained weights).
- **Architecture:** 8-block Hybrid Port-Hamiltonian Liquid-Cayley SSM + Bidirectional RoPE Self-Attention + symmetric cross-attended option scoring (`weights_fp16.npz` SHA-256 `a0681db5ba06e25c3b90f42925c0e0955a8ec08e9152335218f47776124fec99`).
- **Exact Symmetry:** `1.11e-16` max probability drift under random option permutation (`0.0` at 4-decimal wire rounding).
- **Physical Inference Latency:** `0.727 ms` median batch-1 on TPU `v6e-1` (`p95 = 0.757 ms`).

## Official `jev-bench` Test Summary (`22,773` Records, Exact `jevify/bench/metrics.py` Formula)

| Split / Primitive | Acc | ECE *(15 bins)* | Brier | Sel@90 | TVD→human |
|---|---:|---:|---:|---:|---:|
| **Macro (`22` tasks)** | **0.4900** | **0.0787** | **0.5421** | **0.5064** | **0.3673** *(4 soft tasks)* |
| `choice` | 0.3822 | 0.0704 | 0.7358 | 0.4018 | — |
| `score` | 0.4670 | 0.0624 | 0.6393 | 0.4839 | — |
| `noul` | 0.6485 | 0.1033 | 0.2099 | 0.6601 | — |
| `chaosnli` *(Zero-Shot, `1,599`)* | 0.4334 | 0.0287 | 0.2098 | 0.4496 | **0.3357** |
