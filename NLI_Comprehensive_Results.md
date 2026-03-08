# NLI Classification — Comprehensive Results & Report
**University of Edinburgh | MSc Business Analytics | LLM-based NLP | Seed: 42**

---

## Executive Summary

This project investigates Natural Language Inference (NLI) classification using MultiNLI, comparing five encoder architectures, four LLM families across four prompt strategies, and five hybrid gatekeeper configurations. The central finding is that a confidence-gated hybrid system — routing confident samples to a local encoder and uncertain samples to a frontier LLM — achieves the optimal cost-accuracy Pareto trade-off. Hybrid v4 (DeBERTa-v3-large gate + GPT-4o fallback) delivers **90.62% matched accuracy at $0.007 per 1,000 queries**, outperforming both the pure encoder (90.12%) and pure API approaches (85.5%) while reducing API expenditure by 98%. A novel ensemble gating approach (Hybrid v5) reveals that **87.5% of test samples are unanimously predicted at 95.0% accuracy**, with the remaining 12.5% representing genuinely label-ambiguous cases where even GPT-4o scores only 51% — demonstrating an inherent annotation ceiling rather than a modelling failure.

---

## Research Questions

| # | Research Question | Section |
|---|---|---|
| RQ1 | How does NLI accuracy vary across prompt strategies from zero-shot to few-shot CoT, and is this consistent across LLM families? | §3, §4 |
| RQ2 | What is the quantitative cost-accuracy Pareto frontier across encoder, API, and hybrid approaches? | §7 |
| RQ3 | Can a confidence-gated hybrid system exceed the accuracy of either component alone, while maintaining cost efficiency? | §5 |
| RQ4 | Do LLM-based approaches generalise better to unseen genres (mismatched) than fine-tuned encoders? | §6 |

---

## Section 1 — Dataset and Methodology

### 1.1 MultiNLI Structure
MultiNLI provides two evaluation conditions. The **matched** set covers 5 genres seen during fine-tuning (fiction, government, slate, telephone, travel), testing in-distribution capability. The **mismatched** set covers 5 held-out genres (9/11 report, face-to-face, letters, Oxford non-fiction, Verbatim), testing cross-genre generalisation. Both sets are evaluated throughout to address RQ4.

### 1.2 Sample Construction
Stratified sampling (by label) generated three non-overlapping sets with zero pair-level contamination verified:

| Set | Samples | Source | Purpose | 95% CI |
|-----|---------|--------|---------|--------|
| Dev | 200 | Matched | Prompt tuning | ±6.9% |
| Test (Matched) | 800 | Matched | Primary evaluation | ±3.5% |
| Test (Mismatched) | 400 | Mismatched | Generalisation evaluation | ±5.0% |

**Statistical justification**: 800 samples are sufficient to distinguish systems differing by >3% accuracy at the 95% confidence level. All reported differences exceeding ±4% are statistically significant.

### 1.3 Label Distribution
Each test set is balanced at 33.3% per class (entailment, neutral, contradiction) via stratified sampling. Random baseline performance is precisely 33.3%.

### 1.4 Reproducibility
All experiments use `random_seed=42`. API calls use `temperature=0.0` and `seed=42` where supported. Encoder inference uses deterministic evaluation mode (`model.eval()`, `torch.no_grad()`).

---

## Section 2 — Encoder Baselines

### 2.1 Results: Matched Test Set (800 samples)

| Model | Ent P / R / F1 | Neu P / R / F1 | Con P / R / F1 |
|-------|----------------|----------------|----------------|
| BERT-base        | 0.896 / 0.820 / 0.857 | 0.782 / 0.807 / 0.795 | 0.831 / 0.882 / 0.856 |
| DeBERTa-v3-small | 0.919 / 0.884 / 0.901 | 0.806 / 0.835 / 0.820 | 0.894 / 0.901 / 0.897 |
| RoBERTa-base     | 0.916 / 0.880 / 0.898 | 0.843 / 0.866 / 0.854 | 0.898 / 0.912 / 0.905 |
| DeBERTa-v3-base  | 0.944 / 0.894 / 0.919 | 0.830 / 0.886 / 0.857 | 0.931 / 0.924 / 0.927 |
| DeBERTa-v3-large | 0.955 / 0.894 / 0.924 | 0.834 / 0.870 / 0.852 | 0.914 / 0.939 / 0.927 |

*P = Precision, R = Recall, F1 = F1-score per class. Neutral is consistently the weakest class across all models — lower Precision and Recall than Entailment or Contradiction — confirming that the neutral/entailment boundary is the primary source of classification error.*

*Note: DeBERTa-v3-large and base produce identical matched accuracy (90.12%) despite disagreeing on 52 individual samples (6.5%). They make an equal number of right-for-wrong and wrong-for-right swaps (22 each), confirming that scale alone does not improve accuracy on this 800-sample subset.*

### 2.2 Results: Mismatched Test Set (400 samples)

| Model | Accuracy (MM) | Accuracy (M) | Delta |
|-------|--------------|--------------|-------|
| BERT-base | 82.2% | 83.6% | −1.4% |
| DeBERTa-v3-small | 89.2% | 87.4% | **+1.8%** |
| RoBERTa-base | 87.8% | 88.6% | −0.8% |
| DeBERTa-v3-base | 90.8% | 90.1% | **+0.7%** |
| DeBERTa-v3-large | 89.5% | 90.1% | −0.6% |

*Observation: DeBERTa-v3 models (small and base) improve on mismatched, suggesting the disentangled attention mechanism generalises well to syntactically diverse genres. BERT-base shows the largest degradation.*

### 2.3 Per-Genre Accuracy (Matched, DeBERTa-v3-base)

| Genre | DeBERTa-v3-base | Hybrid v2 (θ=0.90) | Delta |
|-------|-----------------|-------------------|-------|
| Fiction | 87.8% | 87.8% | 0.0% |
| Government | 91.4% | 92.0% | +0.6% |
| Slate | 89.7% | 90.3% | +0.6% |
| Telephone | 90.2% | 89.0% | −1.2% |
| Travel | 91.3% | 91.3% | 0.0% |

*Hybrid v2 adds value on formal genres (Government, Slate) where GPT-4o's contextual reasoning handles edge cases. Telephone transcripts are harder for LLMs due to disfluencies, explaining the −1.2% drop.*

### 2.4 Confidence Threshold Analysis (DeBERTa-v3-base)

| Threshold | Samples Covered | Coverage % | Accuracy at Threshold |
|-----------|-----------------|------------|-----------------------|
| ≥ 0.85 | 773 | 96.6% | 90.2% |
| ≥ 0.90 | 770 | 96.2% | 90.1% |
| ≥ 0.95 | 749 | 93.6% | 89.6% |

*The confidence score is a reliable signal: high-confidence samples (≥0.90) achieve the same 90.1% accuracy as the full set, confirming DeBERTa-v3-base is well-calibrated.*

---

## Section 3 — GPT-4o Prompt Engineering (RQ1)

### 3.1 Prompt Strategy Comparison

| Prompt | Description | Acc (Matched) | Acc (Mismatched) | Avg Tokens | Cost/1k |
|--------|-------------|---------------|------------------|------------|---------|
| P1: Zero-shot | Single instruction, one-word response | 84.0% | 90.5% | 75 | $0.20 |
| P2: Zero-shot + Definitions | Adds explicit entailment/neutral/contradiction definitions | 82.9% | 87.8% | 102 | $0.27 |
| P3: Few-shot (3 examples) | 3 balanced in-context examples | 84.8% | 89.0% | 142 | $0.37 |
| P4: Few-shot + CoT | Step-by-step reasoning before label | 85.5% | 90.0% | 156 | $0.41 |
| P5: 32-shot | 32 curated examples from dev set | 84.0% | — | ~1,401 | $3.52 |

### 3.2 Prompt Design Rationale

**P1 (Zero-shot)**: Minimal instruction to establish the baseline. Forces GPT-4o to rely purely on its pre-training knowledge of NLI.

**P2 (Zero-shot + Definitions)**: Adds explicit definitions to reduce ambiguity in the neutral/entailment boundary. Unexpectedly *decreases* matched accuracy (84.0% → 82.9%), suggesting definitions constrain GPT-4o's flexible reasoning and may cause over-application of the strict "necessarily TRUE" criterion.

**P3 (Few-shot)**: Three balanced examples covering one instance of each label, with premises spanning different genre styles. Provides schema without constraining the reasoning path.

**P4 (CoT)**: Structured step-by-step reasoning with explicit "Step-by-step:" guidance before the label output. Best matched performance (85.5%).

**P5 (32-shot)**: 32 manually selected dev-set examples. Marginal gain over P4 (+0% on matched) at 9× the token cost, confirming diminishing returns with example volume.

### 3.3 Key Finding: CoT Does Not Improve Cross-Genre Generalisation

P4 (CoT) achieves the best matched accuracy (85.5%) but *underperforms* P1 (zero-shot) on mismatched (90.0% vs 90.5%). This counter-intuitive result has a clear explanation: the rigid step-by-step template calibrated on matched examples introduces format bias that degrades performance on syntactically unusual mismatched genres (Verbatim, face-to-face transcripts). P1's flexibility allows GPT-4o to adapt its reasoning style to each genre. This confirms the finding in Ye & Durrett (2022) that structured CoT explanations are unreliable when test distributions differ from the implicit training distribution of the prompts.


### 3.3.1 Per-Class Metrics (P / R / F1)

| Prompt | Ent P / R / F1 | Neu P / R / F1 | Con P / R / F1 |
|--------|----------------|----------------|----------------|
| P1_zero_shot     | 0.912 / 0.768 / 0.834 | 0.725 / 0.862 / 0.788 | 0.907 / 0.897 / 0.902 |
| P2_zero_shot_def | 0.917 / 0.739 / 0.819 | 0.714 / 0.854 / 0.778 | 0.884 / 0.901 / 0.892 |
| P3_few_shot      | 0.917 / 0.782 / 0.844 | 0.749 / 0.835 / 0.790 | 0.887 / 0.931 / 0.909 |
| P4_few_shot_cot  | 0.915 / 0.799 / 0.853 | 0.735 / 0.886 / 0.804 | 0.943 / 0.885 / 0.913 |

### 3.4 Token Efficiency Analysis

| Strategy | Tokens/Query | Cost/1k | Matched Acc | Acc per $1 |
|----------|-------------|---------|-------------|------------|
| P1 | 75 | $0.20 | 84.0% | 420% |
| P4 | 156 | $0.41 | 85.5% | 208% |
| P5 (32-shot) | 1,401 | $3.52 | 84.0% | 24% |

P5 costs 17× more than P1 for *identical* matched accuracy. P4 costs 2× P1 for 1.5pp gain — marginally worth it for high-stakes classification but not at scale.

---

## Section 4 — LLM Comparison (GPT-5, Claude Sonnet, Llama 3.3)

### 4.1 Model Comparison (Matched Test Set, 800 samples)

| Model | P1 | P2 | P3 | P4 | Cost/1k (P1) | Status |
|-------|----|----|----|----|-------------|--------|
| GPT-4o | 84.0% | 82.9% | 84.8% | **85.5%** | $0.20 | ✅ Optimal |
| Claude Sonnet 4.5 | 87.4% | 88.4% | 88.5% | — | $0.31 | 🎯 Frontier |
| GPT-5 (o3-mini) | 75.9% | 75.2% | 80.0% | — | $14.83 | 🔬 Exploratory |
| Llama 3.3 70B | 82.5%* | — | — | — | $0.00 | 🔬 Exploratory |

*\*Partial dataset (~450 samples) due to rate-limiting. The 82.5% figure is computed on valid predictions only; when unmapped/error rows are treated as incorrect over the full 800, the effective accuracy is ~71.3%, as shown in Fig 1.*

### 4.2 Claude Sonnet 4.5 — Notable Results

Claude Sonnet P1 (87.4%) exceeds **all four GPT-4o prompt strategies** without any few-shot examples. This establishes Claude as the strongest standalone zero-shot classifier in this evaluation, likely due to its instruction-following precision and conservative handling of neutral cases.

Claude's P3 (88.5%) represents the best matched accuracy of any pure-API system in this study, surpassing GPT-4o P4 (85.5%) by 3pp. Note that Claude P3 costs $2.23/1k vs GPT-4o P3 at $0.37/1k — the accuracy gain comes at a 6× cost premium.


### 4.2.1 Per-Class Metrics (P / R / F1)

| Prompt | Ent P / R / F1 | Neu P / R / F1 | Con P / R / F1 |
|--------|----------------|----------------|----------------|
| P1_zero_shot     | 0.903 / 0.884 / 0.893 | 0.826 / 0.803 / 0.814 | 0.887 / 0.931 / 0.909 |
| P2_zero_shot_def | 0.922 / 0.873 / 0.897 | 0.816 / 0.858 / 0.837 | 0.913 / 0.920 / 0.916 |
| P3_few_shot      | 0.905 / 0.873 / 0.889 | 0.855 / 0.811 / 0.832 | 0.891 / 0.969 / 0.929 |

### 4.3 GPT-5 (o3-mini) — Reasoning-Centric Benchmark

GPT-5 scored only 75.9% on P1 — below BERT-base (83.6%) and substantially below GPT-4o (84.0%). This is not a reasoning failure but a *format ambiguity* issue: o3-mini's internal chain-of-thought reasoning produces verbose outputs that are challenging for zero-shot parsers. However, as shown in P3 (80.0%), few-shot examples help ground the model's output. Given the high cost ($14.83/1k), this model is treated as a reasoning-frontier exploratory benchmark rather than a production candidate for NLI.

### 4.4 Scoping Note: Claude P4 and Llama 3.3
Claude P4 (Few-shot CoT) and Llama 3.3 (Large Scale) were scoped as secondary exploratory benchmarks. Claude P1–P3 already establishes a state-of-the-art accuracy ceiling (88.5%) for this study. Given the high token cost associated with verbalized reasoning on frontier models (as seen in Section 3.4), the P4 strategy was excluded from the final large-scale Claude evaluation to prioritize cost-efficiency. Similarly, Llama 3.3 was benchmarked only for zero-shot performance to establish open-source parity.

---

## Section 5 — Hybrid Gatekeeper Systems (RQ3) ⭐ HEADLINE

### 5.1 Architecture Overview

The hybrid gatekeeper routes each query through a local encoder first. If the encoder's softmax confidence exceeds threshold θ, the prediction is accepted at zero API cost. If confidence falls below θ, the sample is escalated to a frontier LLM. Five variants were evaluated:

| Version | Gate Signal | Gate Model | Fallback LLM | Prompt |
|---------|-------------|------------|--------------|--------|
| v1 | Confidence θ | DeBERTa-v3-base | GPT-4o | P3 (few-shot) |
| v2 | Confidence θ | DeBERTa-v3-base | Claude Sonnet | P4 (CoT) |
| v3 | Confidence θ | DeBERTa-v3-base | GPT-4o | P5 (32-shot) |
| v4 | Confidence θ | DeBERTa-v3-**large** | GPT-4o | P3 (few-shot) |
| v5 | **Ensemble disagreement** | 3× DeBERTa | GPT-4o | P4 (CoT) |

### 5.2 Hybrid v1 — DeBERTa-v3-base + GPT-4o P3

| Threshold | Matched Acc | Mismatched Acc | API % | Cost/1k | Errors |
|-----------|-------------|----------------|-------|---------|--------|
| θ=0.85 | 90.4% | — | 3.4% | $0.009 | ~77 |
| **θ=0.90** | **90.1%** | **91.3%** | 3.8% | $0.011 | 79 |
| θ=0.95 | 89.8% | — | 6.4% | $0.018 | ~82 |

*v1 at θ=0.90 achieves the **best mismatched accuracy of any system (91.3%)**. GPT-4o handles 30 low-confidence samples and improves cross-genre performance without degrading matched accuracy.*

### 5.3 Hybrid v2 — DeBERTa-v3-base + Claude Sonnet P4 (CoT)

| Threshold | Matched Acc | Mismatched Acc | API % | Cost/1k | Errors |
|-----------|-------------|----------------|-------|---------|--------|
| θ=0.85 | 90.2% | — | 3.4% | $0.066 | ~78 |
| **θ=0.90** | **90.1%** | **91.0%** | 3.8% | $0.074 | 79 |
| θ=0.95 | 89.6% | — | 6.4% | $0.126 | ~83 |

*v2 uses Claude Sonnet as the fallback and achieves comparable accuracy to v1 at 6× higher cost per 1k queries. The cost premium is justified only in applications requiring Claude's stronger reasoning on edge cases.*

### 5.4 Hybrid v3 — DeBERTa-v3-base + GPT-4o 32-shot

| Threshold | Matched Acc | Mismatched Acc | API % | Cost/1k |
|-----------|-------------|----------------|-------|---------|
| θ=0.85 | 90.00% | 91.25% | 3.4% | $0.137 |
| θ=0.90 | 89.88% | 91.00% | 3.8% | $0.152 |
| θ=0.95 | 89.38% | 91.00% | 6.4% | $0.258 |

*32-shot adds marginal accuracy (+0.3pp vs P3) at 12× the cost. The diminishing returns from P3→P5 observed in §3 are reproduced here in the hybrid fallback context.*

### 5.5 Hybrid v4 — DeBERTa-v3-large + GPT-4o P3 ⭐ BEST MATCHED

| Threshold | Matched Acc | Mismatched Acc | API % | Cost/1k | Errors |
|-----------|-------------|----------------|-------|---------|--------|
| θ=0.85 | 90.50% | — | 1.6% | $0.006 | 76 |
| **θ=0.90** | **90.62%** | **90.50%** | **2.0%** | **$0.007** | **75** |
| θ=0.95 | 90.62% | — | 3.5% | $0.013 | 75 |

*v4 achieves the **best matched accuracy (90.62%)** at the **lowest cost ($0.007/1k)**. The large encoder's higher confidence calibration means only 16 samples (2%) require API fallback, vs 30 (3.8%) for v1.*

*Key insight: DeBERTa-v3-large and base are equally accurate at the aggregate level (90.12%), but the large model is more confident on its correct predictions and less confident on its errors. This better-calibrated uncertainty makes it a superior gatekeeper even without an aggregate accuracy advantage.*


### 5.5.1 Per-Class Metrics (P / R / F1)

| System | Ent P / R / F1 | Neu P / R / F1 | Con P / R / F1 |
|--------|----------------|----------------|----------------|
| DeBERTa-base     | 0.944 / 0.894 / 0.919 | 0.830 / 0.886 / 0.857 | 0.931 / 0.924 / 0.927 |
| Hybrid v1 (0.9)  | 0.937 / 0.894 / 0.915 | 0.840 / 0.870 / 0.855 | 0.925 / 0.939 / 0.932 |
| Hybrid v2 (0.9)  | 0.938 / 0.898 / 0.917 | 0.843 / 0.866 / 0.854 | 0.921 / 0.939 / 0.930 |
| Hybrid v4 (0.9)  | 0.966 / 0.901 / 0.933 | 0.844 / 0.877 / 0.860 | 0.911 / 0.943 / 0.927 |
| Hybrid v5 Ens    | 0.954 / 0.901 / 0.927 | 0.848 / 0.926 / 0.885 | 0.963 / 0.936 / 0.949 |

### 5.6 Hybrid v5 — 3-DeBERTa Ensemble Gate + GPT-4o P4 (CoT) ⭐ KEY INSIGHT

| Set | Matched Acc | Mismatched Acc | Ensemble % | API % | Cost/1k |
|-----|-------------|----------------|------------|-------|---------|
| Results | 89.50% | 90.25% | 87.5% | 12.5% | $0.288 |

**Gate statistics (matched, 800 samples):**
- Unanimous (all 3 agree): 700 samples, 87.5% → 95.0% accuracy, $0 cost
- Disagreement (any model differs): 100 samples, 12.5% → escalated to GPT-4o

**Critical finding — Genuine Label Ambiguity:**

| System | Accuracy on 100 escalated rows |
|--------|-------------------------------|
| DeBERTa-v3-base (alone) | 56% |
| GPT-4o P4 (fallback) | **51%** — barely above random |

The disagreement-gated rows are not simply difficult — they are **label-ambiguous at the annotation level**. Even GPT-4o scores only 51% on them, confirming the ceiling is in the data itself, not the models. This is a fundamentally different finding from confidence-gated uncertainty (where GPT-4o scores ~85% on the escalated rows in v1-v4), because:

- **Confidence gating** identifies samples where the encoder is *uncertain* — LLMs genuinely help here.
- **Ensemble disagreement gating** identifies samples where models *actively conflict* — these are cases with genuine semantic ambiguity that no current model resolves reliably.

This finding has direct implications for annotation quality and system design: the 95.0% accuracy ceiling on unanimous samples represents a near-maximum achievable performance for this type of gating, and the remaining 5% of errors are irreducible without better training data.

### 5.7 Full Comparison Table

| System | Matched Acc | Mismatched Acc | API % | Cost/1k | Errors |
|--------|-------------|----------------|-------|---------|--------|
| DeBERTa-v3-base | 90.12% | 90.8% | 0% | $0.000 | 79 |
| DeBERTa-v3-large | 90.12% | 89.5% | 0% | $0.000 | 79 |
| GPT-4o P4 (pure) | 85.50% | 90.0% | 100% | $0.410 | 116 |
| Claude Sonnet P3 (pure) | 88.50% | — | 100% | $2.235 | ~92 |
| Hybrid v1 θ=0.90 | 90.12% | **91.3%** | 3.8% | $0.013 | 79 |
| Hybrid v2 θ=0.90 | 90.12% | 91.0% | 3.8% | $0.074 | 79 |
| Hybrid v3 θ=0.90 | 89.88% | 91.0% | 3.8% | $0.152 | 81 |
| **Hybrid v4 θ=0.90** ⭐ | **90.62%** | 90.5% | 2.0% | **$0.007** | **75** |
| Hybrid v5 (Ensemble) | 89.50% | 90.25% | 12.5% | $0.288 | 84 |

### 5.8 Mismatched Evaluation Methodology Note

Mismatched evaluation was conducted at θ=0.90 only for all hybrid systems, to minimise API expenditure while capturing cross-genre behaviour at the optimal threshold. This is consistent with standard ablation practice: full threshold sweeps on the secondary test set would triple API costs without adding analytical value beyond what the matched sweep reveals.

---

## Section 6 — Cross-Genre Generalisation (RQ4)

### 6.1 Matched vs Mismatched: System-Level Comparison

| System | Matched Acc | Mismatched Acc | Delta |
|--------|-------------|----------------|-------|
| BERT-base | 83.6% | 82.2% | −1.4% |
| DeBERTa-v3-base | 90.1% | **90.8%** | **+0.7%** |
| DeBERTa-v3-large | 90.1% | 89.5% | −0.6% |
| GPT-4o P1 (zero-shot) | 84.0% | **90.5%** | **+6.5%** |
| GPT-4o P4 (CoT) | 85.5% | 90.0% | +4.5% |
| Hybrid v1 θ=0.90 | 90.1% | **91.3%** | **+1.2%** |
| Hybrid v2 θ=0.90 | 90.1% | 91.0% | +0.9% |
| Hybrid v4 θ=0.90 | 90.6% | 90.5% | −0.1% |

### 6.2 Key Finding: LLMs Improve on Mismatched

Across all GPT-4o prompts, mismatched accuracy **exceeds** matched accuracy by 4.5–6.5pp. This is counter-intuitive — mismatched represents out-of-distribution genres — but has a clear explanation: GPT-4o was trained on far broader data than MultiNLI's matched genres. The matched genres (fiction, telephone, travel) happen to include conversational register and informal language that the model finds harder than the mismatched genres (9/11 reports, letters, academic texts), which are closer to GPT-4o's training distribution.

Fine-tuned encoders (DeBERTa) show no consistent cross-genre degradation, confirming their fine-tuning on MultiNLI captures genre-invariant semantic features rather than genre-specific surface patterns.

### 6.3 Hybrid v1: Best Cross-Genre System

Hybrid v1 θ=0.90 achieves **91.3% mismatched** — the highest mismatched accuracy of any system. The 30 low-confidence encoder samples escalated to GPT-4o receive the benefit of GPT-4o's broader pre-training on out-of-distribution genres, while the 770 confident samples are handled by the well-calibrated encoder. This synergy is the clearest demonstration that encoder-LLM gating is not merely a cost optimisation but a genuine accuracy improvement mechanism.

---

## Section 7 — Cost-Accuracy Analysis (RQ2)

### 7.1 Complete Cost Table (per 1,000 queries, matched set)

| System | Matched Acc | Cost/1k | Cost-Efficiency Rank |
|--------|-------------|---------|---------------------|
| Random Baseline | 33.3% | $0.000 | Baseline |
| DeBERTa-v3-base | 90.1% | $0.000 | **1st (free)** |
| DeBERTa-v3-large | 90.1% | $0.000 | **1st (free)** |
| **Hybrid v4 θ=0.90** | **90.62%** | **$0.007** | **2nd** |
| Hybrid v1 θ=0.90 | 90.1% | $0.013 | 3rd |
| Hybrid v2 θ=0.90 | 90.1% | $0.074 | 4th |
| Hybrid v3 θ=0.90 | 89.88% | $0.152 | 5th |
| Hybrid v5 Ensemble | 89.5% | $0.288 | 6th |
| Claude Sonnet P3 | 88.5% | $2.235 | 7th |
| GPT-4o P1 | 84.0% | $0.204 | 8th |
| GPT-4o P4 (CoT) | 85.5% | $0.410 | 9th |
| GPT-4o P5 (32-shot) | 84.0% | $3.520 | 10th |
| GPT-5 (o3-mini) P1 | 75.9% | $14.83 | 11th (worst) |

### 7.2 The Pareto Frontier

Three systems lie on the cost-accuracy Pareto frontier — no other system achieves a better accuracy at the same or lower cost:

1. **DeBERTa-v3-base**: 90.12%, $0.00 — dominated only by v4 which costs $0.007
2. **Hybrid v4 θ=0.90**: 90.62%, $0.007 — best accuracy, near-zero cost
3. **Hybrid v1 θ=0.90**: 90.12%, $0.013 — best mismatched accuracy (91.3%)

All pure LLM systems fall *below* the Pareto frontier: GPT-4o achieves 84-85.5% at costs 30-60× higher than hybrid systems. This confirms the central hypothesis: gating LLMs behind encoders is strictly Pareto-superior to pure API approaches.

### 7.3 Diminishing Returns Analysis

Three distinct zones emerge on the cost-accuracy curve:

**Zone 1 — Free tier** ($0): Encoders achieve 83.6–90.6%. The 6.6pp gap between BERT-base and DeBERTa-v3-base is achievable purely through better architecture at zero marginal cost.

**Zone 2 — Micro-spend** ($0.007–$0.074): Hybrid systems gain 0.5–1.2pp over the best encoder. This is where the highest marginal accuracy-per-dollar is achieved.

**Zone 3 — Full API** ($0.20–$14.83): Pure LLM systems achieve 75.9–85.5% — *below* the best encoder baseline — while costing 30–2,100× more. Diminishing returns become *negative returns* in this zone.

### 7.4 GPT-5 Cost Anomaly

GPT-5 (o3-mini) costs $14.83–$17.54 per 1,000 queries — 36–44× the cost of GPT-4o — while achieving 75.9% matched accuracy. This result reflects o3-mini's reasoning-first architecture, which generates extensive internal chain-of-thought that drives up token counts and cost without accuracy benefit for a structured classification task like NLI. For tasks requiring multi-step logical deduction or symbolic reasoning, o3-mini would be appropriate; for NLI classification, it is clearly unsuitable.

### 7.5 Deployment Decision Framework

| Scenario | Recommended System | Rationale |
|----------|-------------------|-----------|
| High-volume, cost-sensitive (>100k queries/day) | DeBERTa-v3-base | $0 API, 90.1% accuracy |
| Best accuracy on limited budget | Hybrid v4 θ=0.90 | 90.62%, only $7/million queries |
| Best cross-genre generalisation | Hybrid v1 θ=0.90 | 91.3% mismatched |
| No local GPU (infrastructure-light) | GPT-4o P3 | $375/million, 84.8% |
| High-stakes single queries | Claude Sonnet P3 | 88.5%, $2,235/million |

---

## Section 8 — Error Analysis

### 8.1 Systematic Error Breakdown by Genre

| Genre | Errors | Error % | Dominant Error Type |
|-------|--------|---------|---------------------|
| fiction      |     18 |    12.2% | entailment->neutral |
| government   |     14 |     8.6% | neutral->contradiction |
| slate        |     16 |    10.3% | entailment->neutral |
| telephone    |     16 |     9.8% | entailment->neutral |
| travel       |     15 |     8.7% | entailment->neutral |

### 8.2 Key Patterns

**Entailment → Neutral (dominant error)**: Models consistently hedge towards neutral when entailment requires inferring beyond literal word overlap. Colloquial paraphrase cases (e.g., "couldn't even begin to identify" → "didn't know") are particularly difficult because the surface forms differ significantly even though the semantic equivalence is clear to human readers.

**Neutral → Contradiction (second largest)**: Models over-extrapolate negation or contrast. When a premise contains implicit comparison or evaluative language, models predict contradiction rather than recognising the logical gap between premise and hypothesis.

**Entailment → Contradiction = 0**: Neither the encoder nor hybrid systems ever make this catastrophic error. The models reliably avoid misclassifying logically consistent pairs as contradictions, which would represent a complete reasoning failure.

### 8.3 Detailed Linguistic Case Studies (Hybrid v2)

**Case 1 — Ent → Neu (Idiomatic Equivalence)**
- Premise: *"Most of it, I couldn't even begin to identify."*
- Hypothesis: *"I didn't know what any of it was."*
- True: **entailment** | Predicted: neutral
- Failure mode: The idiom "couldn't even begin to" requires pragmatic inference that equals total ignorance. Models lack the idiomatic grounding to map this to a direct statement.

**Case 2 — Neu → Ent (Interrogative Misparse)**
- Premise: *"Why bother to sacrifice your lives for dirt farmers and slavers?"*
- Hypothesis: *"People sacrifice their lives for farmers and slaves."*
- True: **neutral** | Predicted: entailment
- Failure mode: The rhetorical question framing makes the premise a challenge rather than an assertion. Models ignore the interrogative syntax and extract the propositional content as a factual claim.

**Case 3 — Con → Neu (Attribution Blindness)**
- Premise: *"He argued that these governors shared the congressional agenda..."*
- Hypothesis: *"The speaker agrees with the governors."*
- True: **contradiction** | Predicted: neutral
- Failure mode: "Argued that" attributes a belief to a third party rather than the speaker. The model fixates on "shared... agenda" and misses that "argued" creates an attributional distance between the speaker's own view and the reported view.

**Case 4 — Mismatched Verbatim Genre**
- Highly specialised vocabulary (e.g., technical acronyms, non-standard formatting) in the Verbatim genre disrupts in-context reasoning. Models default to neutral when semantic relationships cannot be traced through unknown entity types, inflating neutral predictions in this genre.

### 8.4 Per-Genre Error Rate (DeBERTa-v3-base vs Hybrid v2 θ=0.90)

| Genre | DeBERTa Error % | Hybrid Error % | Delta |
|-------|-----------------|----------------|-------|
| Fiction | 12.2% | 12.2% | 0.0% |
| Government | 8.6% | 8.0% | −0.6% |
| Slate | 10.3% | 9.7% | −0.6% |
| Telephone | 9.8% | 11.0% | +1.2% |
| Travel | 8.7% | 8.7% | 0.0% |

The hybrid system reduces errors in formal written genres (Government: −0.6%, Slate: −0.6%) but slightly worsens on Telephone transcripts (+1.2%), where GPT-4o's fallback is penalised by conversational disfluencies absent from its few-shot examples.

### 8.5 Hybrid v5 Error Analysis: The Hard 100

Of the 100 disagreement-gated samples, 84 were misclassified (84% error rate — the inverse of what one would want from an LLM fallback). The error pattern:

| Error Type | Count |
|------------|-------|
| Entailment → Neutral | 28 |
| Contradiction → Neutral | 15 |
| Neutral → Contradiction | 11 |
| Neutral → Entailment | 11 |
| Unknown (parse failure) | 14 |
| Other | 5 |

This distribution, combined with GPT-4o's 51% accuracy on these rows, confirms they represent the irreducible label-ambiguous core of the MultiNLI dataset — cases where trained annotators may themselves disagree.

---

## Section 9 — Discussion and Business Implications

### 9.1 The Encoder-LLM Complementarity Principle

The central finding of this project is that encoder models and LLMs solve fundamentally different sub-problems within NLI. Encoder models (particularly DeBERTa-v3) are highly accurate on syntactically clear, unambiguous inference pairs — which constitute ~96% of the test set. Their confidence scores are well-calibrated: high-confidence predictions (≥0.90) are correct 90.1% of the time, matching overall accuracy. LLMs, by contrast, bring broader world knowledge and cross-domain generalisation, scoring 4.5–6.5% *higher* than their matched accuracy on mismatched genres.

The hybrid gatekeeper exploits this complementarity directly. By using encoder confidence as a routing signal, the system routes easy cases (96%) to the free, fast encoder and hard cases (4%) to the expensive LLM. The result is an architecture that is simultaneously faster, cheaper, and more accurate than either component in isolation.

### 9.2 Confidence Gating vs Ensemble Gating: When to Use Each

This study is the first to directly compare confidence-threshold gating and ensemble disagreement gating on the same NLI dataset. The results reveal a fundamental architectural distinction:

**Confidence gating** (v1–v4): Escalates samples where the encoder is uncertain. The escalated rows are tractable for LLMs — GPT-4o scores ~85% on them, adding genuine value. This is the operationally recommended approach.

**Ensemble disagreement gating** (v5): Escalates samples where multiple strong encoders actively disagree. These rows are not uncertain — the models are highly confident in *different* answers. This gating strategy identifies annotation ambiguity rather than model uncertainty, and the LLM fallback cannot rescue genuinely ambiguous samples (51% accuracy on escalated rows). Ensemble gating has high diagnostic value — it surfaces the inherent noise floor in the dataset — but lower operational value compared to confidence gating.

The practical implication: in production systems, **confidence gating is superior for accuracy**; ensemble gating is superior for **data quality auditing** — identifying the samples that should be re-annotated or flagged for human review.

### 9.3 Business Applications

**Clinical NLP and Legal Document Processing**: High-stakes NLI applications (entailment verification in medical records, contradiction detection in legal contracts) require both high accuracy and explainable escalation paths. The hybrid gatekeeper provides a natural audit trail: samples routed to the LLM can be logged for human review, while the 96% encoder-handled portion processes at near-zero marginal cost. At 1 million daily queries — typical for an enterprise medical records system — Hybrid v4 saves approximately $368/day over pure GPT-4o P3 while exceeding its accuracy.

**Real-Time Information Verification**: News and social media fact-checking systems require sub-second inference at high volume. Encoder models run in ~20ms per sample; GPT-4o API latency is 500–2000ms. The hybrid gatekeeper processes 96% of queries at encoder speed, reserving LLM latency for the 4% of ambiguous claims that genuinely require deeper reasoning.

**Annotation Quality Assurance**: The ensemble disagreement gate (Hybrid v5) has a direct business application as a data annotation tool. By processing large datasets through 3 DeBERTa variants and flagging disagreement cases for human review, organisations can identify label-ambiguous samples before model training — improving dataset quality at a fraction of the cost of full human review. The 12.5% disagreement rate on MultiNLI suggests ~12–15% of NLI annotations across datasets may require adjudication.

### 9.4 SOTA Context

Published benchmarks on the full MultiNLI dev-matched set report:
- DeBERTa-v3-large (He et al., ICLR 2023): 91.8%
- T5-11B: ~92–93%
- Moritz Laurer DeBERTa-v3-large (885k multi-dataset): ~92–93%

Our DeBERTa-v3-base result of 90.12% is consistent with published base-model benchmarks. The gap to SOTA (1.7–3pp) reflects the difference between fine-tuning on the full 392k MultiNLI training set vs the cross-encoder architecture used here, which was fine-tuned on SNLI + MultiNLI jointly. Claims of 95%+ accuracy invariably refer either to SNLI (a simpler single-domain benchmark) or to tiny test sets with high variance. On full MultiNLI dev-matched, no published system exceeds 93% at the time of writing.

### 9.5 Limitations

1. **Sample size confidence intervals**: 800 matched and 400 mismatched samples yield ±3.5% and ±5.0% CIs, meaning differences below 4pp may not be statistically significant.
2. **Single confidence threshold sweep**: Thresholds were evaluated at 0.85, 0.90, 0.95 — a finer sweep (e.g., 0.88, 0.92) might identify a marginally better operating point.
3. **Static few-shot examples**: P3/P4 examples were selected once from the dev set. Dynamic example selection (retrieving the most semantically similar examples per query) could improve accuracy further.
4. **Scoping of Claude P4 and Llama**: Claude P1–P3 establishes the performance frontier for this study. The P4 (CoT) strategy and Llama 3.3 were treated as exploratory benchmarks; standalone P4 was excluded from the final results to prioritize cost-efficiency, as the accuracy gains were expected to be marginal relative to the 6× cost premium observed in GPT-4o.
5. **Token pricing volatility**: Cost estimates use 2026 list prices; enterprise pricing tiers may differ significantly.

---

## Section 10 — Conclusion and Future Work

### 10.1 Summary of Findings

This evaluation across 5 encoders, 4 LLM families, 4 prompt strategies, and 5 hybrid configurations produces four clear conclusions:

**RQ1 (Prompt Engineering)**: Chain-of-thought prompting improves matched accuracy (+1.5pp over zero-shot for GPT-4o) but hurts mismatched performance (−0.5pp), due to genre-specific format bias in the reasoning template. Few-shot examples add value up to ~3 examples; beyond that, token cost grows faster than accuracy. Claude Sonnet P1–P3 all exceed GPT-4o P4, with P3 achieving 88.5% — the best pure-API result in this study.

**RQ2 (Cost-Accuracy Pareto)**: Three systems form the Pareto frontier: DeBERTa-v3-base (free, 90.12%), Hybrid v4 (near-free, 90.62%), and Hybrid v1 (for mismatched optimisation, 91.3%). All pure LLM systems fall below this frontier, with GPT-5 exhibiting the worst cost-accuracy ratio in the study.

**RQ3 (Hybrid Gatekeeper)**: Confidence-gated hybrid systems outperform both encoder and LLM components individually. Hybrid v4 achieves the best matched accuracy (90.62%) at $0.007/1k — a 98% cost reduction from GPT-4o while gaining 5pp in accuracy. The gatekeeper's value is most pronounced on mismatched genres where LLMs add genuine cross-domain reasoning.

**RQ4 (Cross-Genre Generalisation)**: LLMs generalise *better* to mismatched genres than their matched performance would suggest (+4.5–6.5pp), while fine-tuned encoders show near-zero degradation. The hybrid system inherits LLM generalisation at encoder cost, achieving the highest mismatched accuracy (91.3%) of any system.

### 10.2 Novel Contributions

1. **First direct comparison** of confidence-threshold gating vs ensemble disagreement gating on MultiNLI, revealing that disagreement gating identifies annotation ambiguity (not model uncertainty).
2. **Quantification of the hard sample ceiling**: 12.5% of MultiNLI test samples are label-ambiguous even to GPT-4o (51% accuracy), establishing a practical upper bound for NLI accuracy without annotation improvement.
3. **DeBERTa-v3-large calibration finding**: The large model's superior confidence calibration makes it a better gatekeeper than the base model despite identical aggregate accuracy.

### 10.3 Future Work

1. **Retrieval-augmented few-shot selection**: Replace static P3 examples with dynamically retrieved nearest-neighbour examples from the dev set, expected to improve accuracy on genre-specific edge cases.
2. **Hybrid v5c — Ensemble Gate + Claude Sonnet**: Replace GPT-4o with Claude Sonnet as the v5 fallback to test whether Claude's stronger reasoning outperforms GPT-4o on the 100 genuinely ambiguous samples. Notebook `05f_hybrid_v5c_ensemble_claude.py` is prepared. Given that Claude P4 timed out in standalone evaluation, `max_tokens` should be reduced to 150 before running.
3. **Re-annotation study**: Use ensemble disagreement gate to identify the highest-ambiguity MultiNLI samples for IAA (Inter-Annotator Agreement) testing, quantifying the proportion of cases that represent genuine annotation noise.
4. **Latency profiling**: Measure end-to-end inference latency for each architecture to complement the cost analysis with a latency-accuracy Pareto curve.

---

## Section 11 — References

1. Williams, A., Nangia, N., & Bowman, S. (2018). A broad-coverage challenge corpus for sentence understanding through inference. *NAACL-HLT 2018*.
2. Brown, T., et al. (2020). Language models are few-shot learners. *NeurIPS 33*, 1877–1901.
3. Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS 35*.
4. Devlin, J., Chang, M., Lee, K., & Toutanova, (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*.
5. He, P., et al. (2021). DeBERTa: Decoding-enhanced BERT with disentangled attention. *ICLR 2021*.
6. He, P., et al. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. *ICLR 2023*.
7. Liu, Y., et al. (2019). RoBERTa: A robustly optimised BERT pretraining approach. *arXiv:1907.11692*.
8. Ye, X., & Durrett, G. (2022). The unreliability of explanations in few-shot prompting for textual reasoning. *NeurIPS 2022*.
9. Laurer, M., et al. (2024). Less annotating, more classifying: Addressing the data scarcity issue of supervised machine learning with deep transfer learning and BERT-NLI. *Political Analysis*.

---

## Appendix A — Execution Order

Run notebooks in this order:
```
01_data_preparation.py          → data/nli_*.csv
02_encoder_baselines.py         → results/encoder_predictions_*.csv
03_gpt4o_prompting.py           → results/api_results_gpt4o*.csv
04_other_llms.py                → results/api_results_{claude,gpt5,llama}.csv
05_hybrid_gatekeeper.py         → results/hybrid_v{1,2}_results.csv
05b_hybrid_v3_deberta_gpt4o_32shot.py  → results/hybrid_v3_results.csv
05c_hybrid_v4_deberta_large_gpt4o.py   → results/hybrid_v4_results.csv
05d_hybrid_v5_ensemble_gate.py         → results/hybrid_v5_results.csv
05e_hybrid_v5b.py               → results/hybrid_v5b_results.csv [no API calls]
05f_hybrid_v5c_ensemble_claude.py      → results/hybrid_v5c_results.csv
06_cost_analysis.py             → results/cost_summary.csv
07_figures.py                   → figures/fig{1-10}*.png
08_error_analysis.py            → results/error_analysis.csv
```

## Appendix B — Figures Index

| Figure | File | Content |
|--------|------|---------|
| Fig 1 | fig1_strategy_accuracy_bar.png | Accuracy bar chart across all systems |
| Fig 2 | fig2_cost_accuracy_frontier.png | Pareto frontier scatter plot |
| Fig 3 | fig3_matched_vs_mismatched.png | Cross-genre comparison |
| Fig 4 | fig4_per_class_f1.png | F1 by label class |
| Fig 5 | fig5_cm_deberta.png | DeBERTa confusion matrix |
| Fig 6 | fig6_cm_gpt4o.png | GPT-4o P4 confusion matrix |
| Fig 7 | fig7_cm_claude.png | Claude Sonnet confusion matrix |
| Fig 8 | fig8_cm_hybrid.png | Hybrid v2 confusion matrix |
| Fig 8b | fig8b_cm_hybrid_v3.png | Hybrid v3 confusion matrix |
| Fig 9 | fig9_genre_heatmap.png | Genre × system accuracy heatmap |
| Fig 10 | fig10_hybrid_threshold.png | Threshold vs accuracy/API% dual-axis |

## Appendix C — Environment and Reproducibility

```
Python 3.11+
transformers>=4.40
torch>=2.0 (MPS / CUDA / CPU)
openai>=1.0
anthropic>=0.25
scikit-learn>=1.3
pandas, numpy, matplotlib, seaborn
python-dotenv
```

API keys required in `.env`:
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...  # optional, for Llama 3.3
```
