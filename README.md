# NLI Classification — LLM-Based NLP
**University of Edinburgh | MSc Business Analytics**

---

## Project Overview

Comprehensive evaluation of Natural Language Inference (NLI) systems using MultiNLI, comparing encoder baselines, LLM prompting strategies, and hybrid gatekeeper architectures.

**Best system**: Hybrid v4 (DeBERTa-v3-large + GPT-4o) — **90.62% matched, $0.007/1k queries**

**Key finding**: Ensemble disagreement gating reveals that 87.5% of MultiNLI samples are unanimously solvable at 95.0% accuracy, while the remaining 12.5% represent genuinely label-ambiguous cases that even GPT-4o resolves only at 51% accuracy.

---

## Directory Structure

```
LLM Final/
├── data/
│   ├── nli_dev_200.csv          # Prompt tuning set (200 samples)
│   ├── nli_test_800.csv         # Primary test set (800 matched)
│   └── nli_test_mm_400.csv      # Generalisation test set (400 mismatched)
│
├── src/
│   ├── 01_data_preparation.py   # Stratified sampling from MultiNLI JSONL
│   ├── 02_encoder_baselines.py  # 5 encoder models (BERT, RoBERTa, DeBERTa ×3)
│   ├── 03_gpt4o_prompting.py    # GPT-4o P1–P4 + P5 (32-shot), matched+mismatched
│   ├── 04_other_llms.py         # Claude Sonnet 4.5, GPT-5 (o3-mini), Llama 3.3
│   ├── 05a_hybrid_v1_v2_gatekeeper.py  # Hybrid v1 (GPT-4o) + v2 (Claude), θ ∈ {0.85,0.90,0.95}
│   ├── 05b_hybrid_v3_deberta_gpt4o_32shot.py      # Hybrid v3: 32-shot fallback
│   ├── 05c_hybrid_v4_deberta_large_gpt4o.py      # Hybrid v4: DeBERTa-large gate ⭐ BEST MATCHED
│   ├── 05d_hybrid_v5_ensemble_gate.py      # Hybrid v5: 3-DeBERTa ensemble gate ⭐ KEY INSIGHT
│   ├── 05e_hybrid_v5b_tiered.py        # Hybrid v5b: tiered fallback (no new API calls)
│   ├── 05f_hybrid_v5c_ensemble_claude.py     # Hybrid v5c: ensemble gate + Claude Sonnet
│   ├── 06_cost_analysis.py      # Token usage and cost aggregation
│   ├── 07a_figures_main.py            # 12 publication-quality figures
│   ├── 07b_figure2_pareto.py          # Pareto frontier figure script
│   ├── 08_error_analysis.py     # Error type distribution + linguistic cases
│   └── 09_genre_label_analysis.py      # Genre/class P/R/F1 breakdowns
│
├── utils/
│   ├── evaluate.py              # Common evaluation functions
│   └── generate_tables.py       # Table generation utilities
│
├── results/                     # All CSV outputs (auto-generated)
├── figures/                     # All figure PNGs (auto-generated)
├── NLI_Comprehensive_Results.md # Main report document
├── requirements.txt
├── .env                         # API keys (not committed)
└── .env.example
```

---

## Execution Order

```bash
# 1. Data (one-time, requires MultiNLI JSONL files)
python src/01_data_preparation.py

# 2. Encoders (~30 min on MPS/GPU)
python src/02_encoder_baselines.py

# 3. GPT-4o prompting (~$3 total)
python src/03_gpt4o_prompting.py

# 4. Other LLMs (Claude running, Llama rate-limited)
python src/04_other_llms.py

# 5. Hybrid systems (run in order)
python src/05a_hybrid_v1_v2_gatekeeper.py       # v1 + v2
python src/05b_hybrid_v3_deberta_gpt4o_32shot.py  # v3
python src/05c_hybrid_v4_deberta_large_gpt4o.py   # v4 ⭐
python src/05d_hybrid_v5_ensemble_gate.py          # v5 ⭐

# 5b. Post-processing (no API calls required)
python src/05e_hybrid_v5b_tiered.py             # v5b: tiered ensemble

# 5c. Claude ensemble (when ready)
python src/05f_hybrid_v5c_ensemble_claude.py  # v5c: Claude fallback

# 6. Analysis
python src/06_cost_analysis.py
python src/07a_figures_main.py
python src/07b_figure2_pareto.py
python src/08_error_analysis.py
python src/09_genre_label_analysis.py
```

---

## Key Results Summary

| System | Matched Acc | Mismatched Acc | API % | Cost/1k |
|--------|-------------|----------------|-------|---------|
| DeBERTa-v3-base | 90.12% | 90.8% | 0% | $0.000 |
| DeBERTa-v3-large | 90.12% | 89.5% | 0% | $0.000 |
| GPT-4o P4 (pure) | 85.50% | 90.0% | 100% | $0.410 |
| Claude Sonnet P3 (pure) | 88.50% | — | 100% | $2.235 |
| Hybrid v1 θ=0.90 | 90.12% | **91.3%** | 3.8% | $0.011 |
| **Hybrid v4 θ=0.90** ⭐ | **90.62%** | 90.5% | 2.0% | **$0.007** |
| Hybrid v5 (Ensemble) | 89.50% | 90.25% | 12.5% | $0.288 |

---

## Environment Setup

```bash
pip install -r requirements.txt

# Create .env from example
cp .env.example .env
# Add your keys to .env:
#   OPENAI_API_KEY=...
#   ANTHROPIC_API_KEY=...
#   GROQ_API_KEY=...
```

---

## Report

Full methodology, results, and analysis: `NLI_Comprehensive_Results.md`

---

*Seed: 42 | All experiments reproducible with provided notebooks*
