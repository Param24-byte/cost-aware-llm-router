# Cost-Aware LLM Router

A college machine-learning project that learns how to route each question to a suitable LLM while reducing inference cost.

## Simple idea

Imagine 11 teachers:
- some are expensive,
- some are cheap,
- some are better at certain questions.

Instead of always asking the most expensive teacher, a small ML model first reads the question and estimates how well each teacher is likely to perform. The router then chooses the cheapest model that satisfies a quality threshold.

That small decision model is our **DistilBERT router**.

---

## Final test result

On the untouched test set:

| System | Mean Performance | Mean Historical Cost / Query |
|---|---:|---:|
| Cheapest Fixed | 0.322704 | 0.000049 |
| TF-IDF Router (extended grid, θ=0.97) | 0.732804 | 0.002197 |
| **DistilBERT Router** | **0.760448** | **0.002877** |
| GPT-4 Fixed | 0.768067 | 0.003738 |
| Oracle | 0.888659 | 0.000263 |

The DistilBERT router stayed about **0.76 percentage points below GPT-4** while reducing historical average cost by about **23%**.

The TF-IDF row above uses the **validation-selected extended-grid point (θ=0.97)**. No TF-IDF threshold met the validation target, so 0.97 was chosen as the highest-validation-performance fallback; its test value is reported descriptively. See [validation threshold curves](assets/validation_threshold_curves.png) and [test threshold curves](assets/test_threshold_curves.png).

### Key findings

A simple **domain rule essentially matches GPT-4's test performance** (0.768394 vs 0.768067) at about **21.97% lower historical cost**. The frozen DistilBERT router saves about **23.05%** at an observed **0.76 percentage-point performance loss**. Relative to the domain rule, prompt-level DistilBERT routing adds a cost-efficiency trade-off in **MMLU** and **ARC** (about 17.9% and 31.9% lower cost respectively, with performance losses of about 0.93 pp and 1.36 pp), but it is **dominated on GSM8K** (about 0.44 pp lower performance and 6.8% higher cost). Finally, the bootstrap interval extends beyond the project's 1-point margin, so the point estimate met the target but strict statistical non-inferiority was not established.

These are benchmark-specific findings, not claims about arbitrary production prompts.

Frozen routing threshold: **0.94**

### Robustness and additional baselines

The original frozen Colab run remains the primary reported experiment. Additional checks were added afterward without retuning that frozen result.

| System | Validation Performance | Validation Cost | Test Performance | Test Cost |
|---|---:|---:|---:|---:|
| GPT-4 Fixed | 0.775044 | 0.003638 | 0.768067 | 0.003738 |
| Domain Rule | 0.776786 | 0.002860 | 0.768394 | 0.002917 |
| DistilBERT Argmax Only | 0.775479 | 0.002939 | 0.761754 | 0.002985 |
| Domain Classifier Router | 0.776786 | 0.002860 | 0.768394 | 0.002917 |
| DistilBERT Cost-Aware | 0.767639 | 0.002836 | 0.760448 | 0.002877 |

The benchmark-domain classifier reached 100% accuracy on this restricted three-domain split, so the domain-rule and domain-classifier routers produce the same routing result. This should not be interpreted as 100% domain accuracy on arbitrary real-world prompts.

Across three additional training seeds (13, 42, 77), the DistilBERT cost-aware router achieved:

- test performance: **0.761428 ± 0.001359**
- historical test cost: **0.002966 ± 0.000093**
- gap vs GPT-4: **0.664 ± 0.136 percentage points**
- cost saving vs GPT-4: **20.65% ± 2.49%**

These three runs were performed on **Kaggle with Python 3.12.13 and PyTorch 2.10.0+cu128**, whereas the original frozen Colab experiment used a different PyTorch/CUDA environment. The Kaggle seed-42 result was **0.762952** versus **0.760448** in the frozen Colab run, a difference of about **0.25 percentage points**. The 1,725-step training trace plus PyTorch gather warnings are consistent with multi-GPU data-parallel execution (effective global batch approximately 32 rather than 16), although the notebook only printed the first Tesla T4 device. Therefore, the three-seed standard deviation should be treated as a **descriptive robustness check, not a precise variance estimate**; five or more seeds would be preferable for a stronger estimate.

A paired bootstrap on the original frozen test choices estimated a router-minus-GPT-4 performance difference of **-0.007619**, with a 95% percentile interval of approximately **[-0.013714, -0.001524]**. The observed point estimate met the project's 1-percentage-point criterion, but strict statistical non-inferiority at that margin was not established.

---

## Project pipeline

```text
RouterBench
    ↓
Data audit
    ↓
MMLU + ARC + GSM8K subset
    ↓
X = question text
Y = 11 model performance scores
C = 11 model costs
    ↓
80 / 10 / 10 split
    ↓
TF-IDF + Ridge baseline
    ↓
DistilBERT multi-output regression
    ↓
Cost-aware routing
    ↓
Validation threshold selection
    ↓
One-time test evaluation
    ↓
Gradio demo
```

---

## Dataset

The project uses the 0-shot release of **RouterBench** and focuses on:
- MMLU
- ARC-Challenge
- GSM8K / grade-school-math

RouterBench contains historical outputs, performance scores, and estimated costs for multiple LLMs.

In the selected data, including GSM8K, the model-performance fields use five discrete score levels: **0, 0.25, 0.50, 0.75, and 1.00**. We preserve these as ordered fractional performance targets: 0 is the lowest score, 1 is the highest, and the intermediate values are fractional score levels supplied by the benchmark. They are **not calibrated probabilities**. Because these targets are not purely binary, the project uses bounded multi-output regression rather than forcing them into 0/1 labels.

The released performance fields are therefore kept as values in **[0, 1]**, rather than being forcibly converted to binary labels.

---

## Models in the routing pool

The project uses 11 historical RouterBench models, including:
- GPT-4
- GPT-3.5 Turbo
- Claude v1 / v2 / Instant
- Mixtral
- Mistral
- Yi-34B
- WizardLM
- Llama-2-70B
- CodeLlama-34B

---

## ML methods

### Baseline
TF-IDF + Ridge Regression

### Main model
DistilBERT with 11 regression outputs.

### Related work and routing-policy attribution

The general cost/quality routing idea is **adapted from prior multi-LLM routing work**, not claimed as a new routing principle. Relevant references include **FrugalGPT**, **Hybrid LLM**, **RouteLLM**, and **Switchcraft**. Switchcraft is especially close at the policy level: it uses a lightweight DistilBERT-based router and selects a lower-cost model subject to a correctness requirement. Our project-specific adaptation is to predict **11 RouterBench performance scores** and apply a validation-selected threshold followed by a cheapest-eligible rule. Full citations and links are in [`references/SOURCES.md`](references/SOURCES.md).

For one input question:

```text
question
   ↓
DistilBERT
   ↓
11 predicted performance scores
   ↓
cost-aware policy
   ↓
selected LLM
```

---

## Routing rule

For a threshold `θ`:

1. predict one score for every candidate LLM;
2. keep models with predicted score >= `θ`;
3. among them, choose the cheapest model;
4. if none crosses the threshold, choose the model with the highest predicted score.

The final threshold `0.94` was selected using validation data only.

---

## Test-set routing distribution

The DistilBERT router selected approximately:

| Model | Selection Share |
|---|---:|
| GPT-4 | 58.34% |
| Claude-v2 | 19.76% |
| Mixtral-8x7B | 9.23% |
| Claude-v1 | 6.70% |
| Yi-34B | 4.31% |
| GPT-3.5 Turbo | 0.96% |
| Claude Instant | 0.70% |

The other four models were not selected on the frozen test set.

---

## Per-domain test performance

| Domain | Router | GPT-4 |
|---|---:|---:|
| MMLU | 0.795018 | 0.804270 |
| ARC | 0.952381 | 0.965986 |
| GSM8K | 0.657383 | 0.660738 |

The overall project target is satisfied, although the same one-percentage-point gap is not guaranteed independently for every domain.

---

## Gradio demo

The demo contains two modes.

### Live Router

> **⚠️ Out-of-distribution warning:** the training prompts carried benchmark-specific templates and formatting. Arbitrary free-text questions may therefore behave differently from the held-out RouterBench evaluation. Use **Benchmark Replay** when demonstrating the evaluated experiment.

Enter an arbitrary question and inspect:
- predicted score for each LLM,
- selected model,
- threshold,
- estimated historical cost,
- estimated saving vs GPT-4.

Because arbitrary questions may differ from RouterBench's benchmark prompt format, Live Router includes:
- **Evaluation Mode** — fixed threshold 0.94
- **Exploration Mode** — adjustable threshold for demonstrating the quality/cost trade-off

Exploration Mode is not part of the reported test results.

### Benchmark Replay
Select a held-out RouterBench test question and inspect:
- frozen router decision,
- predicted scores,
- actual historical RouterBench performance,
- actual historical cost,
- GPT-4 comparison.

---

## Repository structure

```text
cost-aware-llm-router/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── app/
│   └── app.py
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_build_dataset.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_train_distilbert.ipynb
│   ├── 05_router_eval.ipynb
│   ├── 06_final_test_eval.ipynb
│   ├── 07_analysis_and_visuals.ipynb
│   ├── 08_gradio_demo.ipynb
│   ├── 09_robustness_baselines_bootstrap.ipynb
│   ├── 10-multiseed-distilbert.ipynb
│   └── 11_publish_to_huggingface.ipynb
├── data/
│   └── benchmark_replay_sample.csv
├── results/
│   ├── final_test_metrics.csv
│   ├── extended_baseline_comparison.csv
│   ├── paired_bootstrap_summary.csv
│   ├── distilbert_multiseed_runs.csv
│   └── distilbert_multiseed_summary.csv
├── assets/
├── references/
│   └── SOURCES.md
└── models/
    └── README.md
```

---

## How to reproduce

### Required inputs

For the full pipeline, start from the public RouterBench 0-shot data. The project then creates:

- `router_project_canonical.pkl` — canonical prompt/performance/cost table;
- `router_split_indices.npz` — original fixed 80/10/10 split;
- the saved DistilBERT checkpoint for later evaluation/demo steps.

For the Kaggle multi-seed notebook, only `router_project_canonical.pkl` is required because notebook 10 reconstructs the original split deterministically with scikit-learn 1.6.1.

### Notebook order

1. `01_data_audit.ipynb` — inspect RouterBench, domains, target values, and environment.
2. `02_build_dataset.ipynb` — build the canonical dataset and fixed train/validation/test split.
3. `03_baselines.ipynb` — train TF-IDF + Ridge baseline.
4. `04_train_distilbert.ipynb` — train the 11-output DistilBERT regressor.
5. `05_router_eval.ipynb` — choose the DistilBERT routing threshold using validation only.
6. `06_final_test_eval.ipynb` — run the frozen one-time test evaluation.
7. `07_analysis_and_visuals.ipynb` — per-domain analysis, model-selection plots, and examples.
8. `08_gradio_demo.ipynb` — original interactive demo.
9. `09_robustness_baselines_bootstrap.ipynb` — extended threshold curves, domain/argmax baselines, paired bootstrap, and replay sample.
10. `10-multiseed-distilbert.ipynb` — three-seed Kaggle robustness run on the same split.
11. `11_publish_to_huggingface.ipynb` — publish the trained checkpoint to Hugging Face.

The original frozen Colab result should remain separate from later robustness runs; do not replace it with the best seed.

## Trained checkpoint

The trained DistilBERT checkpoint is intentionally **not committed to normal Git history** because it is large.

Published model:

`Paam24/cost-aware-llm-router-distilbert`

Hugging Face: https://huggingface.co/Paam24/cost-aware-llm-router-distilbert

The standalone app loads this public Hugging Face checkpoint by default, so no retraining is required before running the demo.

Optional overrides:

1. set `ROUTER_MODEL_ID=<another-hugging-face-model>`, or
2. set `ROUTER_MODEL_PATH=/path/to/local/checkpoint`.

The standalone app includes both **Live Router** and **Benchmark Replay**. The replay tab uses the small frozen sample stored in `data/benchmark_replay_sample.csv`.

---

## Installation

```bash
git clone https://github.com/Param24-byte/cost-aware-llm-router.git
cd cost-aware-llm-router

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python app/app.py
```

---

## Limitations

- historical model pool,
- historical RouterBench costs,
- benchmark-oriented inputs,
- live arbitrary questions may be out of distribution,
- no actual downstream LLM APIs are called by the demo,
- oracle is theoretical and unavailable at real inference time,
- one global threshold may not be optimal for every domain.

---

## Future work

- current LLM APIs and live pricing,
- domain-specific thresholds,
- probability/score calibration,
- dynamic model pool,
- stronger router backbones,
- out-of-domain evaluation,
- open-ended response quality evaluation,
- reinforcement-learning or bandit routing.

---

## Attribution

This project integrates public research datasets and open-source libraries. See [`references/SOURCES.md`](references/SOURCES.md).

The project contribution is the end-to-end integration, preprocessing, target formulation, baseline comparison, DistilBERT fine-tuning, cost-aware routing policy, evaluation pipeline, analysis, and demo.
