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
| TF-IDF Router | 0.733239 | 0.002181 |
| **DistilBERT Router** | **0.760448** | **0.002877** |
| GPT-4 Fixed | 0.768067 | 0.003738 |
| Oracle | 0.888659 | 0.000263 |

The DistilBERT router stayed about **0.76 percentage points below GPT-4** while reducing historical average cost by about **23%**.

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

The released performance fields are kept as values in **[0, 1]**, rather than being forcibly converted to binary labels.

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
│   └── 08_gradio_demo.ipynb
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

## Important: trained checkpoint

The trained DistilBERT checkpoint is intentionally **not committed to normal Git history** because it is large.

To run the demo:
1. set `ROUTER_MODEL_ID=<username>/<model-repo>` after publishing the checkpoint to Hugging Face Hub, or
2. place a compatible trained checkpoint under `models/distilbert_llm_router/`, or
3. set `ROUTER_MODEL_PATH` to another local checkpoint path.

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
