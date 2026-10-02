# Notebooks

The repository follows the experimental pipeline in order:

1. `01_data_audit.ipynb` — inspect RouterBench schema, domains, targets, and environment.
2. `02_build_dataset.ipynb` — build the canonical X/Y/C dataset and the original 80/10/10 split.
3. `03_baselines.ipynb` — TF-IDF + Ridge baseline.
4. `04_train_distilbert.ipynb` — fine-tune the 11-output DistilBERT regressor.
5. `05_router_eval.ipynb` — validation-only threshold selection.
6. `06_final_test_eval.ipynb` — frozen held-out test evaluation.
7. `07_analysis_and_visuals.ipynb` — final plots, per-domain analysis, and examples.
8. `08_gradio_demo.ipynb` — original Colab demo.
9. `09_robustness_baselines_bootstrap.ipynb` — extended threshold curves, domain/argmax baselines, paired bootstrap, and replay sample export.

Additional reproducibility notebooks used after the original frozen experiment:
- `10-multiseed-distilbert.ipynb` — three-seed robustness experiment on Kaggle GPU.
- `11_publish_router_to_huggingface.ipynb` — publish the saved checkpoint to Hugging Face Hub (to be added after the Hub publish step).

The original frozen Colab result and the later robustness experiments are reported separately.
