# Sources and Attribution

## RouterBench
Dataset / benchmark used for historical model performance and cost information.

- Hugging Face dataset: `withmartian/routerbench`
- RouterBench repository/paper should be cited in the final report using its official citation details.

## DistilBERT
Pretrained text encoder used as the routing model backbone.

- Model: `distilbert-base-uncased`
- Library: Hugging Face Transformers

## scikit-learn
Used for:
- TF-IDF
- Ridge Regression
- regression metrics

## Gradio
Used to build the interactive demonstration interface.

## NumPy / Pandas / Matplotlib
Used for numerical processing, tabular data handling, and visualizations.

---

## Reuse policy

This project does not claim ownership of:
- RouterBench,
- DistilBERT,
- Hugging Face Transformers,
- scikit-learn algorithms,
- Gradio.

The project-specific work consists of:
- data inspection and validation,
- three-domain subset construction,
- canonical X/Y/C dataset,
- preserving RouterBench [0,1] performance targets,
- classical baseline pipeline,
- DistilBERT score-prediction fine-tuning,
- validation-only threshold selection,
- cost-aware routing policy,
- final held-out evaluation,
- error analysis,
- interactive demonstration,
- report and presentation integration.
