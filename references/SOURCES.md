# Sources and Attribution

This project combines public benchmark data, pretrained models, and open-source software. The items below are external dependencies or research sources and are **not claimed as original work**.

## 1. RouterBench — benchmark, data, and routing framework

**Paper**

Qitian Jason Hu, Jacob Bieker, Xiuyu Li, Nan Jiang, Benjamin Keigwin, Gaurav Ranganath, Kurt Keutzer, and Shriyash Kaustubh Upadhyay.  
**"ROUTERBENCH: A Benchmark for Multi-LLM Routing System."**  
arXiv preprint arXiv:2403.12031, 2024.

- Paper: https://arxiv.org/abs/2403.12031
- Official code: https://github.com/withmartian/routerbench
- Hugging Face dataset: https://huggingface.co/datasets/withmartian/routerbench
- Dataset DOI shown by Hugging Face: https://doi.org/10.57967/hf/1996

The official RouterBench repository is released under the MIT license. The Hugging Face dataset card does not currently state a separate dataset license, so this project does not infer one.

This project uses the public **0-shot** RouterBench data and selects the MMLU, ARC-Challenge, and GSM8K / grade-school-math portions.

### RouterBench BibTeX

```bibtex
@article{hu2024routerbench,
  title   = {ROUTERBENCH: A Benchmark for Multi-LLM Routing System},
  author  = {Qitian Jason Hu and Jacob Bieker and Xiuyu Li and Nan Jiang and Benjamin Keigwin and Gaurav Ranganath and Kurt Keutzer and Shriyash Kaustubh Upadhyay},
  year    = {2024},
  journal = {arXiv preprint arXiv:2403.12031}
}
```

## 2. DistilBERT — pretrained routing backbone

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf.  
**"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter."**  
arXiv preprint arXiv:1910.01108, 2019.

- Paper: https://arxiv.org/abs/1910.01108
- Base model: https://huggingface.co/distilbert/distilbert-base-uncased

The Hugging Face model card lists the base DistilBERT model under the Apache-2.0 license.

### DistilBERT BibTeX

```bibtex
@article{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2019}
}
```

## 3. Hugging Face Transformers

Used to load, fine-tune, save, and run DistilBERT.

- Documentation: https://huggingface.co/docs/transformers/
- Project: https://github.com/huggingface/transformers

Project environment used Transformers **5.17.0**.

## 4. Hugging Face Datasets

Used to construct tokenized training, validation, and test datasets for the Trainer pipeline.

- Documentation: https://huggingface.co/docs/datasets/
- Project: https://github.com/huggingface/datasets

## 5. PyTorch

Used as the deep-learning framework for DistilBERT fine-tuning and inference.

- Documentation: https://pytorch.org/docs/
- Project: https://github.com/pytorch/pytorch

The main Colab experiment reported PyTorch **2.11.0+cu130** on an NVIDIA Tesla T4.

## 6. scikit-learn

Used for:
- TF-IDF feature extraction,
- Ridge multi-output regression,
- Logistic Regression in the domain-classifier baseline,
- MAE / MSE evaluation.

- Documentation: https://scikit-learn.org/
- TF-IDF: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Project environment used scikit-learn **1.6.1**.

## 7. Gradio

Used for the interactive routing demonstration.

- Documentation: https://www.gradio.app/docs
- Quickstart: https://www.gradio.app/guides/quickstart
- Project: https://github.com/gradio-app/gradio

The final Colab demo reported Gradio **6.27.0**.

## 8. NumPy, Pandas, Matplotlib, and Joblib

Used for numerical operations, tabular processing, plots, and serialization.

- NumPy: https://numpy.org/
- Pandas: https://pandas.pydata.org/
- Matplotlib: https://matplotlib.org/
- Joblib: https://joblib.readthedocs.io/

Recorded core environment:
- NumPy 2.1.3
- Pandas 2.2.3

## 9. Project-specific contribution

The following work belongs to this project rather than the external sources listed above:

- auditing the RouterBench 0-shot schema;
- selecting and standardizing MMLU, ARC-Challenge, and GSM8K;
- constructing aligned prompt / performance / cost matrices;
- recognizing and preserving fractional RouterBench performance scores as bounded regression targets;
- creating a TF-IDF + Ridge multi-output baseline;
- fine-tuning DistilBERT for 11-output performance-score prediction;
- designing the cost-aware threshold routing rule;
- selecting thresholds on validation data;
- evaluating the frozen router on held-out test data;
- adding domain-rule, argmax-only, and domain-classifier baselines;
- paired-bootstrap uncertainty analysis;
- multi-seed robustness analysis;
- cost/performance visualizations and error analysis;
- the Gradio Live Router and Benchmark Replay interface;
- repository, report, and presentation integration.

## 10. Reproducibility note

Exact environment versions used for the final experiments should be preserved in the repository requirements files. CUDA build suffixes such as `+cu130` describe the Colab GPU build and may require the appropriate PyTorch package index on a local machine.
