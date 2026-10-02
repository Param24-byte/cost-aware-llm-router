# Model checkpoint

The trained DistilBERT router is intentionally excluded from normal Git history because the checkpoint is large (the saved `model.safetensors` is about 268 MB).

The original checkpoint is produced by:

`notebooks/04_train_distilbert.ipynb`

Local structure:

```text
models/
└── distilbert_llm_router/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── training_args.bin   # local training metadata; not needed for inference
```

Published Hugging Face model:

`Paam24/cost-aware-llm-router-distilbert`

https://huggingface.co/Paam24/cost-aware-llm-router-distilbert

The standalone app loads this public checkpoint by default.

To override it with another Hugging Face model:

```bash
ROUTER_MODEL_ID=<hugging-face-user>/<model-repo> python app/app.py
```

To use a local checkpoint instead:

```bash
ROUTER_MODEL_PATH=/path/to/checkpoint python app/app.py
```

## Hugging Face artifact policy

`training_args.bin` is not required to load the router for inference and is a PyTorch-serialized training metadata artifact. The published model repository should contain the model/tokenizer/config files but omit this file. Notebook 11 includes a cleanup cell for deleting it from the Hub after publishing.
