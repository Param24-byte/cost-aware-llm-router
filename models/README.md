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
    └── training_args.bin
```

The standalone app supports either:

```bash
ROUTER_MODEL_ID=<hugging-face-user>/<model-repo> python app/app.py
```

or a local checkpoint:

```bash
ROUTER_MODEL_PATH=/path/to/checkpoint python app/app.py
```

The Hugging Face repository ID will be documented here after the checkpoint is published.
