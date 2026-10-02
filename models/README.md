# Model checkpoint

The trained DistilBERT router is intentionally excluded from normal Git history because of its size.

Expected local structure:

```text
models/
└── distilbert_llm_router/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── vocab.txt
    └── ...
```

You can create this checkpoint by running:

`notebooks/04_train_distilbert.ipynb`

The Gradio app also supports:

```bash
ROUTER_MODEL_PATH=/path/to/checkpoint python app/app.py
```
