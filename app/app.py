import os
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_COLS = [
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]

DISPLAY_NAMES = [
    "WizardLM-13B",
    "Claude Instant",
    "Claude v1",
    "Claude v2",
    "GPT-3.5 Turbo",
    "GPT-4",
    "CodeLlama-34B",
    "Llama-2-70B",
    "Mistral-7B",
    "Mixtral-8x7B",
    "Yi-34B",
]

FINAL_THRESHOLD = 0.94
MAX_LENGTH = 384
GPT4_INDEX = MODEL_COLS.index("gpt-4-1106-preview")

# Training-split mean historical costs captured by the project.
MEAN_TRAIN_COST = np.array([
    7.9269856e-05,
    0.00026399308,
    0.0023590098,
    0.0027059794,
    0.00026386313,
    0.003713676,
    0.00018116446,
    0.00021006948,
    4.8790404e-05,
    0.00014154297,
    0.00019933419,
], dtype=np.float32)

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "distilbert_llm_router"
)

MODEL_PATH = Path(
    os.environ.get(
        "ROUTER_MODEL_PATH",
        str(DEFAULT_MODEL_PATH),
    )
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found at {MODEL_PATH}. "
        "Run notebook 04 or set ROUTER_MODEL_PATH."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)
model.eval()


def predict_scores(question: str) -> np.ndarray:
    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
        if key != "token_type_ids"
    }

    with torch.no_grad():
        outputs = model(**inputs)

    scores = (
        outputs.logits
        .squeeze(0)
        .float()
        .cpu()
        .numpy()
    )

    return np.clip(scores, 0.0, 1.0)


def select_model(scores: np.ndarray, threshold: float):
    cost_order = np.argsort(MEAN_TRAIN_COST)

    eligible = [
        i for i in cost_order
        if scores[i] >= threshold
    ]

    if eligible:
        selected = eligible[0]
        reason = (
            f"{len(eligible)} model(s) crossed threshold "
            f"{threshold:.2f}; the cheapest eligible model was selected."
        )
    else:
        selected = int(np.argmax(scores))
        reason = (
            f"No model crossed threshold {threshold:.2f}; "
            "the highest predicted performer was selected."
        )

    return selected, reason, len(eligible)


def live_router(question, mode, demo_threshold):
    if not str(question).strip():
        return "Please enter a question.", pd.DataFrame()

    scores = predict_scores(str(question))
    threshold = (
        FINAL_THRESHOLD
        if mode == "Evaluation Mode"
        else float(demo_threshold)
    )

    selected, reason, eligible_count = select_model(scores, threshold)

    selected_cost = float(MEAN_TRAIN_COST[selected])
    gpt4_cost = float(MEAN_TRAIN_COST[GPT4_INDEX])
    saving = (1 - selected_cost / gpt4_cost) * 100

    table = pd.DataFrame({
        "Model": DISPLAY_NAMES,
        "Predicted Performance": scores.round(4),
        "Estimated Historical Cost ($)": MEAN_TRAIN_COST.round(7),
        "Above Threshold": scores >= threshold,
    })

    table["Selected"] = False
    table.loc[selected, "Selected"] = True
    table = table.sort_values("Predicted Performance", ascending=False)

    mode_note = (
        "Evaluation Mode uses the frozen experimental threshold."
        if mode == "Evaluation Mode"
        else (
            "Exploration Mode changes the threshold for demonstration only; "
            "these choices are not part of the reported test experiment."
        )
    )

    summary = f"""
## Routing Decision

**Mode:** `{mode}`

**Selected model:** `{DISPLAY_NAMES[selected]}`

**Predicted performance:** `{scores[selected]:.3f}`

**Threshold used:** `{threshold:.2f}`

**Models above threshold:** `{eligible_count}`

**Estimated historical cost:** `${selected_cost:.6f}`

**GPT-4 historical average cost:** `${gpt4_cost:.6f}`

**Estimated saving vs GPT-4:** `{saving:.2f}%`

**Reason:** {reason}

{mode_note}

> The predicted value is a RouterBench performance-score estimate, not a calibrated probability.
"""
    return summary, table


with gr.Blocks(title="Cost-Aware LLM Router") as demo:
    gr.Markdown(
        """
# Cost-Aware LLM Router

A DistilBERT-based ML router that estimates expected RouterBench
performance for 11 candidate LLMs and chooses the cheapest model
that satisfies a quality threshold.

### Final held-out test result

- DistilBERT Router: **0.7604 mean performance**
- GPT-4 Fixed: **0.7681 mean performance**
- Historical average cost saving: **~23%**
"""
    )

    with gr.Tab("Live Router"):
        question_input = gr.Textbox(
            label="Enter a question",
            placeholder="Example: What is the derivative of x^2?",
            lines=5,
        )

        mode_input = gr.Radio(
            choices=["Evaluation Mode", "Exploration Mode"],
            value="Evaluation Mode",
            label="Routing Mode",
        )

        threshold_slider = gr.Slider(
            minimum=0.50,
            maximum=0.99,
            value=FINAL_THRESHOLD,
            step=0.01,
            label="Exploration Threshold (ignored in Evaluation Mode)",
        )

        route_button = gr.Button("Route Question", variant="primary")
        routing_summary = gr.Markdown()
        routing_table = gr.Dataframe(
            label="Predicted Model Performance",
            interactive=False,
        )

        route_button.click(
            fn=live_router,
            inputs=[question_input, mode_input, threshold_slider],
            outputs=[routing_summary, routing_table],
        )

    gr.Markdown(
        """
---

### Limitation

This application demonstrates **routing**. It does not call the
downstream LLMs to generate answers. Costs refer to historical
RouterBench estimates.
"""
    )


if __name__ == "__main__":
    demo.launch()
