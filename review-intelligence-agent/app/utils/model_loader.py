"""
model_loader.py
----------------
Loads and caches Hugging Face sentiment-analysis pipelines for
RoBERTa, DeBERTa, and DistilBERT so the Streamlit app can switch
between them without reloading on every rerun.
"""

import streamlit as st
from transformers import pipeline

# Pretrained, sentiment-finetuned checkpoints for each architecture.
# These are public Hugging Face models — first use will download weights.
MODEL_OPTIONS = {
    "DistilBERT": {
        "checkpoint": "distilbert-base-uncased-finetuned-sst-2-english",
        "description": "Lightweight & fast (66M params). Good baseline for quick iteration.",
    },
    "RoBERTa": {
        "checkpoint": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "description": "Robust general-purpose sentiment model (125M params), 3-class (pos/neu/neg).",
    },
    "DeBERTa": {
        "checkpoint": "MoritzLaurer/deberta-v3-base-zeroshot-v1",
        "description": "Disentangled attention architecture (184M params). Uses a DeBERTa "
                        "NLI checkpoint in zero-shot mode for pos/neg/neutral classification.",
    },
}


@st.cache_resource(show_spinner=False)
def _load_pipeline(checkpoint: str, task: str = "sentiment-analysis"):
    return pipeline(task, model=checkpoint, tokenizer=checkpoint)


@st.cache_resource(show_spinner=False)
def _load_zero_shot(checkpoint: str):
    return pipeline("zero-shot-classification", model=checkpoint)


class ZeroShotSentimentWrapper:
    """
    Wraps a zero-shot classification pipeline so it exposes the same
    call signature as a standard sentiment-analysis pipeline:
    pipe(list_of_texts) -> [{"label": ..., "score": ...}, ...]
    Used as a fallback for base DeBERTa (not sentiment fine-tuned).
    """

    def __init__(self, zs_pipe):
        self.zs_pipe = zs_pipe
        self.candidate_labels = ["positive", "negative", "neutral"]

    def __call__(self, texts, truncation=True):
        if isinstance(texts, str):
            texts = [texts]
        results = []
        for t in texts:
            out = self.zs_pipe(t, candidate_labels=self.candidate_labels, truncation=True)
            top_label = out["labels"][0]
            top_score = out["scores"][0]
            results.append({"label": top_label.upper(), "score": float(top_score)})
        return results


def get_sentiment_pipeline(model_key: str):
    """
    Returns a callable pipeline for the given model key that accepts
    a list of strings and returns [{"label":..., "score":...}, ...]
    """
    config = MODEL_OPTIONS[model_key]
    checkpoint = config["checkpoint"]

    if model_key == "DeBERTa":
        # Uses a DeBERTa NLI checkpoint in zero-shot mode, since public
        # DeBERTa checkpoints are not directly sentiment fine-tuned.
        # For higher accuracy, fine-tune DeBERTa yourself on the Amazon
        # Reviews dataset and point 'checkpoint' at your saved model dir.
        zs = _load_zero_shot(checkpoint)
        return ZeroShotSentimentWrapper(zs)

    return _load_pipeline(checkpoint)
