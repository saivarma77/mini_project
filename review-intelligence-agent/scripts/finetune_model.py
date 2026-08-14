"""
finetune_model.py
------------------
Optional script to fine-tune a transformer model (DistilBERT / RoBERTa / DeBERTa)
on the Amazon Reviews Polarity dataset from Kaggle:
https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

This is NOT required to run the Streamlit app (which uses pretrained
sentiment checkpoints out of the box), but is provided so you can train
your own model on the actual project dataset for a stronger report /
better accuracy comparison.

Usage:
    1. Download & unzip the Kaggle dataset so you have:
       train.ft.txt.bz2 and test.ft.txt.bz2
    2. Run:
       python finetune_model.py --model distilbert-base-uncased \
           --train_file train.ft.txt.bz2 --sample_size 20000 --epochs 2
"""

import argparse
import bz2
import random

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

MODEL_CHECKPOINTS = {
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
    "deberta": "microsoft/deberta-v3-base",
}


def load_fasttext_format(path, sample_size=20000, seed=42):
    """Parses the fastText-style '__label__1 <text>' format used by this dataset."""
    rows = []
    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label_str, text = line.split(" ", 1)
            label = 0 if label_str == "__label__1" else 1  # 0=negative, 1=positive
            rows.append({"text": text, "label": label})

    random.seed(seed)
    random.shuffle(rows)
    rows = rows[:sample_size]
    return pd.DataFrame(rows)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CHECKPOINTS.keys()), required=True)
    parser.add_argument("--train_file", required=True, help="Path to train.ft.txt or train.ft.txt.bz2")
    parser.add_argument("--test_file", default=None, help="Optional path to test.ft.txt(.bz2)")
    parser.add_argument("--sample_size", type=int, default=20000)
    parser.add_argument("--test_sample_size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", default="./trained_model")
    args = parser.parse_args()

    checkpoint = MODEL_CHECKPOINTS[args.model]
    print(f"Loading tokenizer & model: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    print("Loading & sampling training data...")
    train_df = load_fasttext_format(args.train_file, sample_size=args.sample_size)

    if args.test_file:
        test_df = load_fasttext_format(args.test_file, sample_size=args.test_sample_size)
    else:
        # carve out a validation split from train if no test file given
        test_df = train_df.sample(frac=0.1, random_state=42)
        train_df = train_df.drop(test_df.index)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Final evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    print("To use this fine-tuned model in the Streamlit app, update MODEL_OPTIONS "
          "in app/utils/model_loader.py to point at this output_dir.")


if __name__ == "__main__":
    main()
