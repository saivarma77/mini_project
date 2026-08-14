# AI Agent for Customer Review Sentiment Analysis & Business Intelligence Generation

A mini project that classifies customer review sentiment using transformer models
(**RoBERTa**, **DeBERTa**, **DistilBERT**) and automatically generates a business
intelligence report — sentiment breakdown, category insights, aspect-level analysis
(what customers praise / complain about), and actionable recommendations.

Dataset: [Amazon Reviews (Kaggle, bittlingmayer/amazonreviews)](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)

---

## 📁 Project Structure

```
review-intelligence-agent/
├── app/
│   ├── app.py                     # Main Streamlit application
│   ├── requirements.txt
│   ├── utils/
│   │   ├── model_loader.py        # Loads & caches HF sentiment pipelines
│   │   ├── text_utils.py          # Cleaning + aspect extraction
│   │   └── report_generator.py    # Auto BI report generator
│   └── data/
│       └── sample_reviews.csv     # Demo dataset (works out of the box)
├── scripts/
│   └── finetune_model.py          # Optional: fine-tune on the real Kaggle dataset
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd review-intelligence-agent/app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> First run will download the pretrained model weights from Hugging Face
> (a few hundred MB per model) — make sure you have an internet connection.

### 2. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### 3. Try it out

- Use the **sample dataset** (loaded automatically) or **upload your own CSV**
  with a `review` text column (an Amazon Reviews CSV export works great).
- Pick a model in the sidebar: **DistilBERT**, **RoBERTa**, or **DeBERTa**.
- Click **Run Sentiment Analysis**.
- Explore the **Dashboard**, **Aspect Analysis**, and **BI Report** tabs.
- Click **Benchmark All 3 Models** to compare speed & confidence side by side.

---

## 🧠 Models Used

| Model | Checkpoint used | Notes |
|---|---|---|
| DistilBERT | `distilbert-base-uncased-finetuned-sst-2-english` | Fast, lightweight, binary sentiment |
| RoBERTa | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 3-class (positive/neutral/negative) |
| DeBERTa | `MoritzLaurer/deberta-v3-base-zeroshot-v1` | Zero-shot NLI-based classification |

These are all public, pretrained checkpoints so the app works immediately without
any training. For a stronger academic result (and to properly use the
**Amazon Reviews Polarity dataset** specified in your problem statement), fine-tune
each model yourself using `scripts/finetune_model.py` — see below.

---

## 🏋️ Fine-tuning on the real Amazon Reviews dataset (optional, recommended for your report)

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
   (you'll get `train.ft.txt.bz2` and `test.ft.txt.bz2`)

2. Install extra dependencies:
   ```bash
   pip install datasets accelerate
   ```

3. Run fine-tuning for each model, e.g.:
   ```bash
   python scripts/finetune_model.py \
       --model distilbert \
       --train_file train.ft.txt.bz2 \
       --test_file test.ft.txt.bz2 \
       --sample_size 20000 \
       --test_sample_size 2000 \
       --epochs 2 \
       --output_dir ./trained_models/distilbert-amazon

   python scripts/finetune_model.py --model roberta  --train_file train.ft.txt.bz2 --test_file test.ft.txt.bz2 --output_dir ./trained_models/roberta-amazon
   python scripts/finetune_model.py --model deberta  --train_file train.ft.txt.bz2 --test_file test.ft.txt.bz2 --output_dir ./trained_models/deberta-amazon
   ```

4. Record the accuracy / F1 / precision / recall printed at the end of each run —
   this gives you a genuine model comparison table for your project report.

5. (Optional) Point the Streamlit app at your fine-tuned models by editing
   `app/utils/model_loader.py` and changing the `checkpoint` value to your
   local `output_dir` path.

---

## 📊 Features

- **Multi-model sentiment classification** — switch between 3 transformer architectures
- **Interactive dashboard** — sentiment distribution, category breakdown, trend over time, confidence histogram
- **Aspect-level analysis** — automatically surfaces what's praised vs. complained about
- **Auto-generated BI report** — Markdown report with executive summary and recommendations, downloadable
- **Model benchmarking** — compare speed, confidence, and predictions across models
- **CSV upload** — bring your own review dataset
- **Downloadable results** — export scored data as CSV, report as Markdown

---

## 🎓 Suggested Report Additions

- Include the fine-tuning metrics table (accuracy/F1/precision/recall per model)
- Add a confusion matrix screenshot per model
- Discuss trade-offs: DistilBERT (speed) vs RoBERTa (balance) vs DeBERTa (accuracy)
- Screenshot the dashboard and BI report tabs for your submission document

---

## ⚠️ Notes

- The bundled `sample_reviews.csv` is a small synthetic demo set so the app works
  immediately without downloading the full multi-GB Kaggle dataset.
- `date` and `category` columns are auto-generated if not present in an uploaded
  CSV, purely to power the demo dashboard's trend/category charts.
- DeBERTa runs in zero-shot mode by default (no public sentiment-finetuned DeBERTa
  checkpoint exists) — fine-tune it yourself via `scripts/finetune_model.py` for
  a true apples-to-apples comparison.
