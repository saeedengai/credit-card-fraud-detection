# Credit Card Fraud Detection

End-to-end workflow to detect fraudulent transactions in an anonymized credit card dataset (PCA features `V1–V28`, plus `Amount`, `Time`; target `Class`).
This repo includes a clean project structure, your notebook(s), lightweight CLI scripts for reproducibility, and figures (PR/ROC + EDA).

---

## 🔎 Key Results

* **Best overall (PR-AUC): `RandomForest = 0.7844`**
  Also delivers **Precision = 0.93** and **Recall = 0.73** at the default **0.5** threshold → practical for reducing false alarms while catching most frauds.

* **LightGBM:** **ROC-AUC = 0.9698**; trails RF on PR-AUC; trades some precision (**≈0.72**) for slightly higher recall (**≈0.76**).

* **Logistic Regression (balanced):** very high **Recall ≈ 0.90**, with lower precision (fill exact value if logged).

> You can tune the classification threshold based on the cost of false positives vs. false negatives and available review capacity.

---

## 📁 Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .gitattributes           # Git LFS patterns for CSVs/models/figures
├── .env.example             # template for DB connection; don't commit .env
├── data/
│   ├── raw/                 # original CSV (e.g., creditcard.csv)
│   └── processed/           # cleaned CSV (e.g., creditcard_clean.csv)
├── notebooks/
│   └── YourNotebook.ipynb   # primary analysis/EDA
├── src/
│   ├── preprocess.py        # raw → clean (CLI)
│   ├── train.py             # train RF/LogReg baselines (CLI)
│   └── evaluate.py          # evaluate a saved model (CLI)
├── models/                  # saved models (.pkl/.joblib)
└── reports/
    └── figures/             # PR/ROC curves, EDA plots
```

---

## ⚙️ Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
git lfs install    # recommended if you track CSVs/models/figures
```

---

## 📦 Data

* Put the **raw CSV** in `data/raw/` (e.g., `data/raw/creditcard.csv`).
* Put the **cleaned CSV** in `data/processed/` (e.g., `data/processed/creditcard_clean.csv`).
* If the dataset has licensing/terms, prefer **linking to the source** rather than committing a large copy.

---

## 🧪 Reproduce via CLI (Option A)

These scripts are **lightweight helpers**; the notebook remains the main place for EDA/plots.

```bash
# 1) Preprocess (raw → clean)
python src/preprocess.py --input data/raw/creditcard.csv --output data/processed/creditcard_clean.csv

# 2) Train a baseline model (RandomForest or LogisticRegression)
python src/train.py --input data/processed/creditcard_clean.csv --target Class --model rf --out models/baseline_rf.pkl
# or
python src/train.py --input data/processed/creditcard_clean.csv --target Class --model logreg --out models/baseline_logreg.pkl

# 3) Evaluate a saved model
python src/evaluate.py --model models/baseline_rf.pkl --data data/processed/creditcard_clean.csv --target Class
```

---

## 📓 Notebook

* Primary analysis/EDA lives in: `notebooks/YourNotebook.ipynb`.
* Keep diffs small by clearing outputs before commit:

  ```bash
  jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/YourNotebook.ipynb
  ```

---

## 📊 Figures

Add figures to `reports/figures/` and reference them here:

* **Precision–Recall curves**
  `reports/figures/pr_curves.png`
  ![PR curves](reports/figures/pr_curves.png)

* **ROC curves**
  `reports/figures/roc_curves.png`
  ![ROC curves](reports/figures/roc_curves.png)

* **Fraud class distribution**
  `reports/figures/class_distribution.png`
  ![Class distribution](reports/figures/class_distribution.png)

* **Transaction amount distribution (log scale)**
  `reports/figures/amount_distribution_log.png`
  ![Amount distribution (log)](reports/figures/amount_distribution_log.png)

* **Time vs. Transaction Amount**
  `reports/figures/time_vs_amount.png`
  ![Time vs Amount](reports/figures/time_vs_amount.png)

* **Transaction time density by class**
  `reports/figures/time_density_by_class.png`
  ![Time density by class](reports/figures/time_density_by_class.png)

* **Correlation heatmap (V1–V28)**
  `reports/figures/corr_heatmap_v1_v28.png`
  ![Correlation heatmap](reports/figures/corr_heatmap_v1_v28.png)

> If your file names differ, adjust the paths above.

---

## 🧠 Notes & Interpretation

* **PR-AUC vs ROC-AUC:** With heavy class imbalance, PR-AUC usually reflects the precision/recall trade-off more faithfully than ROC-AUC, so a model with slightly lower ROC-AUC can still be superior operationally if it has higher PR-AUC.
* **Thresholding:** Numbers above use **0.5** by default; tune the threshold to align with review bandwidth and business costs.
* **Next steps:** cost-sensitive thresholds, LightGBM tuning, calibration, error analysis on top false positives/negatives.

---

## 🔐 Credentials

If you use a database connection, copy `.env.example` → `.env`, fill values, and load via `python-dotenv`.
**Never commit `.env`.**

---

## 📝 License

MIT (see `LICENSE`).

---

## 👤 Author

**Saeed Hosseinzadeh** — [@saeedengai](https://github.com/saeedengai)
