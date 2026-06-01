# Group 7 MLT1 Formative 2: SDG 3 Indicator Text Classification

This project builds a multi-label text classification system for predicting relevant Sustainable Development Goal 3 indicators from Devex text samples. The task is multi-label because one document can match more than one SDG 3 indicator.

## Project Summary

- Dataset: `Devex_train.csv` and `Devex_test_questions.csv`
- Train rows: 2,995
- Test rows: 998
- Target labels: 27 SDG 3 indicators
- Primary metric: Hamming Loss, where lower is better
- Final model: tuned ensemble of word/character TF-IDF Logistic Regression and MiniLM sentence embeddings
- Best validation result: Hamming Loss around `0.0417`

## Repository Structure

```text
datasets/                  Raw assignment CSV files
src/                       Reusable project code
scripts/                   Command-line workflow scripts
reports/figures/           Generated EDA and result figures
reports/tables/            Generated EDA and experiment tables
reports/predictions/       Final test predictions
notebooks/                 Notebook walkthroughs
config.yaml                Paths, seed, split, and experiment settings
requirements.txt           Python dependencies
```

## Team Roles by Branch

- `kelvin-branch`: data loading, EDA, preprocessing, and baseline experiments.
- `samuel-branch`: TF-IDF feature engineering, Logistic Regression, Linear SVM, and word/character n-gram experiments.
- `preye-branch`: MiniLM sentence embeddings and Classifier Chain experiments.
- `rene-branch`: Hamming Loss evaluation, threshold tuning, ensemble, results, predictions, and README.

## Setup

In Google Colab or locally:

```bash
pip install -r requirements.txt
```

The CSV files are already stored in `datasets/`. They are read with `latin1` encoding because the source files are not clean UTF-8.

## Reproducible Commands

Run EDA and preprocessing evidence:

```bash
python scripts/run_eda.py
```

Run all 8 experiments:

```bash
python scripts/run_experiments.py
```

Generate final test predictions:

```bash
python scripts/generate_predictions.py
```

## Experiments

| Experiment | Method | Main Change |
|---|---|---|
| 1 | TF-IDF + Logistic Regression | Baseline word n-grams |
| 2 | TF-IDF + Logistic Regression | Preprocessing plus `Type` metadata token |
| 3 | TF-IDF + Linear SVM | Margin-based sparse classifier |
| 4 | Word + character TF-IDF + Logistic Regression | Character n-grams and class balancing |
| 5 | MiniLM embeddings + Logistic Regression | Semantic sentence representation |
| 6 | MiniLM embeddings + Classifier Chain | Label dependency modeling |
| 7 | Best TF-IDF model + tuned thresholds | Direct Hamming Loss optimization |
| 8 | TF-IDF + MiniLM ensemble | Combines lexical and semantic signals |

The full table is generated at `reports/tables/experiment_results.csv`. The sorted model comparison table is generated at `reports/tables/final_model_comparison.csv`.

## Outputs

Important generated files:

- `reports/tables/dataset_summary.csv`
- `reports/tables/label_frequency.csv`
- `reports/tables/experiment_results.csv`
- `reports/tables/final_model_comparison.csv`
- `reports/tables/label_wise_metrics.csv`
- `reports/figures/label_frequency.png`
- `reports/figures/label_cardinality.png`
- `reports/figures/type_distribution.png`
- `reports/figures/label_cooccurrence_heatmap.png`
- `reports/figures/experiment_hamming_loss.png`
- `reports/figures/f1_comparison.png`
- `reports/figures/label_error_heatmap.png`
- `reports/predictions/devex_test_predictions.csv`

## Final Model Interpretation

The best validation result came from the ensemble in Experiment 8. It combines sparse TF-IDF features, which capture exact health and development terminology, with MiniLM embeddings, which capture broader semantic similarity. Per-label threshold tuning improved Hamming Loss because the labels are imbalanced and a single `0.5` threshold is too rigid for all SDG indicators.

## Demo Guide

- Kelvin: explain dataset shape, labels, EDA, and preprocessing.
- Samuel: explain TF-IDF, Logistic Regression, SVM, and word/character n-grams.
- Preye: explain MiniLM embeddings and Classifier Chain.
- Rene: explain Hamming Loss, threshold tuning, ensemble, final results, limitations, and responsible AI.

## Limitations and Responsible AI Notes

The dataset is small and label frequencies are imbalanced, so rare indicators are harder to predict. The model may reflect biases in development-sector writing, funding language, and source types. Predictions should support human review rather than replace expert judgment, especially where SDG indicator assignment affects reporting or decision-making.
