# Group 7 MLT1 Formative 2: SDG 3 Indicator Text Classification

This repository contains a complete multi-label text classification workflow for the Machine Learning Techniques 1 Assignment 2. The goal is to classify Devex text samples into the relevant Sustainable Development Goal 3 (SDG 3) indicators.

The task is multi-label, not multi-class. One document can belong to more than one SDG 3 indicator, so the target is represented as a multi-hot matrix with one binary column per indicator.

## Project Summary

- Dataset: `Devex_train.csv` and `Devex_test_questions.csv`
- Training rows: 2,995
- Test rows: 998
- Target labels: 27 SDG 3 indicators
- Primary metric: Hamming Loss, where lower is better
- Secondary metrics: micro F1, macro F1, precision, recall, exact match accuracy, and runtime
- Final selected model: tuned ensemble of word/character TF-IDF Logistic Regression and MiniLM sentence embeddings
- Best validation result: Hamming Loss `0.0417`, micro F1 `0.6653`

## Repository Structure

```text
group7-mlt1-formative2/
  README.md
  requirements.txt
  config.yaml
  datasets/
    Devex_train.csv
    Devex_test_questions.csv
  src/
    data.py
    preprocessing.py
    features.py
    models.py
    evaluation.py
    experiments.py
    embeddings.py
    inference.py
    plotting.py
    utils.py
  scripts/
    run_eda.py
    run_experiments.py
    generate_predictions.py
  notebooks/
    01_eda_preprocessing.ipynb
    02_experiments_results.ipynb
  reports/
    figures/
    tables/
    predictions/
  cache/
    embeddings/
    models/
```

## Dataset

The project expects the assignment files to be in `datasets/`:

```text
datasets/Devex_train.csv
datasets/Devex_test_questions.csv
```

The train file contains the input text, metadata, and label columns. The test file contains the same input structure but does not include the target labels.

Important dataset details:

- The files are read using `latin1` encoding.
- `Text` is the main input column.
- `Type` is used as optional metadata.
- `Label 1` to `Label 12` are parsed into 27 unique SDG 3 indicator labels.
- Each training row has at least one label.
- The generated dataset summary is saved to `reports/tables/dataset_summary.csv`.

## Setup

Python 3.10 or later is recommended. In Google Colab, the default Python runtime should work.

Create and activate a virtual environment locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you are using Cursor or VS Code notebooks, select the same Python environment where the requirements were installed. The notebooks also include a first setup cell that checks and installs missing packages into the active notebook kernel.

## End-to-End Workflow

Run the scripts from the repository root.

### 1. Run EDA and Preprocessing Evidence

```bash
python scripts/run_eda.py
```

This creates dataset summaries, label statistics, source type counts, text length statistics, and EDA figures.

Main outputs:

- `reports/tables/dataset_summary.csv`
- `reports/tables/label_frequency.csv`
- `reports/tables/label_cardinality.csv`
- `reports/tables/type_distribution.csv`
- `reports/tables/text_length_summary.csv`
- `reports/tables/label_cooccurrence.csv`
- `reports/figures/label_frequency.png`
- `reports/figures/label_cardinality.png`
- `reports/figures/type_distribution.png`
- `reports/figures/text_length_distribution.png`
- `reports/figures/label_cooccurrence_heatmap.png`

### 2. Run All Experiments

```bash
python scripts/run_experiments.py
```

This runs all 8 validation experiments, saves the results, and updates comparison figures.

Main outputs:

- `reports/tables/experiment_results.csv`
- `reports/tables/final_model_comparison.csv`
- `reports/tables/label_wise_metrics.csv`
- `reports/figures/experiment_hamming_loss.png`
- `reports/figures/f1_comparison.png`
- `reports/figures/label_error_heatmap.png`

### 3. Generate Final Test Predictions

```bash
python scripts/generate_predictions.py
```

This trains the selected final ensemble workflow and generates predictions for the unlabelled test set.

Main outputs:

- `reports/predictions/devex_test_predictions.csv`
- `reports/tables/final_prediction_metadata.csv`

The prediction file contains `Unique ID` plus one binary column for each SDG 3 indicator label.

## Notebook Workflow

Two notebooks are included for walkthrough and presentation:

- `notebooks/01_eda_preprocessing.ipynb`
- `notebooks/02_experiments_results.ipynb`

Run the notebooks from top to bottom. The first executable cell in each notebook:

- detects the project root,
- adds the repository to `sys.path`,
- checks missing dependencies,
- installs missing packages into the active notebook kernel,
- prints the kernel path being used.

The notebooks call the same scripts as the command-line workflow, so notebook results and script results stay consistent.

## Experiments

The project uses 8 experiments. The progression is intentional: each experiment changes one important modelling choice or adds a feature to test whether it improves validation performance.

| Experiment | Method | Main Parameters / Change | Purpose |
|---|---|---|---|
| 1 | TF-IDF + Logistic Regression | word n-grams `(1,2)`, `max_features=20000`, `C=1.0`, threshold `0.5` | Establish baseline lexical model. |
| 2 | TF-IDF + Logistic Regression | adds `Type` token, word n-grams `(1,3)`, `min_df=2`, `max_df=0.95` | Test preprocessing and metadata. |
| 3 | TF-IDF + Linear SVM | word n-grams `(1,2)`, balanced Linear SVM, threshold `0.0` | Compare margin-based sparse classifier. |
| 4 | Word + char TF-IDF + Logistic Regression | word `(1,2)`, char `(3,5)`, `class_weight="balanced"`, `C=2.0` | Improve noisy text and rare-label handling. |
| 5 | MiniLM embeddings + Logistic Regression | `all-MiniLM-L6-v2`, one-vs-rest Logistic Regression | Test semantic sentence representation. |
| 6 | MiniLM embeddings + Classifier Chain | label order by frequency, Logistic Regression base model | Test label dependency modelling. |
| 7 | Tuned word + char TF-IDF + Logistic Regression | same as Exp 4, per-label thresholds from `0.1` to `0.9` | Optimize Hamming Loss directly. |
| 8 | TF-IDF + MiniLM ensemble | TF-IDF/MiniLM weights `[0.3, 0.5, 0.7]`, tuned thresholds | Combine lexical and semantic strengths. |

## Current Validation Results

| Rank | Experiment | Hamming Loss | Micro F1 | Macro F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| 1 | `exp08_tfidf_minilm_ensemble` | 0.0417 | 0.6653 | 0.5969 | 0.8043 | 0.5673 |
| 2 | `exp07_tuned_word_char_logreg` | 0.0430 | 0.6554 | 0.5883 | 0.7907 | 0.5597 |
| 3 | `exp05_minilm_logreg` | 0.0501 | 0.5293 | 0.3643 | 0.8413 | 0.3861 |
| 4 | `exp06_minilm_classifier_chain` | 0.0503 | 0.5417 | 0.3681 | 0.8084 | 0.4073 |
| 5 | `exp03_tfidf_linear_svm` | 0.0505 | 0.6216 | 0.5379 | 0.6861 | 0.5682 |

The complete table is generated at `reports/tables/final_model_comparison.csv`.

## Final Model Interpretation

The selected model is Experiment 8: a tuned ensemble of word/character TF-IDF Logistic Regression and MiniLM embeddings.

This model performed best because the two representations help in different ways:

- TF-IDF captures exact SDG and health-sector language such as `HIV`, `malaria`, `vaccines`, `maternal mortality`, and `health workers`.
- Character n-grams help with acronyms, partial terms, and noisy spelling.
- MiniLM embeddings capture broader semantic similarity when the same topic is expressed using different wording.
- Per-label threshold tuning improves Hamming Loss because the labels are imbalanced and a fixed `0.5` threshold is too rigid for all indicators.

## Configuration

Important settings are stored in `config.yaml`:

- dataset paths,
- output paths,
- random seed,
- validation split size,
- threshold grid,
- sentence transformer model name.

The default seed is `42`, and the validation split is `20%`.

## Troubleshooting

If a notebook says `ModuleNotFoundError`, run the first setup cell again or install the requirements manually:

```bash
pip install -r requirements.txt
```

If Matplotlib has cache permission issues, set a writable cache directory:

```bash
mkdir -p cache/matplotlib
MPLBACKEND=Agg MPLCONFIGDIR=cache/matplotlib python scripts/run_eda.py
```

If Sentence-Transformers downloads the MiniLM model on the first run, the first experiment run may take longer. Later runs use the cache under `cache/embeddings/`.

## Team Roles by Branch

- `kelvin-branch`: data loading, schema validation, EDA, preprocessing, and baseline experiments.
- `samuel-branch`: TF-IDF feature engineering, Logistic Regression, Linear SVM, and word/character n-gram experiments.
- `preye-branch`: MiniLM sentence embeddings, embedding cache, and Classifier Chain experiments.
- `rene-branch`: Hamming Loss evaluation, threshold tuning, ensemble, final prediction generation, results, and README/report support.

## Limitations and Responsible AI Notes

The dataset is small and label frequencies are imbalanced, so rare SDG 3 indicators are harder to predict. Some labels have very clear lexical signals, while others are broader or have fewer examples.

The model should support human review rather than replace it. Automated SDG indicator assignment can affect reporting and interpretation of development work, so false positives and false negatives should be reviewed carefully.
