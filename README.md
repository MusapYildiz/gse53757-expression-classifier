# GSE53757 Expression Classification Pipeline
---

## 📖 Project Overview

This repository implements a reproducible analysis pipeline for classifying clear cell renal cell carcinoma (ccRCC) vs. normal kidney tissue using the Gene Expression Omnibus (GEO) dataset **GSE53757**. It covers:

1. **Data Loading & Label Extraction**
2. **Preprocessing & Feature Selection**
3. **Model Training with Hyperparameter Tuning**
4. **Stratified Cross-Validation & Evaluation**
5. **Result Aggregation & Visualization**

All steps are automated in a single script (`src/main.py`), enabling easy reproduction and extension.

---

## 🗂️ Repository Structure

```bash
gse53757-expression-classifier/
├── data/
│   └── GSE53757_series_matrix.txt      # Raw GEO series matrix (71 MB)
├── src/
│   └── main.py          # Main analysis script
├── outputs/
│   ├── confusion_matrix_N5_LR.png      # Average confusion matrices
│   ├── confusion_matrix_N5_RF.png
│   ├── confusion_matrix_N5_SVM.png
│   └── summary_results.csv             # Tabular summary of metrics
├── requirements.txt                    # Exact Python dependencies
└── README.md                           # This documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/MusapYildiz/gse53757-expression-classifier.git
cd gse53757-expression-classifier
```

### 2. Data Preparation

Download `GSE53757_series_matrix.txt` from [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53757) and place it in the `data/` directory.

### 3. Create Virtual Environment

```bash
python3 -m venv venv
activate venv         # macOS/Linux
# On Windows: venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> **requirements.txt** (example):
>
> ```text
> pandas==1.5.3
> numpy==1.24.2
> scikit-learn==1.2.1
> matplotlib==3.6.2
> ```

---

## 🚀 Usage

Run the full classification pipeline with a single command:

```bash
python src/main.py
```

---

## 📝 Script Breakdown (`main.py`)

### 1. Data Loading Functions

```python
def load_series_matrix(path):
    """
    Reads the GEO series matrix, skipping lines starting with '!'.
    Index is set to the probe IDs (first column).
    """
    return pd.read_csv(path, sep="\t", comment="!", index_col=0)

def extract_labels_by_field(path, field_name):
    """
    Parses '!Sample_characteristics_ch1' entries to extract values for the specified field.
    Returns a pandas Series of labels.
    """
    # Implementation details...
```

### 2. Label Binarization

* Tumor samples: those whose `tissue:` field contains "carcinoma"
* Normal samples: all others

```python
y_binary = y_tissue.apply(lambda x: 1 if 'carcinoma' in x.lower() else 0)
```

### 3. Cross-Validation & Feature Selection

* **StratifiedKFold** preserves class distribution.
* In each fold, compute variances on training set and select **top N** genes.

### 4. Model Definitions & Hyperparameter Grids

| Model | Constructor              | Parameters                                       |
| ----- | ------------------------ | ------------------------------------------------ |
| LR    | `LogisticRegression`     | C: \[0.01, 1, 100]                               |
| RF    | `RandomForestClassifier` | n\_estimators: \[50,100], max\_depth: \[None,10] |
| SVM   | `SVC(kernel='rbf')`      | C: \[0.1,1], gamma: \['scale','auto']            |

### 5. Evaluation Metrics

For each fold and (N, Model) combination, compute:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Confusion Matrix** (TN, FP, FN, TP)

Metrics are aggregated across folds into mean ± standard deviation.

### 6. Result Visualization

* Write `outputs/summary_results.csv` containing all aggregated metrics.
* Save confusion matrices as heatmap PNGs: `confusion_matrix_N{N}_{Model}.png`.

Example summary row:

| N  | Model | Acc Mean | Acc Std | Prec Mean | Recall Mean | F1 Mean |
| -- | ----- | -------- | ------- | --------- | ----------- | ------- |
| 20 | RF    | 0.88     | 0.05    | 0.90      | 0.87        | 0.88    |

---

## 📈 Outputs

All generated artifacts are saved under `outputs/`:

* **`summary_results.csv`**: Aggregated metrics table.
* **`confusion_matrix_N*_*.png`**: Heatmaps for each N and model.

Example directory:

```bash
outputs/
├── summary_results.csv
├── confusion_matrix_N5_LR.png
├── confusion_matrix_N5_RF.png
└── confusion_matrix_N5_SVM.png
```

---

## 🧪 Reproducing & Extending

* **Adjust `N-list`:** Try different numbers of genes via `--N-list`.
* **Explore models:** Add more classifiers or adjust hyperparameter grids.
* **Plot ROC/PR curves:** Integrate ROC/PR plot generation after CV.
* **Parallel execution:** Use Dask or joblib for faster feature selection.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a feature branch
3. Commit your changes with clear messages
4. Open a Pull Request

Please follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

*Data Source: [GEO GSE53757 Series Matrix File](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53757) — downloaded the **Series Matrix File** (`GSE53757_series_matrix.txt`), which contains processed expression values and sample metadata.*
