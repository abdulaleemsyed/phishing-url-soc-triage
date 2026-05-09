# Confidence-Calibrated Explainable Learning for Phishing URL Detection and SOC Triage

This repository contains the experimental code and supporting material for the paper:

**Confidence-Calibrated Explainable Learning for Phishing URL Detection and SOC Triage**

The project evaluates phishing URL detection models using the PhiUSIIL Phishing URL Dataset. The study compares classical machine learning, ensemble learning, gradient boosting, and neural models, then adds feature ablation, confidence calibration, SOC triage thresholds, and SHAP explainability.

## Research Goal

The goal of this work is not only to classify URLs as phishing or legitimate, but also to support SOC analysts with:

- calibrated confidence scores
- auto-allow, manual-review, and auto-block triage thresholds
- explainable model outputs using SHAP
- robustness checks through feature ablation

## Dataset

This project uses the **PhiUSIIL Phishing URL Dataset** from the UCI Machine Learning Repository.

The dataset is loaded directly using the `ucimlrepo` Python package. The raw dataset is not included in this repository to keep the repository lightweight and to respect dataset distribution practices.

Dataset information:

- Records: 235,795
- Original features: 54
- Numeric features used: 50
- Legitimate URLs: 134,850
- Phishing URLs: 100,945

Original label encoding:

- `1` = legitimate
- `0` = phishing

This project converts the target so that:

- `1` = phishing
- `0` = legitimate

## Methods

The experiment includes:

1. Dataset loading and preprocessing
2. Numeric feature selection
3. Model benchmarking
4. Feature-target correlation analysis
5. Feature ablation using top 5, 10, and 15 correlated features
6. Confidence calibration using isotonic and sigmoid calibration
7. SOC triage threshold analysis
8. SHAP-based explainability

Models evaluated:

- Logistic Regression
- Random Forest
- Extra Trees
- XGBoost
- Multi-Layer Perceptron Neural Network

## Final Model

The final model used in the paper is:

**Sigmoid-Calibrated XGBoost after removing the top 15 target-correlated features**

Final results:

- Accuracy: 99.9279%
- Precision: 99.9554%
- Recall: 99.8762%
- F1-score: 99.9158%
- ROC-AUC: 0.999995
- Expected Calibration Error: 0.000233

## SOC Triage Result

Using the conservative threshold pair:

- Auto-allow threshold: 0.01
- Auto-block threshold: 0.99

The model achieved:

- Auto-allow count: 26,956
- Manual review count: 49
- Auto-block count: 20,154
- Manual review rate: 0.1039%
- Auto-allow correctness: 99.9555%
- Auto-block precision: 99.9802%

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── run_experiment.py
├── data/
│   └── README.md
├── results/
│   ├── tables/
│   └── figures/
├── notebooks/
│   └── README.md
└── paper/
    └── README.md
```

## How to Run

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the experiment:

```bash
python src/run_experiment.py
```

Outputs will be saved under:

```text
results/tables/
results/figures/
```

## Important Notes

The results may vary slightly depending on Python package versions, random seeds, and execution environment. The original paper experiments were run in Google Colab.

The raw dataset is not committed to this repository. The code fetches the dataset directly from UCI using `ucimlrepo`.

## Author

Abdul Aleem Syed  
AI Security Researcher
