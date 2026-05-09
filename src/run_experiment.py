"""
Confidence-Calibrated Explainable Learning for Phishing URL Detection and SOC Triage

This script reproduces the main experiments:
1. Load PhiUSIIL dataset from UCI
2. Select numeric features
3. Benchmark ML and neural models
4. Run feature correlation and ablation analysis
5. Compare calibration methods
6. Run SOC triage threshold analysis
7. Generate SHAP feature importance

Author: Abdul Aleem Syed
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from xgboost import XGBClassifier
import shap


RANDOM_STATE = 42
RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]

        if i == 0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob > lower) & (y_prob <= upper)

        if np.sum(mask) > 0:
            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_prob[mask])
            bin_weight = np.mean(mask)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def build_models():
    """Build the models used in the benchmark."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        "MLP Neural Network": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=40,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
        ),
    }


def evaluate_model(name, model, X_test, y_test):
    """Evaluate a fitted model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Brier Score": brier_score_loss(y_test, y_prob),
        "Log Loss": log_loss(y_test, y_prob),
        "ECE": expected_calibration_error(y_test, y_prob),
    }


def soc_triage_analysis(y_true, y_prob, lower_threshold, upper_threshold):
    """Map calibrated probabilities into SOC triage actions."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    auto_allow = y_prob <= lower_threshold
    manual_review = (y_prob > lower_threshold) & (y_prob < upper_threshold)
    auto_block = y_prob >= upper_threshold

    total = len(y_true)

    allow_count = np.sum(auto_allow)
    review_count = np.sum(manual_review)
    block_count = np.sum(auto_block)

    allow_accuracy = np.mean(y_true[auto_allow] == 0) if allow_count > 0 else np.nan
    block_precision = np.mean(y_true[auto_block] == 1) if block_count > 0 else np.nan
    review_phishing_rate = np.mean(y_true[manual_review] == 1) if review_count > 0 else np.nan

    return {
        "Lower Threshold": lower_threshold,
        "Upper Threshold": upper_threshold,
        "Auto-Allow Count": allow_count,
        "Auto-Allow %": allow_count / total * 100,
        "Manual Review Count": review_count,
        "Manual Review %": review_count / total * 100,
        "Auto-Block Count": block_count,
        "Auto-Block %": block_count / total * 100,
        "Auto-Allow Correct %": allow_accuracy * 100 if not np.isnan(allow_accuracy) else np.nan,
        "Auto-Block Precision %": block_precision * 100 if not np.isnan(block_precision) else np.nan,
        "Manual Review Phishing Rate %": review_phishing_rate * 100 if not np.isnan(review_phishing_rate) else np.nan,
    }


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    print("Loading PhiUSIIL dataset from UCI...")
    dataset = fetch_ucirepo(id=967)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    print(f"Original features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    y_clean = (y["label"] == 0).astype(int)
    X_numeric = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).copy()

    dataset_summary = pd.DataFrame({
        "Item": [
            "Total records",
            "Original features",
            "Numeric features used",
            "Legitimate URLs",
            "Phishing URLs",
            "Missing values",
        ],
        "Value": [
            len(X),
            X.shape[1],
            X_numeric.shape[1],
            int((y_clean == 0).sum()),
            int((y_clean == 1).sum()),
            int(X.isnull().sum().sum()),
        ],
    })
    dataset_summary.to_csv(TABLES_DIR / "dataset_summary.csv", index=False)

    print("Running all-feature benchmark...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric,
        y_clean,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_clean,
    )

    benchmark_results = []
    benchmark_models = build_models()

    for name, model in benchmark_models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        benchmark_results.append(evaluate_model(name, model, X_test, y_test))

    benchmark_df = pd.DataFrame(benchmark_results).sort_values(
        by="F1-score", ascending=False
    )
    benchmark_df.to_csv(TABLES_DIR / "all_feature_benchmark_results.csv", index=False)

    print("Running feature correlation analysis...")
    corr_df = X_numeric.copy()
    corr_df["phishing_target"] = y_clean.values

    feature_corr = (
        corr_df.corr(numeric_only=True)["phishing_target"]
        .drop("phishing_target")
        .abs()
        .sort_values(ascending=False)
    )

    top_corr_df = feature_corr.head(20).reset_index()
    top_corr_df.columns = ["Feature", "Absolute Correlation with Target"]
    top_corr_df.to_csv(TABLES_DIR / "top_20_feature_correlations.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(
        top_corr_df["Feature"].head(15),
        top_corr_df["Absolute Correlation with Target"].head(15),
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Absolute Correlation with Target")
    plt.title("Top Features Correlated with Phishing Target")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_feature_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Running ablation experiments...")
    all_ablation_results = []

    for top_k in [5, 10, 15]:
        features_to_remove = list(feature_corr.head(top_k).index)
        X_ablation = X_numeric.drop(columns=features_to_remove)

        X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
            X_ablation,
            y_clean,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_clean,
        )

        models_a = build_models()

        for name, model in models_a.items():
            print(f"Training {name} after removing top {top_k} features...")
            model.fit(X_train_a, y_train_a)
            row = evaluate_model(name, model, X_test_a, y_test_a)
            row["Ablation Setting"] = f"Removed Top {top_k}"
            row["Removed Feature Count"] = top_k
            row["Remaining Features"] = X_ablation.shape[1]
            all_ablation_results.append(row)

    ablation_df = pd.DataFrame(all_ablation_results)
    ablation_df.to_csv(TABLES_DIR / "ablation_results.csv", index=False)

    print("Training final model with top 15 correlated features removed...")
    features_to_remove_15 = list(feature_corr.head(15).index)
    X_paper = X_numeric.drop(columns=features_to_remove_15)

    X_temp, X_test_c, y_temp, y_test_c = train_test_split(
        X_paper,
        y_clean,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_clean,
    )

    X_train_c, X_calib, y_train_c, y_calib = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    xgb_base = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    xgb_base.fit(X_train_c, y_train_c)

    y_prob_uncal = xgb_base.predict_proba(X_test_c)[:, 1]
    calibration_results = [evaluate_model("Uncalibrated XGBoost", xgb_base, X_test_c, y_test_c)]

    calibrated_models = {}

    for method in ["isotonic", "sigmoid"]:
        try:
            calibrated_model = CalibratedClassifierCV(
                estimator=xgb_base,
                method=method,
                cv="prefit",
            )
        except TypeError:
            calibrated_model = CalibratedClassifierCV(
                base_estimator=xgb_base,
                method=method,
                cv="prefit",
            )

        calibrated_model.fit(X_calib, y_calib)
        calibrated_models[method] = calibrated_model

        y_prob_method = calibrated_model.predict_proba(X_test_c)[:, 1]
        y_pred_method = (y_prob_method >= 0.5).astype(int)

        calibration_results.append({
            "Model": f"{method.capitalize()}-Calibrated XGBoost",
            "Accuracy": accuracy_score(y_test_c, y_pred_method),
            "Precision": precision_score(y_test_c, y_pred_method),
            "Recall": recall_score(y_test_c, y_pred_method),
            "F1-score": f1_score(y_test_c, y_pred_method),
            "ROC-AUC": roc_auc_score(y_test_c, y_prob_method),
            "Brier Score": brier_score_loss(y_test_c, y_prob_method),
            "Log Loss": log_loss(y_test_c, y_prob_method),
            "ECE": expected_calibration_error(y_test_c, y_prob_method),
        })

    calibration_df = pd.DataFrame(calibration_results)
    calibration_df.to_csv(TABLES_DIR / "calibration_methods_comparison.csv", index=False)

    final_model = calibrated_models["sigmoid"]
    y_prob_sigmoid = final_model.predict_proba(X_test_c)[:, 1]
    y_pred_sigmoid = (y_prob_sigmoid >= 0.5).astype(int)

    final_results = pd.DataFrame([{
        "Model": "Sigmoid-Calibrated XGBoost",
        "Accuracy": accuracy_score(y_test_c, y_pred_sigmoid),
        "Precision": precision_score(y_test_c, y_pred_sigmoid),
        "Recall": recall_score(y_test_c, y_pred_sigmoid),
        "F1-score": f1_score(y_test_c, y_pred_sigmoid),
        "ROC-AUC": roc_auc_score(y_test_c, y_prob_sigmoid),
        "Brier Score": brier_score_loss(y_test_c, y_prob_sigmoid),
        "Log Loss": log_loss(y_test_c, y_prob_sigmoid),
        "ECE": expected_calibration_error(y_test_c, y_prob_sigmoid),
    }])
    final_results.to_csv(TABLES_DIR / "final_sigmoid_calibrated_xgboost_results.csv", index=False)

    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test_c, y_pred_sigmoid)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Phishing"],
    )
    disp.plot()
    plt.title("Confusion Matrix: Sigmoid-Calibrated XGBoost")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_sigmoid_xgboost.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Generating calibration curve...")
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_test_c, y_prob_uncal, n_bins=10)
    prob_true_sigmoid, prob_pred_sigmoid = calibration_curve(y_test_c, y_prob_sigmoid, n_bins=10)

    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred_uncal, prob_true_uncal, marker="o", label="Uncalibrated XGBoost")
    plt.plot(prob_pred_sigmoid, prob_true_sigmoid, marker="s", label="Sigmoid-Calibrated XGBoost")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curve_sigmoid_xgboost.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Running SOC triage threshold analysis...")
    threshold_pairs = [(0.10, 0.90), (0.05, 0.95), (0.01, 0.99)]
    triage_results = []

    for lower, upper in threshold_pairs:
        triage_results.append(soc_triage_analysis(y_test_c, y_prob_sigmoid, lower, upper))

    triage_df = pd.DataFrame(triage_results)
    triage_df.to_csv(TABLES_DIR / "soc_triage_threshold_analysis.csv", index=False)

    print("Running SHAP explainability analysis...")
    sample_X = X_test_c.sample(n=2000, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(xgb_base)
    shap_values = explainer.shap_values(sample_X)

    shap.summary_plot(shap_values, sample_X, show=False)
    plt.title("SHAP Summary Plot: XGBoost")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_xgboost.png", dpi=300, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        "Feature": sample_X.columns,
        "Mean Absolute SHAP Value": mean_abs_shap,
    }).sort_values(by="Mean Absolute SHAP Value", ascending=False)

    shap_importance_df.head(15).to_csv(TABLES_DIR / "top_15_shap_features.csv", index=False)

    print("Experiment completed successfully.")
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
