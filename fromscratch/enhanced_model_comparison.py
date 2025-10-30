"""
Enhanced Model Comparison for Biomass Prediction from NDVI

This script loads the NDVI/biomass dataset, trains multiple regression models,
evaluates them on training and test sets, compares performance, explores a
polynomial feature extension, and generates clear visualizations.

Models included:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor (optional; skipped if xgboost not installed)
- Support Vector Regressor (SVR)

Outputs:
- model_comparison_detailed.csv: metrics for each model
- metrics_bars.png: bar charts for R², RMSE, MAE across models
- residuals_grid.png: residual scatter plots for each model
- feature_importance_random_forest.png and feature_importance_xgboost.png (if available)
- Printed summary indicating best model and brief rationale

Run:
    python enhanced_model_comparison.py
"""

import os
import sys
import time
import warnings
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Detect XGBoost availability with clear diagnostics
HAS_XGB = False
XGB_VERSION = None
try:
    import importlib.util as _ilu
    _spec = _ilu.find_spec("xgboost")
    if _spec is not None:
        from xgboost import XGBRegressor, __version__ as XGB_VERSION
        HAS_XGB = True
    else:
        HAS_XGB = False
except Exception as _xgberr:
    HAS_XGB = False
    XGB_VERSION = None


def load_dataset(csv_path: str = "sample_ndvi_biomass_data.csv") -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load the dataset and return X, y, and the full DataFrame.

    Expects columns: 'ndvi' and 'biomass'
    """
    df = pd.read_csv(csv_path)
    if "ndvi" not in df.columns or "biomass" not in df.columns:
        raise ValueError("CSV must contain 'ndvi' and 'biomass' columns.")

    X = df["ndvi"].values.reshape(-1, 1)
    y = df["biomass"].values
    return X, y, df


def get_models(random_state: int = 42) -> Dict[str, object]:
    """
    Construct baseline models. SVR and Linear use pipelines with scaling where appropriate.
    """
    models: Dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        ),
        "SVR": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svr", SVR(C=10.0, epsilon=0.1, kernel="rbf")),
            ]
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state,
        )
    return models


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R², RMSE, MAE, and MSE."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"r2": float(r2), "rmse": rmse, "mae": float(mae), "mse": float(mse)}


def evaluate_model(name: str, model: object, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Fit model, compute train/test metrics, and measure timings.
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    t1 = time.time()
    y_pred_test = model.predict(X_test)
    predict_time = time.time() - t1

    y_pred_train = model.predict(X_train)

    test_metrics = compute_metrics(y_test, y_pred_test)
    train_metrics = compute_metrics(y_train, y_pred_train)

    return {
        "model": name,
        "r2_test": test_metrics["r2"],
        "rmse_test": test_metrics["rmse"],
        "mae_test": test_metrics["mae"],
        "mse_test": test_metrics["mse"],
        "r2_train": train_metrics["r2"],
        "rmse_train": train_metrics["rmse"],
        "mae_train": train_metrics["mae"],
        "mse_train": train_metrics["mse"],
        "fit_time_s": round(fit_time, 4),
        "predict_time_s": round(predict_time, 4),
    }


def add_polynomial_features(X: np.ndarray, degree: int = 2) -> Tuple[np.ndarray, List[str]]:
    """
    Create polynomial features from NDVI, e.g., [1, NDVI, NDVI^2].
    Returns transformed X and feature names.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    # For degree=2 with one feature, names are ['ndvi', 'ndvi^2']
    feature_names = ["ndvi", "ndvi^2"] if degree == 2 else [f"ndvi^p{i}" for i in range(1, X_poly.shape[1] + 1)]
    return X_poly, feature_names


def plot_metric_bars(df_results: pd.DataFrame, out_path: str = "metrics_bars.png") -> None:
    """Create side-by-side bar charts comparing R², RMSE, MAE across models."""
    metrics = ["r2_test", "rmse_test", "mae_test"]
    titles = {"r2_test": "R² (Test)", "rmse_test": "RMSE (Test)", "mae_test": "MAE (Test)"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, m in enumerate(metrics):
        ax = axes[i]
        ordered = df_results.sort_values(by=m, ascending=(m != "r2_test"))
        sns.barplot(data=ordered, x="model", y=m, ax=ax, palette="viridis")
        ax.set_title(titles[m])
        ax.set_xlabel("")
        ax.set_ylabel(titles[m])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_grid(models_fitted: Dict[str, object], X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, out_path: str = "residuals_grid.png") -> None:
    """Generate residual scatter plots (predicted vs residuals) for each model."""
    names = list(models_fitted.keys())
    n = len(names)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()
    for idx, name in enumerate(names):
        model = models_fitted[name]
        y_pred_test = model.predict(X_test)
        residuals = y_test - y_pred_test
        ax = axes[idx]
        ax.scatter(y_pred_test, residuals, alpha=0.6)
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"Residuals: {name}")
        ax.set_xlabel("Predicted Biomass (test)")
        ax.set_ylabel("Residuals (Actual - Predicted)")
        ax.grid(True, alpha=0.3)
    # Hide unused subplots if any
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_rf(model: RandomForestRegressor, feature_names: List[str], out_path: str = "feature_importance_random_forest.png") -> None:
    """Plot feature importances for Random Forest."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances[order], y=np.array(feature_names)[order], palette="magma")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance_xgb(model, feature_names: List[str], out_path: str = "feature_importance_xgboost.png") -> None:
    """Plot feature importances for XGBoost (gain-based)."""
    try:
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        plt.figure(figsize=(6, 4))
        sns.barplot(x=importances[order], y=np.array(feature_names)[order], palette="rocket")
        plt.title("XGBoost Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def select_best_model(df_results: pd.DataFrame) -> str:
    """
    Select the best model primarily by highest R² on test, breaking ties by lowest RMSE.
    """
    df_sorted = df_results.sort_values(by=["r2_test", "rmse_test"], ascending=[False, True])
    return df_sorted.iloc[0]["model"]


def main():
    print("🚀 Enhanced Model Comparison: Starting...")
    # Print interpreter and xgboost detection info to help debug env issues
    print(f"Python interpreter: {sys.executable}")
    try:
        import site
        print(f"site.getsitepackages(): {getattr(site, 'getsitepackages', lambda: [])()}")
    except Exception:
        pass
    if HAS_XGB:
        print(f"XGBoost detected: version {XGB_VERSION}")
    else:
        print("XGBoost not detected in this interpreter. To install:")
        print(f"  {sys.executable} -m pip install xgboost")

    # 1) Load data and split
    X, y, df = load_dataset("sample_ndvi_biomass_data.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2) Train baseline models
    models = get_models()
    if not HAS_XGB:
        print("Note: XGBoost unavailable in this environment. Skipping XGBoost.")

    results: List[Dict[str, float]] = []
    fitted_models: Dict[str, object] = {}
    for name, model in models.items():
        print(f"Training {name} ...")
        res = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results.append(res)
        fitted_models[name] = model

    # 3) Compare and save
    df_results = pd.DataFrame(results)
    df_results_sorted = df_results.sort_values(by=["r2_test", "rmse_test"], ascending=[False, True])
    df_results_sorted.to_csv("model_comparison_detailed.csv", index=False)
    print("\nModel Performance (sorted by R² test, then RMSE):")
    print(df_results_sorted.to_string(index=False))

    # 4) Polynomial features: retrain best model on [ndvi, ndvi^2]
    best_model_name = select_best_model(df_results)
    print(f"\nBest baseline model: {best_model_name}")
    X_poly, feature_names = add_polynomial_features(X, degree=2)
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Recreate the best model (fresh instance) but adapt pipelines if scaling is needed
    base_models = get_models()
    best_model = base_models[best_model_name]

    print(f"Retraining {best_model_name} with polynomial features {feature_names} ...")
    t0 = time.time()
    best_model.fit(Xp_train, yp_train)
    fit_time = time.time() - t0
    y_pred_test_poly = best_model.predict(Xp_test)
    y_pred_train_poly = best_model.predict(Xp_train)

    poly_test_metrics = compute_metrics(yp_test, y_pred_test_poly)
    poly_train_metrics = compute_metrics(yp_train, y_pred_train_poly)

    poly_row = {
        "model": f"{best_model_name}+Poly2",
        "r2_test": poly_test_metrics["r2"],
        "rmse_test": poly_test_metrics["rmse"],
        "mae_test": poly_test_metrics["mae"],
        "mse_test": poly_test_metrics["mse"],
        "r2_train": poly_train_metrics["r2"],
        "rmse_train": poly_train_metrics["rmse"],
        "mae_train": poly_train_metrics["mae"],
        "mse_train": poly_train_metrics["mse"],
        "fit_time_s": round(fit_time, 4),
        "predict_time_s": np.nan,
    }
    df_all = pd.concat([df_results, pd.DataFrame([poly_row])], ignore_index=True)
    df_all_sorted = df_all.sort_values(by=["r2_test", "rmse_test"], ascending=[False, True])
    df_all_sorted.to_csv("model_comparison_with_poly.csv", index=False)

    # 5) Visualizations
    plot_metric_bars(df_all_sorted, out_path="metrics_bars.png")
    # Build a fitted models dict that also includes the polynomial best
    fitted_models_poly = dict(fitted_models)
    fitted_models_poly[f"{best_model_name}+Poly2"] = best_model
    # For residuals, use the appropriate X for each model
    # We'll plot residuals using test sets: baseline models use (X_test, y_test); poly uses (Xp_test, yp_test)
    # To keep one grid function, we will plot baselines first, and separately append poly residuals to grid.
    # Implement simple approach: create two grids and combine to one image by saving sequentially.
    plot_residuals_grid(fitted_models, X_train, y_train, X_test, y_test, out_path="residuals_grid.png")

    # Additional residuals for the polynomial model
    try:
        y_pred_poly_test = best_model.predict(Xp_test)
        residuals_poly = yp_test - y_pred_poly_test
        plt.figure(figsize=(6, 4))
        plt.scatter(y_pred_poly_test, residuals_poly, alpha=0.6)
        plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
        plt.title(f"Residuals: {best_model_name}+Poly2")
        plt.xlabel("Predicted Biomass (test)")
        plt.ylabel("Residuals (Actual - Predicted)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("residuals_poly.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    # Feature importance plots for tree-based models using the polynomial features
    # Random Forest
    if "RandomForest" in fitted_models:
        # Refit RF on polynomial features to get importance over [ndvi, ndvi^2]
        rf_model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
        rf_model.fit(Xp_train, yp_train)
        plot_feature_importance_rf(rf_model, feature_names, out_path="feature_importance_random_forest.png")

    # XGBoost
    if HAS_XGB and "XGBoost" in fitted_models:
        xgb_model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )
        xgb_model.fit(Xp_train, yp_train)
        plot_feature_importance_xgb(xgb_model, feature_names, out_path="feature_importance_xgboost.png")

    # 6) Printed detailed metrics for each model (train and test)
    print("\nDetailed Metrics (Train/Test):")
    printable = df_all_sorted[[
        "model", "r2_train", "rmse_train", "mae_train", "r2_test", "rmse_test", "mae_test"
    ]]
    print(printable.to_string(index=False))

    # 7) Summary: best model and brief rationale
    winner = select_best_model(df_all_sorted)
    winner_row = df_all_sorted[df_all_sorted["model"] == winner].iloc[0]
    print("\n Summary:")
    print(f"Best performing model: {winner}")
    print(
        "It achieved the highest R² on the test set (indicating strongest explanatory power) "
        "and competitive/lowest RMSE and MAE (indicating accurate predictions)."
    )

    print("\n Outputs saved:")
    print(" - model_comparison_detailed.csv")
    print(" - model_comparison_with_poly.csv")
    print(" - metrics_bars.png")
    print(" - residuals_grid.png and residuals_poly.png")
    if os.path.exists("feature_importance_random_forest.png"):
        print(" - feature_importance_random_forest.png")
    if os.path.exists("feature_importance_xgboost.png"):
        print(" - feature_importance_xgboost.png")


if __name__ == "__main__":
    main()


