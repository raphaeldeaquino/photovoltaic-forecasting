"""
Random Forest model utilities for PV forecasting.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


PLANTS = ["Usina_1", "Usina_2", "Usina_3", "Usina_4", "Usina_5"]
DISTANCES = [151.9, 108.5, 62.7, 30.1, 4.0]
LOOKBACK_WINDOWS = [1, 6, 12, 24, 48, 72]
FORECAST_HORIZON = 24


def prepare_rf_data(
    df: pd.DataFrame,
    lookback: int,
    forecast_horizon: int,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build supervised samples for Random Forest.

    Returns:
        X: 2D array of shape (n_samples, lookback * n_features).
        y: 2D array of shape (n_samples, forecast_horizon).
    """
    X, y = [], []

    for i in range(len(df) - lookback - forecast_horizon):
        X.append(df.iloc[i : i + lookback][feature_cols].values.flatten())
        y.append(
            df.iloc[
                i + lookback : i + lookback + forecast_horizon
            ][target_col].values
        )

    return np.array(X), np.array(y)


def train_evaluate_rf(
    df: pd.DataFrame,
    lookback: int,
    forecast_horizon: int,
    input_columns: List[str],
    target_col: str,
    scaler_X: MinMaxScaler,
    scaler_y: MinMaxScaler,
    n_splits: int = 5,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Train and evaluate a Random Forest regressor with K-fold cross-validation.
    """
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    X_train_norm = scaler_X.fit_transform(df_train[input_columns])
    y_train_norm = scaler_y.fit_transform(df_train[[target_col]])

    X_test_norm = scaler_X.transform(df_test[input_columns])
    y_test_norm = scaler_y.transform(df_test[[target_col]])

    df_train_norm = pd.DataFrame(X_train_norm, columns=input_columns)
    df_y_train_norm = pd.DataFrame(y_train_norm, columns=[target_col])

    df_test_norm = pd.DataFrame(X_test_norm, columns=input_columns)
    df_y_test_norm = pd.DataFrame(y_test_norm, columns=[target_col])

    X_all, y_all = prepare_rf_data(
        df_train_norm.join(df_y_train_norm),
        lookback,
        forecast_horizon,
        input_columns,
        target_col,
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_list, rmse_list, mae_list = [], [], []

    for train_idx, val_idx in kf.split(X_all):
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_val)

        y_pred_inv = scaler_y.inverse_transform(y_pred)
        y_val_inv = scaler_y.inverse_transform(y_val)

        r2_list.append(r2_score(y_val_inv.flatten(), y_pred_inv.flatten()))
        rmse_list.append(
            np.sqrt(
                mean_squared_error(y_val_inv.flatten(), y_pred_inv.flatten())
            )
        )
        mae_list.append(
            mean_absolute_error(y_val_inv.flatten(), y_pred_inv.flatten())
        )

    X_test, y_test = prepare_rf_data(
        df_test_norm.join(df_y_test_norm),
        lookback,
        forecast_horizon,
        input_columns,
        target_col,
    )

    rf_final = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    rf_final.fit(X_all, y_all)

    y_pred_test = rf_final.predict(X_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred_test)
    y_true_inv = scaler_y.inverse_transform(y_test)

    mean_r2 = float(np.mean(r2_list))
    mean_rmse = float(np.mean(rmse_list))
    mean_mae = float(np.mean(mae_list))

    return mean_r2, mean_rmse, mean_mae, y_true_inv, y_pred_inv


def plot_rf_results(
    plant: str,
    scenario: str,
    lookback: int,
    forecast_horizon: int,
    r2: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Generate scatter, residual histogram and temporal comparison plots
    for Random Forest results.
    """
    residuals = y_true - y_pred
    n_samples = min(350, len(y_true))

    plt.figure(figsize=(5, 3))
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.6, color="blue")
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        color="black",
        linestyle="--",
        linewidth=1,
    )
    plt.xlabel("Observed hourly Ppv")
    plt.ylabel("Predicted hourly Ppv")
    plt.title(
        f"Scatter - {plant} - {scenario} - RF\n"
        f"Window: {lookback}h, Horizon: {forecast_horizon}h, R²: {r2:.3f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 3))
    sns.histplot(residuals[:, 0], bins=50, kde=True, color="blue")
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(
        f"Residuals histogram - {plant} - {scenario} - RF\n"
        f"Window: {lookback}h, Horizon: {forecast_horizon}h, R²: {r2:.3f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(
        range(n_samples),
        y_true[:n_samples, 0],
        color="blue",
        label="Observed",
    )
    plt.plot(
        range(n_samples),
        y_pred[:n_samples, 0],
        color="red",
        linestyle="--",
        label="Predicted",
    )
    plt.xlabel("Samples")
    plt.ylabel("Hourly Ppv (kWp)")
    plt.title(
        f"Temporal comparison - {plant} - {scenario} - RF\n"
        f"Window: {lookback}h, Horizon: {forecast_horizon}h, R²: {r2:.3f}"
    )
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_rf_heatmaps(results_by_scenario: Dict[str, pd.DataFrame]) -> None:
    """
    Plot heatmaps of R² for each scenario and plant vs lookback window.
    """
    for scenario, df_scenario in results_by_scenario.items():
        heatmap_data = df_scenario.pivot_table(
            index="Usina",
            columns="Janela Histórica (horas)",
            values="R²",
            aggfunc="mean",
        )

        plt.figure(figsize=(8, 5))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            vmin=0.4,
            vmax=1.0,
            cbar_kws={"label": "R²"},
        )
        plt.title(f"R² heatmap - {scenario} - RF")
        plt.xlabel("Lookback window (hours)")
        plt.ylabel("Plant")
        plt.tight_layout()
        plt.show()
