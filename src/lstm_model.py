"""
LSTM model utilities for PV forecasting.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


PLANTS = ["Usina_1", "Usina_2", "Usina_3", "Usina_4", "Usina_5"]
DISTANCES = [151.9, 108.5, 62.7, 30.1, 4.0]
LOOKBACK_WINDOWS = [1, 6, 12, 24, 48, 72]
FORECAST_HORIZON = 24


def prepare_lstm_data(
    df: pd.DataFrame,
    lookback: int,
    forecast_horizon: int,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build 3D input samples for LSTM with a lookback window
    and multi-step forecast horizon.
    """
    X, y = [], []

    for i in range(len(df) - lookback - forecast_horizon):
        X.append(df.iloc[i : i + lookback][feature_cols].values)
        y.append(
            df.iloc[
                i + lookback : i + lookback + forecast_horizon
            ][target_col].values
        )

    return np.array(X), np.array(y)


def build_lstm_model(
    lookback: int,
    n_features: int,
    forecast_horizon: int,
    neurons: int = 15,
    activation: str = "tanh",
    dropout_rate: float = 0.0,
) -> Sequential:
    """
    Build and compile an LSTM model for multi-step regression.
    """
    model = Sequential()
    model.add(
        LSTM(
            neurons,
            activation=activation,
            return_sequences=False,
            input_shape=(lookback, n_features),
        )
    )
    model.add(Dropout(dropout_rate))
    model.add(Dense(forecast_horizon))
    model.compile(optimizer=Adam(), loss="mean_squared_error")
    return model


def train_evaluate_lstm(
    df: pd.DataFrame,
    lookback: int,
    forecast_horizon: int,
    input_columns: List[str],
    target_col: str,
    batch_size: int = 16,
    epochs: int = 64,
    neurons: int = 15,
    activation: str = "tanh",
    dropout_rate: float = 0.0,
    n_splits: int = 5,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Train and evaluate an LSTM model with K-fold cross-validation.

    Returns metrics averaged across folds and final test predictions.
    """
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_norm = scaler_X.fit_transform(df_train[input_columns])
    y_train_norm = scaler_y.fit_transform(df_train[[target_col]])

    X_test_norm = scaler_X.transform(df_test[input_columns])
    y_test_norm = scaler_y.transform(df_test[[target_col]])

    df_train_norm = pd.DataFrame(X_train_norm, columns=input_columns)
    df_y_train_norm = pd.DataFrame(y_train_norm, columns=[target_col])

    df_test_norm = pd.DataFrame(X_test_norm, columns=input_columns)
    df_y_test_norm = pd.DataFrame(y_test_norm, columns=[target_col])

    X_train_full, y_train_full = prepare_lstm_data(
        df_train_norm.join(df_y_train_norm),
        lookback,
        forecast_horizon,
        input_columns,
        target_col,
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores, rmse_scores, mae_scores = [], [], []

    for train_idx, val_idx in kf.split(X_train_full):
        model = build_lstm_model(
            lookback=lookback,
            n_features=len(input_columns),
            forecast_horizon=forecast_horizon,
            neurons=neurons,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        model.fit(
            X_train_full[train_idx],
            y_train_full[train_idx],
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_data=(
                X_train_full[val_idx],
                y_train_full[val_idx],
            ),
        )

        y_val_pred = model.predict(X_train_full[val_idx], verbose=0)
        y_val_pred = scaler_y.inverse_transform(y_val_pred)
        y_val_true = scaler_y.inverse_transform(y_train_full[val_idx])

        r2_scores.append(
            r2_score(y_val_true.flatten(), y_val_pred.flatten())
        )
        rmse_scores.append(
            np.sqrt(
                mean_squared_error(
                    y_val_true.flatten(),
                    y_val_pred.flatten(),
                )
            )
        )
        mae_scores.append(
            mean_absolute_error(
                y_val_true.flatten(),
                y_val_pred.flatten(),
            )
        )

    model_final = build_lstm_model(
        lookback=lookback,
        n_features=len(input_columns),
        forecast_horizon=forecast_horizon,
        neurons=neurons,
        activation=activation,
        dropout_rate=dropout_rate,
    )
    model_final.fit(
        X_train_full,
        y_train_full,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    X_test, y_test = prepare_lstm_data(
        df_test_norm.join(df_y_test_norm),
        lookback,
        forecast_horizon,
        input_columns,
        target_col,
    )

    y_pred_test = model_final.predict(X_test, verbose=0)
    y_pred_inv = scaler_y.inverse_transform(y_pred_test)
    y_true_inv = scaler_y.inverse_transform(y_test)

    mean_r2 = float(np.mean(r2_scores))
    mean_rmse = float(np.mean(rmse_scores))
    mean_mae = float(np.mean(mae_scores))

    return mean_r2, mean_rmse, mean_mae, y_true_inv, y_pred_inv


def plot_lstm_results(
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
    for LSTM results.
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
        f"Scatter - {plant} - {scenario} - LSTM\n"
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
        f"Residuals histogram - {plant} - {scenario} - LSTM\n"
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
        f"Temporal comparison - {plant} - {scenario} - LSTM\n"
        f"Window: {lookback}h, Horizon: {forecast_horizon}h, R²: {r2:.3f}"
    )
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_lstm_heatmaps(results_by_scenario: Dict[str, pd.DataFrame]) -> None:
    """
    Plot heatmaps of R² for each scenario and plant vs lookback window (LSTM).
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
        plt.title(f"R² heatmap - {scenario} - LSTM")
        plt.xlabel("Lookback window (hours)")
        plt.ylabel("Plant")
        plt.tight_layout()
        plt.show()
