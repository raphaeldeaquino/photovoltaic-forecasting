"""
Main script for running PV forecasting experiments (RF and LSTM).

Uses a portable `data/` directory with pathlib.Path.
"""
import time 

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from data_processing import (
    build_hourly_plant_dataframe,
    load_and_process_weather_data,
    merge_plant_and_weather,
    read_filter_merge_columns,
)
from rf_model import (
    DISTANCES as RF_DISTANCES,
    FORECAST_HORIZON as RF_FORECAST_HORIZON,
    LOOKBACK_WINDOWS as RF_LOOKBACK_WINDOWS,
    PLANTS as RF_PLANTS,
    plot_rf_heatmaps,
    plot_rf_results,
    train_evaluate_rf,
)
from lstm_model import (
    DISTANCES as LSTM_DISTANCES,
    FORECAST_HORIZON as LSTM_FORECAST_HORIZON,
    LOOKBACK_WINDOWS as LSTM_LOOKBACK_WINDOWS,
    PLANTS as LSTM_PLANTS,
    plot_lstm_heatmaps,
    plot_lstm_results,
    train_evaluate_lstm,
)

from sklearn.preprocessing import MinMaxScaler


def get_paths() -> Tuple[Path, Path, Path]:
    """
    Build portable paths relative to project root.

    Expects the following structure:

        project_root/
            data/
                raw/
                    Software/...
                    Clima/Clima.csv
                processed/
                    output_horario_filtrado_modificado.xlsx
    """
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    processed_dir.mkdir(parents=True, exist_ok=True)

    return raw_dir, processed_dir, data_dir


def run_random_forest_experiments(df: pd.DataFrame) -> None:
    """
    Run Random Forest experiments for all scenarios, plants and lookback windows.
    """
    df = df.copy()
    df = df.fillna(method="ffill").fillna(method="bfill")
    df["Data"] = pd.to_datetime(df["Data"])

    for i, plant in enumerate(RF_PLANTS):
        df[f"Distancia_{plant}"] = RF_DISTANCES[i]

    scenarios = {
        "Cenário 1": (
            [
                "Temp_Ins_C",
                "Umi_Ins_%",
                "Vel_Ven_ms",
                "Radiacao_KJ_m2",
                "Chuva_mm",
            ]
            + RF_PLANTS
            + [f"Distancia_{plant}" for plant in RF_PLANTS]
        ),
        "Cenário 2": (
            RF_PLANTS
        ),
        "Cenário 3": (
            [
                "Temp_Ins_C",
                "Umi_Ins_%",
                "Vel_Ven_ms",
                "Radiacao_KJ_m2",
                "Chuva_mm",
            ]
            + [f"Distancia_{plant}" for plant in RF_PLANTS]
        ),
    }

    results_by_scenario: Dict[str, pd.DataFrame] = {
        scenario: pd.DataFrame(
            columns=["Usina", "Janela Histórica (horas)", "R²", "RMSE", "MAE"]
        )
        for scenario in scenarios
    }

    pred_dict: Dict[str, Dict[Tuple[str, int], pd.DataFrame]] = {
        scenario: {} for scenario in scenarios
    }
    y_true_dict: Dict[str, Dict[Tuple[str, int], pd.DataFrame]] = {
        scenario: {} for scenario in scenarios
    }

    for scenario, input_cols in scenarios.items():
        print(f"\n=== Random Forest - {scenario} ===")
        scenario_rows: List[List[float]] = []

        for plant_out in RF_PLANTS:
            print(f"Processing {plant_out}...")
            input_columns_scenario = [
                col for col in input_cols if col != plant_out
            ]

            for lookback in RF_LOOKBACK_WINDOWS:
                r2, rmse, mae, y_true, y_pred = train_evaluate_rf(
                    df=df,
                    lookback=lookback,
                    forecast_horizon=RF_FORECAST_HORIZON,
                    input_columns=input_columns_scenario,
                    target_col=plant_out,
                    scaler_X=MinMaxScaler(),
                    scaler_y=MinMaxScaler(),
                )

                scenario_rows.append([plant_out, lookback, r2, rmse, mae])
                pred_dict[scenario][(plant_out, lookback)] = y_pred
                y_true_dict[scenario][(plant_out, lookback)] = y_true

        results_by_scenario[scenario] = pd.DataFrame(
            scenario_rows,
            columns=["Usina", "Janela Histórica (horas)", "R²", "RMSE", "MAE"],
        )

    # Select best R² per plant and plot
    best_results = []
    for plant in RF_PLANTS:
        best_r2 = -float("inf")
        best_entry = None

        for scenario in scenarios:
            df_scenario = results_by_scenario[scenario]
            df_plant = df_scenario[df_scenario["Usina"] == plant]

            if df_plant.empty:
                continue

            idx_best = df_plant["R²"].idxmax()
            r2 = df_plant.loc[idx_best, "R²"]

            if r2 > best_r2:
                best_r2 = r2
                best_entry = {
                    "Usina": plant,
                    "Cenário": scenario,
                    "Janela Histórica (horas)": int(
                        df_plant.loc[idx_best, "Janela Histórica (horas)"]
                    ),
                    "R²": float(r2),
                }

        if best_entry is not None:
            plant = best_entry["Usina"]
            scenario = best_entry["Cenário"]
            lookback = best_entry["Janela Histórica (horas)"]
            y_pred = pred_dict[scenario][(plant, lookback)]
            y_true = y_true_dict[scenario][(plant, lookback)]

            plot_rf_results(
                plant=plant,
                scenario=scenario,
                lookback=lookback,
                forecast_horizon=RF_FORECAST_HORIZON,
                r2=best_entry["R²"],
                y_true=y_true,
                y_pred=y_pred,
            )

            best_results.append(best_entry)

    plot_rf_heatmaps(results_by_scenario)


def run_lstm_experiments(df: pd.DataFrame) -> None:
    """
    Run LSTM experiments for all scenarios, plants and lookback windows.
    """
    df = df.copy()
    df = df.fillna(method="ffill").fillna(method="bfill")
    df["Data"] = pd.to_datetime(df["Data"])

    for i, plant in enumerate(LSTM_PLANTS):
        df[f"Distancia_{plant}"] = LSTM_DISTANCES[i]

    scenarios = {
        "Cenário 1": (
            [
                "Temp_Ins_C",
                "Umi_Ins_%",
                "Vel_Ven_ms",
                "Radiacao_KJ_m2",
                "Chuva_mm",
            ]
            + LSTM_PLANTS
            + [f"Distancia_{plant}" for plant in LSTM_PLANTS]
        ),
        "Cenário 2": (
            LSTM_PLANTS
        ),
        "Cenário 3": (
            [
                "Temp_Ins_C",
                "Umi_Ins_%",
                "Vel_Ven_ms",
                "Radiacao_KJ_m2",
                "Chuva_mm",
            ]
            + [f"Distancia_{plant}" for plant in LSTM_PLANTS]
        ),
    }

    batch_size = 16
    epochs = 64
    activation = "tanh"
    dropout_rate = 0.0
    neurons = 15

    results_by_scenario: Dict[str, List[List[float]]] = {
        scenario: [] for scenario in scenarios
    }
    pred_dict: Dict[str, Dict[Tuple[str, int], pd.DataFrame]] = {
        scenario: {} for scenario in scenarios
    }
    y_true_dict: Dict[str, Dict[Tuple[str, int], pd.DataFrame]] = {
        scenario: {} for scenario in scenarios
    }

    total_combinations = (
        len(scenarios) * len(LSTM_PLANTS) * len(LSTM_LOOKBACK_WINDOWS)
    )
    current_combination = 0
    start_time = time.time()

    for scenario, input_cols in scenarios.items():
        for plant_out in LSTM_PLANTS:
            input_columns_scenario = [
                col for col in input_cols if col != plant_out
            ]

            for lookback in LSTM_LOOKBACK_WINDOWS:
                current_combination += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / current_combination
                remaining_s = avg_time * (
                    total_combinations - current_combination
                )
                mins, secs = divmod(remaining_s, 60)

                print(
                    f"\nLSTM {scenario} | {plant_out} | "
                    f"Window {lookback}h [{current_combination}/{total_combinations}]"
                )
                print(
                    f"Elapsed: {elapsed / 60:.1f} min | "
                    f"ETA: ~{int(mins)} min {int(secs)} s remaining"
                )

                r2, rmse, mae, y_true, y_pred = train_evaluate_lstm(
                    df=df,
                    lookback=lookback,
                    forecast_horizon=LSTM_FORECAST_HORIZON,
                    input_columns=input_columns_scenario,
                    target_col=plant_out,
                    batch_size=batch_size,
                    epochs=epochs,
                    neurons=neurons,
                    activation=activation,
                    dropout_rate=dropout_rate,
                )

                results_by_scenario[scenario].append(
                    [plant_out, lookback, r2, rmse, mae]
                )
                pred_dict[scenario][(plant_out, lookback)] = y_pred
                y_true_dict[scenario][(plant_out, lookback)] = y_true

    end_time = time.time()
    print(f"\nTotal LSTM execution time: {end_time - start_time:.1f} seconds")

    results_df_by_scenario: Dict[str, pd.DataFrame] = {}
    for scenario in scenarios:
        results_df_by_scenario[scenario] = pd.DataFrame(
            results_by_scenario[scenario],
            columns=["Usina", "Janela Histórica (horas)", "R²", "RMSE", "MAE"],
        )

    for scenario, df_scenario in results_df_by_scenario.items():
        best_cases = df_scenario.loc[
            df_scenario.groupby("Usina")["R²"].idxmax()
        ]

        for _, row in best_cases.iterrows():
            plant = row["Usina"]
            lookback = int(row["Janela Histórica (horas)"])
            r2 = float(row["R²"])

            y_true = y_true_dict[scenario][(plant, lookback)]
            y_pred = pred_dict[scenario][(plant, lookback)]

            plot_lstm_results(
                plant=plant,
                scenario=scenario,
                lookback=lookback,
                forecast_horizon=LSTM_FORECAST_HORIZON,
                r2=r2,
                y_true=y_true,
                y_pred=y_pred,
            )

    plot_lstm_heatmaps(results_df_by_scenario)


def main() -> None:
    """
    Main entry point for running data processing and experiments.
    """
    raw_dir, processed_dir, _ = get_paths()

    # Paths for input and output
    base_folder_path = raw_dir / "Software"
    weather_csv_path = raw_dir / "Clima" / "Clima.csv"
    output_excel_path = processed_dir / "output_horario_filtrado_modificado.xlsx"

    plant_folders = [
        "Usina 1 - Carmo do Rio Verde",
        "Usina 2 - Santa Cruz de Goiás",
        "Usina 3 - Gameleira de Goiás",
        "Usina 4 - Trindade",
        "Usina 5 - Goiânia",
    ]
    excel_columns = [1, 3]
    power_column_index = 1
    radiation_shift = -4

    print("Reading and aggregating plant data...")
    merged_df = read_filter_merge_columns(
        base_folder_path,
        plant_folders,
        excel_columns,
        power_column_index,
    )

    print("Building hourly plant DataFrame...")
    hourly_df = build_hourly_plant_dataframe(merged_df)

    print("Loading and processing weather data...")
    weather_df = load_and_process_weather_data(weather_csv_path)

    print("Merging plant and weather data...")
    combined_df = merge_plant_and_weather(
        hourly_df,
        weather_df,
        radiation_shift=radiation_shift,
    )

    print(f"Saving combined dataset to: {output_excel_path}")
    combined_df.to_excel(output_excel_path, index=False)

    print("\nLoading combined dataset for experiments...")
    df_experiments = pd.read_excel(output_excel_path)

    print("\nRunning Random Forest experiments...")
    run_random_forest_experiments(df_experiments)

    print("\nRunning LSTM experiments...")
    run_lstm_experiments(df_experiments)


if __name__ == "__main__":
    main()
