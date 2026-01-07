"""
Data loading and preprocessing utilities for PV plants and weather data.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def read_filter_merge_columns(
    base_path: Path,
    plant_folders: List[str],
    excel_columns: List[int],
    power_column_index: int,
) -> pd.DataFrame:
    """
    Read specific columns from multiple Excel files in multiple folders.

    The function:
    - Reads given columns from all Excel files in each plant folder.
    - Drops the first 3 rows (typically non-data header rows).
    - Renames the power column using the plant name to ensure unique names.
    - Concatenates the results by columns.

    Args:
        base_path: Base directory where the plant folders are located.
        plant_folders: List of folder names corresponding to each plant.
        excel_columns: List of column indices to read from each Excel file.
        power_column_index: Index (within excel_columns) of the power column.

    Returns:
        Merged DataFrame with selected columns for all plants.
    """
    merged_df = pd.DataFrame()

    for plant_name in plant_folders:
        dfs = []
        plant_folder_path = base_path / plant_name

        if not plant_folder_path.exists():
            print(f"Folder not found: {plant_name}")
            continue

        for file_path in plant_folder_path.iterdir():
            if not file_path.suffix.lower() in {".xls", ".xlsx"}:
                continue

            try:
                df = pd.read_excel(file_path, usecols=excel_columns)

                # Drop the first three rows if they are not part of the data
                df = df.iloc[3:]

                # Rename columns: power column gets plant name, others get generic names
                new_columns = []
                for i, col in enumerate(df.columns):
                    if i == power_column_index:
                        new_columns.append(f"{plant_name}_power")
                    else:
                        new_columns.append(f"{col}_{i}")
                df.columns = new_columns

                dfs.append(df)
            except Exception as exc:  # noqa: BLE001
                print(f"Error reading file {file_path.name}: {exc}")

        if dfs:
            plant_df = pd.concat(dfs, ignore_index=True)
            merged_df = pd.concat([merged_df, plant_df], axis=1)

    return merged_df


def build_hourly_plant_dataframe(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate plant data to hourly resolution using the maximum power per hour.

    Assumes that the DataFrame has alternating timestamp and power columns
    for each plant.

    Args:
        merged_df: DataFrame with timestamp and power columns for each plant.

    Returns:
        Hourly-resampled DataFrame with one column per plant and a datetime column.
    """
    hourly_df = pd.DataFrame()

    # Process column pairs [timestamp, power] for each plant
    for i in range(0, len(merged_df.columns), 2):
        timestamp_col = merged_df.columns[i]
        power_col = merged_df.columns[i + 1]

        merged_df[timestamp_col] = pd.to_datetime(
            merged_df[timestamp_col],
            errors="coerce",
        )

        hourly_data = (
            merged_df[[timestamp_col, power_col]]
            .set_index(timestamp_col)
            .resample("H")
            .max()
        )

        plant_id = (i // 2) + 1
        hourly_data.rename(
            columns={power_col: f"Usina_{plant_id}"},
            inplace=True,
        )

        if hourly_df.empty:
            hourly_df = hourly_data.copy()
        else:
            hourly_df = hourly_df.join(hourly_data, how="outer")

    hourly_df.fillna(0, inplace=True)
    hourly_df[hourly_df < 0] = 0

    hourly_df.reset_index(inplace=True)
    hourly_df.rename(columns={"index": "Data"}, inplace=True)

    return hourly_df


def load_and_process_weather_data(csv_path: Path) -> pd.DataFrame:
    """
    Load and clean weather data from a CSV file.

    The function:
    - Reads the file skipping initial metadata rows.
    - Converts comma decimals to floats for selected columns.
    - Selects and renames relevant columns.
    - Fills missing values with zero.

    Args:
        csv_path: Path to the weather CSV file.

    Returns:
        Cleaned DataFrame with selected weather variables.
    """
    df_weather = pd.read_csv(csv_path, delimiter=";", skiprows=5)

    df_weather.iloc[:, 2] = (
        df_weather.iloc[:, 2].str.replace(",", ".").astype(float)
    )
    df_weather.iloc[:, 5] = df_weather.iloc[:, 5].astype(float)
    df_weather.iloc[:, 14] = (
        df_weather.iloc[:, 14].str.replace(",", ".").astype(float)
    )
    df_weather.iloc[:, 17] = (
        df_weather.iloc[:, 17].str.replace(",", ".").astype(float)
    )
    df_weather.iloc[:, 18] = (
        df_weather.iloc[:, 18].str.replace(",", ".").astype(float)
    )

    weather_subset = df_weather.iloc[:, [2, 5, 14, 17, 18]].copy()
    weather_subset.columns = [
        "Temp_Ins_C",
        "Umi_Ins_%",
        "Vel_Ven_ms",
        "Radiacao_KJ_m2",
        "Chuva_mm",
    ]

    for col in weather_subset.columns:
        weather_subset[col] = weather_subset[col].fillna(0)

    return weather_subset


def merge_plant_and_weather(
    hourly_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    radiation_shift: int = -4,
) -> pd.DataFrame:
    """
    Merge hourly plant data with weather variables.

    The function:
    - Trims both DataFrames to the same length.
    - Adds weather columns to the plant DataFrame.
    - Optionally shifts the radiation column to adjust temporal alignment.

    Args:
        hourly_df: Hourly plant power DataFrame.
        weather_df: Weather variables DataFrame.
        radiation_shift: Number of rows to shift the radiation column
            (negative shifts upwards).

    Returns:
        Combined DataFrame with power and weather features.
    """
    min_length = min(len(hourly_df), len(weather_df))
    hourly_trimmed = hourly_df.iloc[:min_length].copy()
    weather_trimmed = weather_df.iloc[:min_length].copy()

    hourly_trimmed["Temp_Ins_C"] = weather_trimmed["Temp_Ins_C"].values
    hourly_trimmed["Umi_Ins_%"] = weather_trimmed["Umi_Ins_%"].values
    hourly_trimmed["Vel_Ven_ms"] = weather_trimmed["Vel_Ven_ms"].values
    hourly_trimmed["Radiacao_KJ_m2"] = weather_trimmed[
        "Radiacao_KJ_m2"
    ].shift(radiation_shift).values
    hourly_trimmed["Chuva_mm"] = weather_trimmed["Chuva_mm"].values

    return hourly_trimmed
