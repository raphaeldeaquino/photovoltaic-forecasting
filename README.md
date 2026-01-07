# Photovoltaic Power Forecasting Without Local Data

This repository contains the code and datasets used in the experiments reported in the article:

> **“Photovoltaic Power Forecasting Without Local Data: A Spatially-Aware Approach Using Neighboring Plants”** (in review).

The project implements and evaluates **Random Forest (RF)** and **Long Short-Term Memory (LSTM)** models for hourly photovoltaic power forecasting using generation data from neighboring plants and meteorological variables from a regional station, explicitly modeling the distance between plants and the station.

***

## Repository structure

```text
.
├── data/
│   ├── raw/
│   │   ├── Software/                 # Raw Excel files from the PV plants
│   │   └── Clima/                    # Raw meteorological data
│   │       └── Clima.csv
│   └── processed/
│       └── output_horario_filtrado_modificado.xlsx  # Processed hourly dataset
├── data_processing.py                # Data loading and preprocessing functions
├── rf_model.py                       # Random Forest utilities (data prep, training, plots)
├── lstm_model.py                     # LSTM utilities (data prep, training, plots)
└── run_experiments.py                # Main script: builds dataset and runs RF/LSTM experiments
```

- All file paths are handled with `pathlib.Path` and are **relative to the project root**, making the code portable across machines and operating systems.

***

## Data description

### Photovoltaic plants

The dataset contains hourly active power measurements (Ppv, kWp) from **five distributed PV plants** located in the state of Goiás, Brazil.

For each plant, the repository includes:

- Raw inverter exports (Excel) in `data/raw/Software/<plant_folder>/`.  
- Processed hourly series in `data/processed/output_horario_filtrado_modificado.xlsx` (one column per plant: `Usina_1` to `Usina_5`).  

The plants differ in:

- **Location** (municipality in Goiás)  
- **Installed power**  
- **Distance to the meteorological station** (4.0–151.9 km)

### Meteorological data

Meteorological variables are obtained from a single INMET station in Goiânia (Brazil), providing hourly measurements of:

- Air temperature  
- Relative humidity  
- Wind speed  
- Global solar radiation  
- Precipitation  

After preprocessing, the following columns are used:

- `Temp_Ins_C`  
- `Umi_Ins_%`  
- `Vel_Ven_ms`  
- `Radiacao_KJ_m2` (optionally shifted in time to improve alignment)  
- `Chuva_mm`  

The same processed file also includes, for each plant, the scalar feature:

- `Distancia_Usina_i` (distance in km between plant *i* and the meteorological station).

***

## Methodology overview

The repository reproduces the experimental pipeline described in the paper, including:

- Construction of **hourly** datasets from raw inverter and weather files.  
- Evaluation of **three input scenarios**:
  - **Scenario 1:** neighboring plants’ generation + meteorological variables + distances.  
  - **Scenario 2:** only neighboring plants’ generation (no meteorology).  
  - **Scenario 3:** only meteorological variables + distances (no neighboring plants’ generation).  
- Comparison of:
  - **Random Forest (RF)** models (multi-step, 24 h ahead).  
  - **LSTM** models (multi-step, up to 48–72 h windows).  

Performance is assessed using:

- **R²** (coefficient of determination)  
- **RMSE** (root mean square error)  
- **MAE** (mean absolute error)

The code includes routines to:

- Prepare supervised learning samples with different **historical windows** (1, 6, 12, 24, 48, 72 h).  
- Perform **5-fold cross-validation** for both RF and LSTM.  
- Generate:
  - Scatter plots of predicted vs. observed Ppv.  
  - Residual histograms.  
  - Temporal comparison plots (time series).  
  - Heatmaps of R² by plant and historical window for each scenario.

***

## Installation and requirements

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

A typical `requirements.txt` for this project includes:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
```

Adjust versions according to your environment or the paper’s experimental setup.

***

## How to run

1. **Organize the data**

   Place the raw files under `data/raw`:

   - PV plant Excel files in:
     - `data/raw/Software/Usina 1 - Carmo do Rio Verde/`
     - `data/raw/Software/Usina 2 - Santa Cruz de Goiás/`
     - `data/raw/Software/Usina 3 - Gameleira de Goiás/`
     - `data/raw/Software/Usina 4 - Trindade/`
     - `data/raw/Software/Usina 5 - Goiânia/`
   - Meteorological CSV file in:
     - `data/raw/Clima/Clima.csv`

2. **Run the full pipeline**

   From the repository root:

   ```bash
   python run_experiments.py
   ```

   This will:

   - Read and preprocess the raw plant and weather data.  
   - Generate `data/processed/output_horario_filtrado_modificado.xlsx`.  
   - Run RF and LSTM experiments for all scenarios and historical windows.  
   - Produce the figures used in the analysis (scatter, residuals, temporal plots, and heatmaps).

***

## Reproducibility and article status

- The code and datasets in this repository are provided **to ensure reproducibility** of the results presented in the article.
- The manuscript is currently **under peer review**; details (e.g., hyperparameter grids, additional ablation studies or figures) may be updated to match the final accepted version.  
- If you use this repository, please cite the article once it is published; a BibTeX entry will be added here after acceptance.

***

## License and data usage

- The article is under an **open access** license (CC BY) in the target journal, but the **raw plant data are private** and are shared here solely for research and reproducibility purposes. Redistribution or commercial use of the raw data may require prior authorization from the data owners.
- The code in this repository can be reused and adapted for academic purposes; please provide proper attribution to the authors and the article.

***

## Contact

For questions about the code, data, or experiments, please contact the corresponding author listed in the manuscript (Raphael de Aquino Gomes) or the first author (Leonardo Alves Messias).