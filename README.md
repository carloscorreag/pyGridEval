# pyGrigEval

A Python tool to calculate difference metrics and generate comparison plots between climate grid data and observational station data.
The GUI allows selecting grids, variables, interpolation methods, and evaluation periods.

---

# System Requirements

- Python 3.x
- Libraries: tkinter, netCDF4, pandas, numpy, seaborn, matplotlib, scipy, cartopy, pyproj

Install via pip:

```bash
pip install tkinter netCDF4 pandas numpy seaborn matplotlib scipy cartopy pyproj
```

Or with Conda:

```bash
conda create -n grid_evaluator tk netCDF4 pandas numpy seaborn matplotlib scipy cartopy pyproj
```

> Note: In Conda, `tkinter` is called `tk`.

---

## 🚀 Usage

1. Place your NetCDFs and CSV input files in the same folder as `pyGridEval_gui.py`.

2. Run the script:

```bash
python3 pyGridEval_gui.py
```

3. Use the GUI to select:

   * The variable to analyze (temperature, precipitation, etc.)
   * Grids to compare
   * Interpolation method: nearest neighbor, bilinear, or 15 km relaxation (Cavalleri et al., 2024; https://doi.org/10.1016/j.atmosres.2024.107734) 
   * Period: start and end year
   * Months (all or custom selection)

4. Click **Generate Metrics & Plots**. Results are saved as CSV and PNG files in the current directory.

---

## Supported Grids

The program currently supports the following climate grids:

WRF, ISIMIP-CHELSA, CHIRTS, CHIRPS, ERA5, ERA5-Land, COSMO-REA6, CERRA, CERRA_LAND, EOBS, EOBS_HR, EOBS_LR, CLARA-A3

> Note: To evaluate additional grids or modify the list of grids, you will need to update the grids list in the script pyGridEval_gui.py.


## Input Files

### 1️⃣ NetCDF Grid Files

* Format: `grid_data_GRID_VARIABLE.nc`
* GRID examples: `ISIMIP-CHELSA`, `CHIRTS`, `ERA5`, `CERRA`, etc.
* VARIABLE examples: `temperature`, `precipitation`, `wind_speed`

Example: `grid_data_CERRA_temperature.nc`

> Recommended: all grids should cover the same period for proper comparison.

### 2️⃣ Station CSV Files

* Format: `stations_data_VARIABLE.csv`
* Columns:

```
station_id,latitude,longitude,date,VARIABLE
```

* Dates in `YYYY-MM-DD`
* Values: temperature (°C), precipitation (mm), humidity (%), wind speed (m/s)

Example:

```
station_id,latitude,longitude,date,temperature
1,35.6895,139.6917,1991-01-01,25.0
1,35.6895,139.6917,1991-01-02,25.7
```

* Missing values can be left blank or skipped. Fill values like `-99`, `-999`, `-9999` are ignored.

---

## ✨ Features

* Completeness check of observational series
* Difference metrics between grid values and station observations:

  * Daily Mean Bias
  * Mean Absolute Error
  * Percentile Biases
  * RMSE
  * Correlation
  * Variance Bias
  * Annual cycle comparison
* Saves metrics as CSV and generates maps/plots in PNG
* Boxplots and PDF comparisons for stations vs grids

> ⚠️ Warning: Selecting a variable not present in a grid will cause an error.

---

## 📊 Output

* CSV files for metrics per station per grid
* Annual cycle comparison CSV
* PNG plots:

  * Metric maps per station
  * Boxplots comparing grids
  * PDFs comparing observations vs grids
  * Annual cycle comparison plots

---

## 📩 Contact

For questions or support, please contact: **[ccorreag@aemet.es](mailto:ccorreag@aemet.es)**

---

## 📖 Citation

Correa, C. (2025). **pyGridEval**: A Python tool to calculate difference metrics and generate comparison plots between climate grid data and observational station data. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17366241.svg)](https://doi.org/10.5281/zenodo.17366241)

You can cite all versions by using the DOI: [10.5281/zenodo.17366240](https://doi.org/10.5281/zenodo.17366240)

