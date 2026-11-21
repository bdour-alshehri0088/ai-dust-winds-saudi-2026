# AI and physics-based 10 m wind forecasts for two 2026 dust storms in Saudi Arabia

Code and result tables for the paper
**"Does lower wind-forecast error mean better strong-wind and dust-storm detection? AI and physics-based forecasts for two 2026 dust storms in Saudi Arabia"** (Alshehri et al., in preparation).

Four AI weather models (AIFS Single, GraphCast, Aurora, Pangu-Weather) and ECMWF IFS HRES are verified against METAR observations at nine Saudi airports for the dust storms of 13 January and 3 May 2026, with forecasts initialised up to five days ahead.

## Setup

```bash
conda create -n dusteval -c conda-forge python=3.11 eccodes cfgrib -y
conda activate dusteval
pip install -r requirements.txt
```

## Run

```bash
./run_all.sh /path/to/srtm_dem.tif
```

The DEM is needed only for Figure 1 and the supplementary surface index. All other inputs are downloaded from public archives; files already present are skipped.

| Step | Script | What it does |
| --- | --- | --- |
| Observations | `03_get_obs.py events`, `03_get_obs.py clim` | METAR reports from the Iowa Environmental Mesonet (event windows and 2016–2025 climatology) |
| Forecasts | `01_get_ecmwf.py` | IFS and AIFS Single from ECMWF open data on AWS; only the needed GRIB messages are fetched |
| Forecasts | `02_get_noaa.py` | GraphCast, Aurora, Pangu-Weather from the NOAA AI Weather Prediction archive on AWS; only the domain is read |
| Reanalysis | `08b_get_era5_arco.py` | ERA5 10 m wind, MSLP and soil moisture from ARCO-ERA5 (no account needed) |
| Reanalysis (alternative) | `08_get_era5.py` | Same from the Copernicus CDS; needs a key in `~/.cdsapirc` |
| Verification | `04_verify.py` | Thresholds, forecast–observation pairs, continuous scores, contingency tables (P90), peak-wind ratio, DUP, block bootstrap |
| Smoothing | `09_era5_diagnostics.py` | Activity ratio, kinetic-energy spectra, synoptic maps |
| Figures | `10_paper_figures.py` | Paper figures in `results/figures/` |
| Supplement | `06_get_soil_ndvi.py`, `05_susceptibility.py`, `07_fig_susceptibility.py` | SoilGrids and NOAA CDR VIIRS NDVI; static surface index E = S × G × B |
| Utility | `dem_check.py` | Compares a local DEM tile with Copernicus GLO-90 to identify its source |

Settings (domain, events, stations, initialisations, thresholds) are in `config.py`; shared functions are in `common.py`.

## Results in this repository

`results/` holds the tables behind the paper:

| File | Content | Paper |
| --- | --- | --- |
| `thresholds.csv` | Emission thresholds u_t and P90/P95 per station | Table 9 |
| `scores_by_lead.csv` | Scores by model and forecast day (common cases) | Tables 11, 12, 14; Figures 4, 5, 7 |
| `scores_by_event_lead.csv` | Same, per storm | Table 16 |
| `scores_by_period_lead.csv` | Same, before / during / after the storm day | Section 5.4 |
| `scores_by_lead_AIFS_IFS_all_inits.csv` | AIFS and IFS on all 12 initialisations | Section 5.3 |
| `peak_wind.csv` | Peak-wind ratio and timing per model, event, initialisation, station | Table 13 |
| `bootstrap_vs_ifs.csv` | RMSE difference from IFS with block-bootstrap intervals | Section 5.3 |
| `smoothing_by_lead.csv`, `smoothing_by_event_lead.csv` | Activity and small-scale power ratios | Table 15, Figure 9 |
| `pairs.csv` | Every forecast–observation pair | All |

Final figures are in `results/figures/`.

## Data sources

ECMWF open data (CC-BY-4.0); NOAA AI Weather Prediction archive (Radford et al., 2025); Iowa Environmental Mesonet METAR archive; ERA5 via ARCO-ERA5; CGIAR-CSI SRTM 90 m v4.1; SoilGrids 2.0; NOAA CDR VIIRS NDVI. Country borders: Natural Earth 1:110m (public domain), in `data/basemap/`.

Figures follow the style of [figures4papers](https://github.com/ChenLiu-1996/figures4papers) (CC BY-NC 4.0).

## Contact

Budour Alshehri, National Center for Meteorology, Jeddah, Saudi Arabia.
