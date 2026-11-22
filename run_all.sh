#!/usr/bin/env bash
# Full pipeline. Re-running skips files already downloaded.
# Usage: ./run_all.sh [path/to/srtm_dem.tif]
set -e
cd "$(dirname "$0")"
DEM="${1:-$HOME/Downloads/merged_dem_middle_east_fixed.tif}"

# 1. Data
python 03_get_obs.py events      # station METARs around the two storms (IEM)
python 03_get_obs.py clim        # 2016-2025 METAR climatology for thresholds (slow, run once)
python 01_get_ecmwf.py           # IFS and AIFS Single (ECMWF open data on AWS)
python 02_get_noaa.py            # GraphCast, Aurora, Pangu-Weather (NOAA AI archive on AWS)
python 08b_get_era5_arco.py      # ERA5 from the public ARCO-ERA5 copy (no key needed)

# 2. Verification and diagnostics
python 04_verify.py              # pairs, scores, contingency tables, peak wind, bootstrap
python 09_era5_diagnostics.py    # activity ratio, kinetic-energy spectra, synoptic maps

# 3. Paper figures
python 10_paper_figures.py "$DEM"

# 4. Supplementary surface index (needs the DEM and 06_get_soil_ndvi.py)
python 06_get_soil_ndvi.py
python 05_susceptibility.py "$DEM"
python 07_fig_susceptibility.py

# 5. Robustness checks (CSI bootstrap, threshold sensitivity)
python 12_robustness.py
python 11_fig_workflow.py
