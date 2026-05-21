# Cosmographic Analysis

Testing Cosmological Isotropy through a Cosmographic approach using Pantheon+ supernova data and the Hemispheric Comparison method.

## Overview

This project tests the cosmological principle — the assumption that the universe is isotropic and homogeneous on large scales — by analyzing the distribution of Type Ia supernovae from the Pantheon+ catalog. The analysis uses a **cosmographic** (model-independent) approach to:

1. Construct all-sky maps of the Hubble constant ($h_0$) and deceleration parameter ($q_0$)
2. Quantify anisotropy in these maps using the Hemispheric Comparison method
3. Compare observed anisotropy against Monte Carlo simulations assuming isotropy (ISO) and ΛCDM

## Project Structure

```
cosmographic_analysis/
├── README.md                         # This file
├── pyproject.toml                    # Package build configuration
├── requirements.txt                  # Python dependencies
├── config.yaml                       # Pipeline configuration
│
├── src/cosmographic_analysis/        # Python package
│   ├── __init__.py
│   ├── config.py                     # Configuration loader
│   ├── data_loader.py                # Pantheon+ data loading
│   ├── cosmology.py                  # Cosmographic distance formulas
│   ├── coordinates.py                # HEALPix coordinate utilities
│   ├── hemispheric_comparison.py     # Hemispheric comparison (core analysis)
│   ├── dw_statistic.py               # Durbin-Watson statistic
│   ├── statistics.py                 # Gaussian fitting + MC statistics
│   ├── maps.py                       # Sky map generation + loading
│   ├── anisotropy.py                 # Max anisotropy direction finder
│   └── plotting/
│       └── histograms.py             # Histogram plotting
│
├── scripts/                          # CLI entry points
│   ├── run_pipeline.py               # Full analysis pipeline
│   ├── run_synthetic_iso.py          # Isotropic simulations
│   └── run_synthetic_lcdm.py         # LCDM simulations
│
├── notebooks/                        # Jupyter notebooks
│   ├── main.ipynb                    # Main analysis (original entry point)
│   ├── synthetic_data.ipynb          # Synthetic data generation demo
│   ├── luminose_distance.ipynb       # Luminosity distance visualization
│   └── luminous_distanceLCDM.ipynb   # ΛCDM distance calculation
│
├── datos/                            # Input data (Pantheon+ catalog)
├── compilations/                     # Output: numerical results
├── figures/                          # Output: sky maps
├── histograms/                       # Output: histogram figures
└── tables/                           # Output: summary tables
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd cosmographic_analysis

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

## Usage

### Run the full pipeline (CLI)

```bash
python scripts/run_pipeline.py
```

With a custom configuration:

```bash
python scripts/run_pipeline.py --config my_config.yaml
```

### Run individual components

```bash
# Isotropic simulations only
python scripts/run_synthetic_iso.py

# LCDM simulations only
python scripts/run_synthetic_lcdm.py
```

### Run the Jupyter notebook

```bash
jupyter notebook notebooks/main.ipynb
```

## Configuration

All pipeline parameters are defined in `config.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `nside` | 16 | HEALPix resolution parameter |
| `h0f` | 0.7304 | Fiducial Hubble constant (Pantheon+ calibration) |
| `q0f` | -0.574 | Fiducial deceleration parameter |
| `zup` | 0.1 | Upper redshift bound |
| `zdown` | 0.01 | Lower redshift bound |
| `repetitions` | 500 | Monte Carlo iterations |
| `prefix_name` | [SH0ES_CALIB] | Output file naming prefix |

## Pipeline Steps

The full pipeline (`scripts/run_pipeline.py`) executes:

1. **Data loading** — Load Pantheon+ SN data, filter by redshift range, build covariance matrix
2. **HEALPix grid** — Define directions for hemispheric comparison
3. **Hemispheric comparison** — For each HEALPix direction, split sky into two hemispheres, fit $h_0$ (fixing $q_0$) and $q_0$ (fixing $h_0$)
4. **Anisotropy measurement** — Compute $\Delta h_0$ and $\Delta q_0$ between opposite hemispheres
5. **Sky maps** — Generate all-sky maps of best-fit parameters
6. **Durbin-Watson statistic** — Test for spatial autocorrelation of residuals
7. **ISO simulations** — 500 MC realizations with isotropic SN positions
8. **LCDM simulations** — 500 MC realizations with ΛCDM-consistent distance moduli
9. **Statistical comparison** — Fit Gaussians to MC distributions, compute p-values

## Dependencies

- Python ≥ 3.10
- numpy, scipy, pandas — numerical computing
- healpy — HEALPix spherical analysis
- matplotlib — plotting
- pyyaml — configuration

## License

[License information]
