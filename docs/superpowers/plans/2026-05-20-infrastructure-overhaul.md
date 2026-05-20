# Infrastructure Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the project into a clean, documented, pip-installable package with CLI pipeline scripts that reproduce the notebook logic.

**Architecture:** Move all existing `.py` modules into a `src/cosmographic_analysis/` installable package. Create a `config.yaml` for centralized parameters. Create CLI scripts under `scripts/` that replicate the full `main.ipynb` pipeline from data loading through statistics. Move notebooks to `notebooks/`. Replace `README.org` with comprehensive `README.md`. No changes to computational logic — only infrastructure.

**Tech Stack:** Python 3.10+, numpy, scipy, healpy, matplotlib, pandas, pyyaml, tqdm

**Plan location:** `docs/superpowers/plans/2026-05-20-infrastructure-overhaul.md`

---
## File Map

### Files to Create
| File | Purpose |
|---|---|
| `src/cosmographic_analysis/__init__.py` | Package init |
| `src/cosmographic_analysis/config.py` | YAML config loader |
| `src/cosmographic_analysis/data_loader.py` | Pantheon+ data loading wrapper |
| `src/cosmographic_analysis/plotting/__init__.py` | Sub-package init |
| `pyproject.toml` | Package build metadata |
| `requirements.txt` | Pinned dependencies |
| `config.yaml` | Centralized pipeline parameters |
| `scripts/run_pipeline.py` | Full CLI pipeline entry point |
| `scripts/run_synthetic_iso.py` | Isotropic simulations CLI |
| `scripts/run_synthetic_lcdm.py` | LCDM simulations CLI |
| `README.md` | Full project documentation |

### Files to Move (copy then delete originals)
| Source | Destination |
|---|---|
| `cosmology.py` | `src/cosmographic_analysis/cosmology.py` |
| `healpix_vectors.py` | `src/cosmographic_analysis/coordinates.py` |
| `hem_comp_functions.py` | `src/cosmographic_analysis/hemispheric_comparison.py` |
| `dw_statistic.py` | `src/cosmographic_analysis/dw_statistic.py` |
| `fit_gaussian.py` | `src/cosmographic_analysis/statistics.py` (merged) |
| `get_statistics.py` | `src/cosmographic_analysis/statistics.py` (merged) |
| `generate_map.py` | `src/cosmographic_analysis/maps.py` (merged) |
| `loadmap.py` | `src/cosmographic_analysis/maps.py` (merged) |
| `max_anisotropy.py` | `src/cosmographic_analysis/anisotropy.py` |
| `generate_histograms.py` | `src/cosmographic_analysis/plotting/histograms.py` |
| `main.ipynb` | `notebooks/main.ipynb` |
| `synthetic_data.ipynb` | `notebooks/synthetic_data.ipynb` |
| `luminose_distance.ipynb` | `notebooks/luminose_distance.ipynb` |
| `luminous_distanceLCDM.ipynb` | `notebooks/luminous_distanceLCDM.ipynb` |

### Files to Modify
| File | Change |
|---|---|
| `.gitignore` | Add `__pycache__/`, `*.egg-info/`, `.Python`, `*.pyc` |

### Files to Delete (after migration)
| File | Reason |
|---|---|
| `cosmology.py` | Moved to package |
| `healpix_vectors.py` | Moved to package as coordinates.py |
| `hem_comp_functions.py` | Moved to package |
| `dw_statistic.py` | Moved to package |
| `fit_gaussian.py` | Merged into statistics.py |
| `get_statistics.py` | Merged into statistics.py |
| `generate_map.py` | Merged into maps.py |
| `loadmap.py` | Merged into maps.py |
| `max_anisotropy.py` | Moved to package as anisotropy.py |
| `generate_histograms.py` | Moved to plotting/ |
| `README.org` | Replaced by README.md |

---
## Task 1: Create package skeleton and build files

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/cosmographic_analysis/__init__.py`
- Create: `src/cosmographic_analysis/plotting/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "cosmographic_analysis"
version = "1.0.0"
description = "Testing Cosmological Isotropy through a Cosmographic approach"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "scipy",
    "healpy",
    "matplotlib",
    "pandas",
    "pyyaml",
    "tqdm",
]

[project.scripts]
cosmographic-pipeline = "cosmographic_analysis.scripts.run_pipeline:main"

[tool.setuptools.packages.find]
where = ["src"]
```

Run: `mkdir -p src/cosmographic_analysis/plotting`

- [ ] **Step 2: Create requirements.txt**

Write `requirements.txt` with pinned versions:
```
numpy>=1.23.0
scipy>=1.10.0
healpy>=1.16.0
matplotlib>=3.7.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.64.0
```

- [ ] **Step 3: Create __init__.py files**

Write `src/cosmographic_analysis/__init__.py`:
```python
"""Cosmographic Analysis Package.

Testing Cosmological Isotropy through a Cosmographic approach
using Pantheon+ supernova data and hemispheric comparison.
"""
```

Write `src/cosmographic_analysis/plotting/__init__.py` (empty).

- [ ] **Step 4: Update .gitignore**

Write `.gitignore`:
```
# Byte-compiled
__pycache__/
*.py[cod]

# Distribution / packaging
*.egg-info/
.Python
*.so

# Jupyter
.ipynb_checkpoints/

# Environment
.env
venv/
.venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 5: Verify package skeleton**

Run: `pip install -e . 2>&1 | tail -5`
Expected: no errors, package installed in editable mode

Run: `python -c "import cosmographic_analysis; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "infra: add package skeleton with pyproject.toml and build files"
```

---
## Task 2: Move existing modules into package (part 1 — simple relocations)

**Files:**
- Create: `src/cosmographic_analysis/cosmology.py`
- Create: `src/cosmographic_analysis/coordinates.py`
- Create: `src/cosmographic_analysis/anisotropy.py`
- Delete: `cosmology.py`
- Delete: `healpix_vectors.py`
- Delete: `max_anisotropy.py`

These are pure relocations with no import changes (they import only stdlib/healpy/numpy).

- [ ] **Step 1: Copy cosmology.py into package**

Copy `cosmology.py` to `src/cosmographic_analysis/cosmology.py` with the import updated:

```python
import numpy as np

def dl(z: float, h0: float, q0: float) -> float:
    y = z / (z + 1.0)
    return (2997.92458 / h0) * (y + (3.0 - q0) * np.power(y, 2) / 2.0)

def mu(z: float, h0: float, q0: float) -> float:
    return 5.0 * np.log10(dl(z, h0, q0)) + 25.0
```

Verify: `python -c "from cosmographic_analysis.cosmology import mu; print(mu(0.05, 0.73, -0.57))"` — should print a number.

- [ ] **Step 2: Copy healpix_vectors.py as coordinates.py**

Copy `healpix_vectors.py` to `src/cosmographic_analysis/coordinates.py`. Content is identical except the file header comment changes. No import changes needed.

Verify: `python -c "from cosmographic_analysis.coordinates import DecRa2Cartesian; print('OK')"`

- [ ] **Step 3: Copy max_anisotropy.py as anisotropy.py**

Copy `max_anisotropy.py` to `src/cosmographic_analysis/anisotropy.py`. Update the import:

```python
import numpy as np
import healpy as hp
from cosmographic_analysis.coordinates import get_healpix_vectors
```

Content of `get_max_anisotropy` stays exactly the same (redundant lines and all — no logic changes yet).

- [ ] **Step 4: Delete original files from root**

```bash
rm cosmology.py healpix_vectors.py max_anisotropy.py
```

- [ ] **Step 5: Verify imports work end-to-end**

Run: `python -c "
from cosmographic_analysis.cosmology import dl, mu
from cosmographic_analysis.coordinates import get_healpix_vectors, DecRa2Cartesian, IndexToDecRa, DecRaToIndex
from cosmographic_analysis.anisotropy import get_max_anisotropy
print('All imports OK')
"`

Expected: `All imports OK`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "infra: move cosmology, healpix_vectors, max_anisotropy into package"
```

---
## Task 3: Move hemispheric comparison and DW statistic modules

**Files:**
- Create: `src/cosmographic_analysis/hemispheric_comparison.py`
- Create: `src/cosmographic_analysis/dw_statistic.py`
- Delete: `hem_comp_functions.py`
- Delete: `dw_statistic.py`

- [ ] **Step 1: Copy hem_comp_functions.py into package**

Copy `hem_comp_functions.py` to `src/cosmographic_analysis/hemispheric_comparison.py`. Update the import:

```python
from cosmographic_analysis.cosmology import mu
```

Everything else (all function bodies, `datos` tuple indexing, `save` parameter logic, file naming conventions, etc.) stays identical.

- [ ] **Step 2: Copy dw_statistic.py into package**

Copy `dw_statistic.py` to `src/cosmographic_analysis/dw_statistic.py`. Update the import:

```python
from cosmographic_analysis.cosmology import mu
```

- [ ] **Step 3: Delete originals**

```bash
rm hem_comp_functions.py dw_statistic.py
```

- [ ] **Step 4: Verify imports**

Run: `python -c "
from cosmographic_analysis.hemispheric_comparison import hem_h0, hem_q0, multi_hem_map, exec_map
from cosmographic_analysis.dw_statistic import entire_dw, hemispheric_dw, total_dw
print('All imports OK')
"`

Expected: `All imports OK`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "infra: move hemispheric comparison and DW statistics into package"
```

---
## Task 4: Consolidate statistics and map modules

**Files:**
- Create: `src/cosmographic_analysis/statistics.py`
- Create: `src/cosmographic_analysis/maps.py`
- Delete: `fit_gaussian.py`
- Delete: `get_statistics.py`
- Delete: `generate_map.py`
- Delete: `loadmap.py`

These consolidate related functions into single files. No logic changes.

- [ ] **Step 1: Create statistics.py (merge fit_gaussian.py + get_statistics.py)**

Content is the concatenation of both files. Imports stay the same (numpy, scipy). No changes to function bodies. Document at top:

```python
"""Statistical utilities for the cosmographic analysis.

Includes Gaussian fitting and Monte Carlo p-value computation.
"""
```

Functions moved in:
- `gaussian_likelihood(params, data)` → from fit_gaussian.py (unchanged)
- `fit_gaussian(data)` → from fit_gaussian.py (unchanged)
- `map_statistics(h0, q0, delta_h0_data_max, delta_q0_data_max)` → from get_statistics.py (unchanged)
- `mc_statistics(maximum_anisotropy_data, maximum_anisotropy_mc)` → from get_statistics.py (unchanged)

- [ ] **Step 2: Create maps.py (merge generate_map.py + loadmap.py)**

Content is the concatenation of both files. Imports remain (numpy, healpy). No changes to function bodies.

Functions moved in:
- `generate_map(nside, theta, phi, h0, q0)` → from generate_map.py (unchanged)
- `load_hubble_data(file_path)` → from loadmap.py (unchanged)

Also add the orphan `load_map_old` function from `hem_comp_functions.py` (it was a dead utility at line 283-287):

```python
# Legacy map loader (kept for backward compatibility)
def load_map_old(file_path: str):
    data = np.loadtxt(file_path, usecols=(0, 2), skiprows=4)
    h0 = data[:, 0]
    q0 = data[:, 1]
    return h0, q0
```

- [ ] **Step 3: Delete originals**

```bash
rm fit_gaussian.py get_statistics.py generate_map.py loadmap.py
```

- [ ] **Step 4: Verify imports**

Run: `python -c "
from cosmographic_analysis.statistics import fit_gaussian, mc_statistics, map_statistics
from cosmographic_analysis.maps import generate_map, load_hubble_data
print('All imports OK')
"`

Expected: `All imports OK`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "infra: consolidate statistics and map modules into package"
```

---
## Task 5: Move histogram plotting module

**Files:**
- Create: `src/cosmographic_analysis/plotting/histograms.py`
- Delete: `generate_histograms.py`

- [ ] **Step 1: Copy generate_histograms.py into plotting package**

Copy `generate_histograms.py` to `src/cosmographic_analysis/plotting/histograms.py`.
No import changes needed (only numpy/matplotlib).

- [ ] **Step 2: Delete original**

```bash
rm generate_histograms.py
```

- [ ] **Step 3: Verify imports**

Run: `python -c "
from cosmographic_analysis.plotting.histograms import plot_histograms, plot_both_histograms
print('All imports OK')
"`

Expected: `All imports OK`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "infra: move histograms plotting into package"
```

---
## Task 6: Create configuration module

**Files:**
- Create: `config.yaml`
- Create: `src/cosmographic_analysis/config.py`

- [ ] **Step 1: Create config.yaml**

Write `config.yaml`:

```yaml
# Cosmographic Analysis Pipeline Configuration
# All parameters that were scattered across notebook cells are centralized here.

data:
  # Paths to Pantheon+ data files (relative to project root)
  lcparam: datos/Pantheon+SH0ES.dat.txt
  cov_matrix: datos/Pantheon+SH0ES_STAT+SYS.cov.txt

parameters:
  # HEALPix resolution
  nside: 16

  # Fiducial cosmological parameters (Pantheon+ calibration)
  h0f: 0.7304
  q0f: -0.574

  # Redshift range
  zup: 0.1
  zdown: 0.01

  # Monte Carlo repetitions
  repetitions: 500

  # File naming prefix
  prefix_name: '[SH0ES_CALIB]'

output:
  # Output directories (relative to project root)
  compilations: compilations/
  figures: figures/
  histograms: histograms/
  tables: tables/
```

- [ ] **Step 2: Create config.py loader**

Write `src/cosmographic_analysis/config.py`:

```python
"""Pipeline configuration loader.

Loads parameters from config.yaml with defaults and CLI overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DataConfig:
    lcparam: str = "datos/Pantheon+SH0ES.dat.txt"
    cov_matrix: str = "datos/Pantheon+SH0ES_STAT+SYS.cov.txt"


@dataclass
class ParametersConfig:
    nside: int = 16
    h0f: float = 0.7304
    q0f: float = -0.574
    zup: float = 0.1
    zdown: float = 0.01
    repetitions: int = 500
    prefix_name: str = "[SH0ES_CALIB]"


@dataclass
class OutputConfig:
    compilations: str = "compilations/"
    figures: str = "figures/"
    histograms: str = "histograms/"
    tables: str = "tables/"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    parameters: ParametersConfig = field(default_factory=ParametersConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file, falling back to defaults."""
    config = Config()

    if config_path is None:
        # Auto-discover config.yaml relative to project root
        candidates = [
            Path("config.yaml"),
            Path(__file__).parent.parent.parent.parent / "config.yaml",
        ]
        for path in candidates:
            if path.exists():
                config_path = str(path)
                break

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        if "data" in raw:
            config.data = DataConfig(**raw["data"])
        if "parameters" in raw:
            config.parameters = ParametersConfig(**raw["parameters"])
        if "output" in raw:
            config.output = OutputConfig(**raw["output"])

    return config


def ensure_output_dirs(config: Config) -> None:
    """Create output directories if they don't exist."""
    for path in [
        config.output.compilations,
        config.output.figures,
        config.output.histograms,
        config.output.tables,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 3: Verify config module**

Run: `python -c "
from cosmographic_analysis.config import load_config, ensure_output_dirs
config = load_config()
print(f'nside={config.parameters.nside}, h0f={config.parameters.h0f}')
"`

Expected: `nside=16, h0f=0.7304`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "infra: add configuration module with YAML-based parameter management"
```

---
## Task 7: Create data loader module

**Files:**
- Create: `src/cosmographic_analysis/data_loader.py`

This wraps the data loading logic from `main.ipynb` cells (cell 4) into a reusable function. Same operations, just packaged.

- [ ] **Step 1: Create data_loader.py**

Write `src/cosmographic_analysis/data_loader.py`:

```python
"""Pantheon+ data loading utilities.

Replicates the exact data loading logic from main.ipynb cell 4.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from cosmographic_analysis.coordinates import DecRa2Cartesian


def load_pantheon_data(
    lcparam_path: str,
    cov_path: str,
    zup: float,
    zdown: float,
) -> Tuple:
    """Load and filter Pantheon+ supernova data.

    Returns the same 'datos' tuple used throughout the pipeline:
    (r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown)

    Also returns individual arrays for convenience.
    """
    lcparam = np.loadtxt(
        lcparam_path, skiprows=1, usecols=(2, 8, 9, 10, 11, 26, 27, 12, 13)
    )
    lcparam_sys = np.loadtxt(cov_path, skiprows=1)

    ind = np.where((lcparam[:, 0] < zup) & (lcparam[:, 0] > zdown))[0]

    zz = lcparam[ind, 0]
    mz = lcparam[ind, 1]
    sigmz = lcparam[ind, 2]
    muz = lcparam[ind, 3]
    sigmuz = lcparam[ind, 4]
    ra = lcparam[ind, 5]
    dec = lcparam[ind, 6]
    muceph = lcparam[ind, 7]
    hostyn = lcparam[ind, 8]

    cov_z = lcparam_sys.reshape(1701, 1701)
    cov_z = cov_z[np.ix_(ind, ind)]
    inv_cov_z = np.linalg.inv(cov_z)
    cov_mat = pd.DataFrame(cov_z, columns=range(len(zz)))

    return zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, inv_cov_z


def build_datos_tuple(
    ra: np.ndarray,
    dec: np.ndarray,
    zz: np.ndarray,
    mz: np.ndarray,
    sigmz: np.ndarray,
    muz: np.ndarray,
    sigmuz: np.ndarray,
    muceph: np.ndarray,
    hostyn: np.ndarray,
    cov_mat: pd.DataFrame,
    h0f: float,
    q0f: float,
    pts: int,
    zup: float,
    zdown: float,
) -> Tuple:
    """Build the 'datos' tuple used by hemispheric comparison functions.

    Replicates main.ipynb cell 4 data packaging.
    """
    r1 = np.column_stack([ra, dec, zz, mz, sigmz, muz, sigmuz, muceph, hostyn])
    v1 = DecRa2Cartesian(dec, ra)
    datos = (r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown)
    return datos
```

- [ ] **Step 2: Verify import**

Run: `python -c "
from cosmographic_analysis.data_loader import load_pantheon_data, build_datos_tuple
print('OK')
"`

Expected: `OK`

(Full integration test happens when the pipeline script runs.)

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "infra: create data loader module wrapping notebook data loading"
```

---
## Task 8: Create pipeline CLI scripts

**Files:**
- Create: `scripts/run_pipeline.py`

This is the key task — it reproduces the entire `main.ipynb` pipeline as a CLI script.

- [ ] **Step 1: Create scripts/run_pipeline.py**

Write `scripts/run_pipeline.py`:

```python
#!/usr/bin/env python3
"""
Full cosmographic analysis pipeline.

Reproduces the entire main.ipynb pipeline as a CLI script.
Usage: python scripts/run_pipeline.py [--config config.yaml]
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import healpy as hp
from IPython.display import clear_output
from tqdm import tqdm

from cosmographic_analysis.config import load_config, ensure_output_dirs
from cosmographic_analysis.data_loader import load_pantheon_data, build_datos_tuple
from cosmographic_analysis.coordinates import get_healpix_vectors, IndexToDecRa
from cosmographic_analysis.hemispheric_comparison import exec_map
from cosmographic_analysis.anisotropy import get_max_anisotropy
from cosmographic_analysis.maps import generate_map
from cosmographic_analysis.dw_statistic import hemispheric_dw, total_dw
from cosmographic_analysis.statistics import fit_gaussian, mc_statistics
from cosmographic_analysis.plotting.histograms import plot_histograms, plot_both_histograms


def run_pipeline(config_path: str):
    config = load_config(config_path)
    ensure_output_dirs(config)
    p = config.parameters
    d = config.data
    o = config.output

    print("=" * 60)
    print("Cosmographic Analysis Pipeline")
    print("=" * 60)
    print(f"  nside={p.nside}, h0f={p.h0f}, q0f={p.q0f}")
    print(f"  redshift range: {p.zdown} < z < {p.zup}")
    print(f"  repetitions: {p.repetitions}")
    print("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────────
    print("\n[1/9] Loading Pantheon+ data...")
    zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, inv_cov_z = \
        load_pantheon_data(d.lcparam, d.cov_matrix, p.zup, p.zdown)
    print(f"  {len(zz)} SNe with {p.zdown} < z < {p.zup}")
    print(f"  Cov matrix shape: {cov_mat.shape}")

    pts = hp.nside2npix(p.nside)
    datos = build_datos_tuple(
        ra, dec, zz, mz, sigmz, muz, sigmuz, muceph, hostyn,
        cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown,
    )
    r1, v1, hostyn_arr, _, _, _, _, _, _ = datos

    # ── Step 2: Get HEALPix vectors ────────────────────────────────────
    print("\n[2/9] Computing HEALPix vectors...")
    npix = hp.nside2npix(p.nside)
    pixel_indices = np.arange(npix)
    healpix_ra, healpix_dec = IndexToDecRa(p.nside, pixel_indices)
    healpix_dirs = get_healpix_vectors(p.nside)

    # ── Step 3: Hemispheric comparison ─────────────────────────────────
    print(f"\n[3/9] Running hemispheric comparison ({len(healpix_dirs)} directions)...")
    results_h0, results_q0 = exec_map(healpix_dirs, datos)
    h0u, h0d, h0u_err, h0d_err = results_h0
    q0u, q0d, q0u_err, q0d_err = results_q0

    h0 = np.concatenate((h0u, h0d))
    h0_err = np.concatenate((h0u_err, h0d_err))
    q0 = np.concatenate((q0u, q0d))
    q0_err = np.concatenate((q0u_err, q0d_err))

    # ── Step 4: Max anisotropy ─────────────────────────────────────────
    print("\n[4/9] Computing maximum anisotropy...")
    hdirs = np.concatenate((np.array(healpix_dirs), -np.array(healpix_dirs)))

    h0_max_err = h0_err[np.argmax(h0)]
    h0_min_err = h0_err[np.argmin(h0)]
    q0_max_err = q0_err[np.argmax(q0)]
    q0_min_err = q0_err[np.argmin(q0)]

    delta_h0_data_err = np.sqrt(h0_max_err**2 + h0_min_err**2)
    delta_q0_data_err = np.sqrt(q0_max_err**2 + q0_min_err**2)

    delta_h0_data = np.abs(np.array(h0u) - np.array(h0d))
    delta_q0_data = np.abs(np.array(q0u) - np.array(q0d))
    delta_h0_data_max = np.max(delta_h0_data)
    delta_q0_data_max = np.max(delta_q0_data)

    print(f"  delta_h0 = {delta_h0_data_max:.6f} +/- {delta_h0_data_err:.6f}")
    print(f"  delta_q0 = {delta_q0_data_max:.6f} +/- {delta_q0_data_err:.6f}")

    # Max anisotropy direction
    bestfit_data = [h0u, h0d, q0u, q0d]
    max_anis_dir = get_max_anisotropy(bestfit_data, healpix_dirs)
    print(f"  Max anisotropy direction (dec, ra) for q0: {max_anis_dir[0]}")

    # ── Step 5: Generate maps ──────────────────────────────────────────
    print("\n[5/9] Generating sky maps...")
    theta = np.arccos(hdirs[:, 2])
    phi = np.radians(180) - np.arctan2(hdirs[:, 1], hdirs[:, 0])

    h0map, q0map = generate_map(p.nside, theta, phi, h0, q0)
    print(f"  Maps generated: h0map ({h0map.size} pixels), q0map ({q0map.size} pixels)")

    # ── Step 6: Durbin-Watson statistic ────────────────────────────────
    print("\n[6/9] Computing Durbin-Watson statistics...")
    max_q0_anis_vec = healpix_dirs[np.argmax(np.abs(delta_q0_data))]
    dw_up, dw_down = hemispheric_dw(max_q0_anis_vec, datos)
    print(f"  DW for max anisotropy direction: North={dw_up:.4f}, South={dw_down:.4f}")

    total_dw_up, total_dw_down, data_dw = total_dw(healpix_dirs, datos)
    print(f"  Mean DW all directions: North={np.mean(total_dw_up):.4f}, South={np.mean(total_dw_down):.4f}")

    # ── Step 7: Synthetic ISO simulation ───────────────────────────────
    print(f"\n[7/9] ISO simulation ({p.repetitions} repetitions)...")
    v1_iso = np.zeros((p.repetitions, len(ra), 3))
    for i in range(p.repetitions):
        vecti = np.random.randn(len(ra), 3)
        vecti /= np.linalg.norm(vecti, axis=1)[:, np.newaxis]
        thetai = np.arccos(vecti[:, 2])
        phii = np.arctan2(vecti[:, 1], vecti[:, 0])
        deci = thetai
        rai = np.pi - phii
        v1i = np.column_stack([
            np.sin(rai)*np.cos(deci),
            np.sin(rai)*np.sin(deci),
            np.cos(rai),
        ])
        v1_iso[i] = v1i

    h0u_iso, h0d_iso, q0u_iso, q0d_iso = [], [], [], []
    for i, v1_it in enumerate(v1_iso):
        clear_output(wait=True)
        print(f"ISO iteration {i+1}/{p.repetitions}")
        datos_lcdm = [r1, v1_it, hostyn_arr, cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown]
        res_h0, res_q0 = exec_map(healpix_dirs, tuple(datos_lcdm))
        h0u_iso.append(np.array(res_h0[0]))
        h0d_iso.append(np.array(res_h0[1]))
        q0u_iso.append(np.array(res_q0[0]))
        q0d_iso.append(np.array(res_q0[1]))

    h0u_iso = np.array(h0u_iso)
    h0d_iso = np.array(h0d_iso)
    q0u_iso = np.array(q0u_iso)
    q0d_iso = np.array(q0d_iso)

    h0m_iso = np.concatenate((h0u_iso, h0d_iso), axis=1)
    q0m_iso = np.concatenate((q0u_iso, q0d_iso), axis=1)
    delta_h0_iso_max = np.max(h0m_iso, axis=1) - np.min(h0m_iso, axis=1)
    delta_q0_iso_max = np.max(q0m_iso, axis=1) - np.min(q0m_iso, axis=1)

    # Save ISO results
    header_iso = (
        f"This is the data for ISO distribution with new set of data for each repetition\n"
        f"{p.repetitions} repetitions, {pts} points q0f= {p.q0f} h0f= {p.h0f}\n\n"
        f"delta_h0_iso_max  delta_q0_iso_max"
    )
    filename_iso = (
        f"{o.compilations}{p.prefix_name}[ISO]({p.h0f}=h0f_{p.q0f}=q0f)"
        f"({p.repetitions}_rep)({pts}_pts)({p.zup}>z>{p.zdown}).txt"
    )
    np.savetxt(filename_iso, np.column_stack([delta_h0_iso_max, delta_q0_iso_max]), header=header_iso)

    # ── Step 8: Synthetic LCDM simulation ──────────────────────────────
    print(f"\n[8/9] LCDM simulation ({p.repetitions} repetitions)...")
    from cosmographic_analysis.cosmology import mu

    mu_fid = np.array([mu(zi, p.h0f, p.q0f) for zi in zz])
    r1_lcdm = np.tile(r1, (p.repetitions, 1, 1))

    for i in range(p.repetitions):
        mu_sample = np.random.normal(mu_fid, sigmuz)
        r1_lcdm[i, :, 5] = mu_sample

    h0um_lcdm, h0dm_lcdm, q0um_lcdm, q0dm_lcdm = [], [], [], []
    for i, r1_it in enumerate(r1_lcdm):
        clear_output(wait=True)
        print(f"LCDM iteration {i+1}/{p.repetitions}")
        datos_lcdm = [r1_it, v1, hostyn_arr, cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown]
        res_h0, res_q0 = exec_map(healpix_dirs, tuple(datos_lcdm))
        h0um_lcdm.append(np.array(res_h0[0]))
        h0dm_lcdm.append(np.array(res_h0[1]))
        q0um_lcdm.append(np.array(res_q0[0]))
        q0dm_lcdm.append(np.array(res_q0[1]))

    h0um_lcdm = np.array(h0um_lcdm)
    h0dm_lcdm = np.array(h0dm_lcdm)
    q0um_lcdm = np.array(q0um_lcdm)
    q0dm_lcdm = np.array(q0dm_lcdm)

    h0m_lcdm = np.concatenate((h0um_lcdm, h0dm_lcdm), axis=1)
    q0m_lcdm = np.concatenate((q0um_lcdm, q0dm_lcdm), axis=1)
    delta_h0_lcdm_max = np.max(h0m_lcdm, axis=1) - np.min(h0m_lcdm, axis=1)
    delta_q0_lcdm_max = np.max(q0m_lcdm, axis=1) - np.min(q0m_lcdm, axis=1)

    # Save LCDM results
    header_lcdm = (
        f"This is the data for LCDM distribution with new set of data for each repetition\n"
        f"{p.repetitions} repetitions, {pts} points q0f= {p.q0f} h0f= {p.h0f}\n\n"
        f"delta_h0_lcdm_max  delta_q0_lcdm_max"
    )
    filename_lcdm = (
        f"{o.compilations}{p.prefix_name}[LCDM]({p.h0f}=h0f_{p.q0f}=q0f)"
        f"({p.repetitions}_rep)({pts}_pts)({p.zup}>z>{p.zdown}).txt"
    )
    np.savetxt(
        filename_lcdm,
        np.column_stack([delta_h0_lcdm_max, delta_q0_lcdm_max]),
        header=header_lcdm,
    )

    # ── Step 9: Fit Gaussians + plot histograms ────────────────────────
    print("\n[9/9] Fitting Gaussians and plotting histograms...")

    # ISO histograms
    x_gauss_h0_iso, y_gauss_h0_iso = fit_gaussian(delta_h0_iso_max)
    x_gauss_q0_iso, y_gauss_q0_iso = fit_gaussian(delta_q0_iso_max)

    data_h0_iso_hist = [delta_h0_iso_max, delta_h0_data_max, x_gauss_h0_iso, y_gauss_h0_iso]
    data_q0_iso_hist = [delta_q0_iso_max, delta_q0_data_max, x_gauss_q0_iso, y_gauss_q0_iso]
    file_name_iso = (
        f"[ISO](hf={p.h0f}_qf={p.q0f})({p.repetitions})_rep_({p.zup}>z>{p.zdown}).png"
    )
    plot_histograms(data_h0_iso_hist, data_q0_iso_hist, filename=f"{o.histograms}{file_name_iso}")

    # LCDM histograms
    x_gauss_h0_lcdm, y_gauss_h0_lcdm = fit_gaussian(delta_h0_lcdm_max)
    x_gauss_q0_lcdm, y_gauss_q0_lcdm = fit_gaussian(delta_q0_lcdm_max)

    data_h0_lcdm_hist = [delta_h0_lcdm_max, delta_h0_data_max, x_gauss_h0_lcdm, y_gauss_h0_lcdm]
    data_q0_lcdm_hist = [delta_q0_lcdm_max, delta_q0_data_max, x_gauss_q0_lcdm, y_gauss_q0_lcdm]
    file_name_lcdm = (
        f"[LCDM](hf={p.h0f}_qf={p.q0f})({p.repetitions})_rep_({p.zup}>z>{p.zdown}).png"
    )
    plot_histograms(
        data_h0_lcdm_hist, data_q0_lcdm_hist,
        titlemarker="LCDM", filename=f"{o.histograms}{file_name_lcdm}",
    )

    # Combined histograms
    data_h0_both = [delta_h0_iso_max, delta_h0_lcdm_max, delta_h0_data_max]
    data_q0_both = [delta_q0_iso_max, delta_q0_lcdm_max, delta_q0_data_max]
    file_name_both = (
        f"[BOTH](hf={p.h0f}_qf={p.q0f})({p.repetitions})_rep_({p.zup}>z>{p.zdown}).png"
    )
    plot_both_histograms(data_h0_both, data_q0_both, filename=f"{o.histograms}{file_name_both}")

    # Statistics
    maximum_anisotropy_data = np.array([delta_h0_data_max, delta_q0_data_max])
    maximum_anisotropy_mc = np.array([
        delta_h0_lcdm_max, delta_q0_lcdm_max,
        delta_h0_iso_max, delta_q0_iso_max,
    ])
    p_values = mc_statistics(maximum_anisotropy_data, maximum_anisotropy_mc)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Cosmographic analysis pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make script executable**

Run: `chmod +x scripts/run_pipeline.py`

- [ ] **Step 3: Verify script parses correctly**

Run: `python -c "import ast; ast.parse(open('scripts/run_pipeline.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "infra: create main pipeline CLI script reproducing main.ipynb"
```

---
## Task 9: Create synthetic data CLI scripts

**Files:**
- Create: `scripts/run_synthetic_iso.py`
- Create: `scripts/run_synthetic_lcdm.py`

These are modular scripts for running only the ISO or LCDM simulation parts.

- [ ] **Step 1: Create scripts/run_synthetic_iso.py**

```python
#!/usr/bin/env python3
"""
Run isotropic synthetic data simulations.

Replicates the ISO simulation section of main.ipynb.
Usage: python scripts/run_synthetic_iso.py [--config config.yaml]
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import healpy as hp
from IPython.display import clear_output

from cosmographic_analysis.config import load_config
from cosmographic_analysis.data_loader import load_pantheon_data, build_datos_tuple
from cosmographic_analysis.coordinates import get_healpix_vectors
from cosmographic_analysis.hemispheric_comparison import exec_map
from cosmographic_analysis.statistics import fit_gaussian
from cosmographic_analysis.plotting.histograms import plot_histograms


def run_iso(config_path: str):
    config = load_config(config_path)
    p = config.parameters
    d = config.data

    print(f"Loading data (z range: {p.zdown} < z < {p.zup})...")
    zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, inv_cov_z = \
        load_pantheon_data(d.lcparam, d.cov_matrix, p.zup, p.zdown)
    pts = hp.nside2npix(p.nside)
    r1 = np.column_stack([ra, dec, zz, mz, sigmz, muz, sigmuz, muceph, hostyn])

    healpix_dirs = get_healpix_vectors(p.nside)

    print(f"Generating {p.repetitions} isotropic realizations...")
    v1_iso = np.zeros((p.repetitions, len(ra), 3))
    for i in range(p.repetitions):
        vecti = np.random.randn(len(ra), 3)
        vecti /= np.linalg.norm(vecti, axis=1)[:, np.newaxis]
        thetai = np.arccos(vecti[:, 2])
        phii = np.arctan2(vecti[:, 1], vecti[:, 0])
        deci = thetai
        rai = np.pi - phii
        v1i = np.column_stack([
            np.sin(rai)*np.cos(deci),
            np.sin(rai)*np.sin(deci),
            np.cos(rai),
        ])
        v1_iso[i] = v1i

    h0u_all, h0d_all, q0u_all, q0d_all = [], [], [], []
    for i, v1_it in enumerate(v1_iso):
        clear_output(wait=True)
        print(f"Iteration {i+1}/{p.repetitions}")
        datos_lcdm = [r1, v1_it, hostyn, cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown]
        res_h0, res_q0 = exec_map(healpix_dirs, tuple(datos_lcdm))
        h0u_all.append(np.array(res_h0[0]))
        h0d_all.append(np.array(res_h0[1]))
        q0u_all.append(np.array(res_q0[0]))
        q0d_all.append(np.array(res_q0[1]))

    h0m = np.concatenate(h0u_all + h0d_all, axis=1)
    q0m = np.concatenate(q0u_all + q0d_all, axis=1)
    delta_h0_max = np.max(h0m, axis=1) - np.min(h0m, axis=1)
    delta_q0_max = np.max(q0m, axis=1) - np.min(q0m, axis=1)

    x_h0, y_h0 = fit_gaussian(delta_h0_max)
    x_q0, y_q0 = fit_gaussian(delta_q0_max)
    plot_histograms(
        [delta_h0_max, 0, x_h0, y_h0],
        [delta_q0_max, 0, x_q0, y_q0],
        filename=f"{config.output.histograms}[ISO](hf={p.h0f}_qf={p.q0f}).png",
    )
    print("ISO simulation complete.")


def main():
    parser = argparse.ArgumentParser(description="Run ISO synthetic simulations")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_iso(args.config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create scripts/run_synthetic_lcdm.py**

Same pattern as ISO script but samples distance moduli instead of positions. Follows the LCDM section of `main.ipynb`.

Write `scripts/run_synthetic_lcdm.py`:
```python
#!/usr/bin/env python3
"""
Run LCDM synthetic data simulations.

Replicates the LCDM simulation section of main.ipynb.
Usage: python scripts/run_synthetic_lcdm.py [--config config.yaml]
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import healpy as hp
from IPython.display import clear_output

from cosmographic_analysis.config import load_config
from cosmographic_analysis.data_loader import load_pantheon_data
from cosmographic_analysis.coordinates import DecRa2Cartesian, get_healpix_vectors
from cosmographic_analysis.cosmology import mu
from cosmographic_analysis.hemispheric_comparison import exec_map
from cosmographic_analysis.statistics import fit_gaussian
from cosmographic_analysis.plotting.histograms import plot_histograms


def run_lcdm(config_path: str):
    config = load_config(config_path)
    p = config.parameters
    d = config.data

    print(f"Loading data (z range: {p.zdown} < z < {p.zup})...")
    zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, inv_cov_z = \
        load_pantheon_data(d.lcparam, d.cov_matrix, p.zup, p.zdown)
    v1 = DecRa2Cartesian(dec, ra)
    r1 = np.column_stack([ra, dec, zz, mz, sigmz, muz, sigmuz, muceph, hostyn])
    pts = hp.nside2npix(p.nside)

    healpix_dirs = get_healpix_vectors(p.nside)

    print(f"Generating {p.repetitions} LCDM realizations...")
    mu_fid = np.array([mu(zi, p.h0f, p.q0f) for zi in zz])
    r1_lcdm = np.tile(r1, (p.repetitions, 1, 1))

    for i in range(p.repetitions):
        mu_sample = np.random.normal(mu_fid, sigmuz)
        r1_lcdm[i, :, 5] = mu_sample

    h0um, h0dm, q0um, q0dm = [], [], [], []
    for i, r1_it in enumerate(r1_lcdm):
        clear_output(wait=True)
        print(f"Iteration {i+1}/{p.repetitions}")
        datos_lcdm = [r1_it, v1, hostyn, cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown]
        res_h0, res_q0 = exec_map(healpix_dirs, tuple(datos_lcdm))
        h0um.append(np.array(res_h0[0]))
        h0dm.append(np.array(res_h0[1]))
        q0um.append(np.array(res_q0[0]))
        q0dm.append(np.array(res_q0[1]))

    h0m = np.concatenate(h0um + h0dm, axis=1)
    q0m = np.concatenate(q0um + q0dm, axis=1)
    delta_h0_max = np.max(h0m, axis=1) - np.min(h0m, axis=1)
    delta_q0_max = np.max(q0m, axis=1) - np.min(q0m, axis=1)

    x_h0, y_h0 = fit_gaussian(delta_h0_max)
    x_q0, y_q0 = fit_gaussian(delta_q0_max)
    plot_histograms(
        [delta_h0_max, 0, x_h0, y_h0],
        [delta_q0_max, 0, x_q0, y_q0],
        titlemarker="LCDM",
        filename=f"{config.output.histograms}[LCDM](hf={p.h0f}_qf={p.q0f}).png",
    )
    print("LCDM simulation complete.")


def main():
    parser = argparse.ArgumentParser(description="Run LCDM synthetic simulations")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_lcdm(args.config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Make scripts executable**

Run: `chmod +x scripts/run_synthetic_iso.py scripts/run_synthetic_lcdm.py`

- [ ] **Step 4: Verify syntax**

Run: `python -c "import ast; ast.parse(open('scripts/run_synthetic_iso.py').read()); ast.parse(open('scripts/run_synthetic_lcdm.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "infra: create synthetic data CLI scripts"
```

---
## Task 10: Move notebooks and update imports

**Files:**
- Move: `main.ipynb` → `notebooks/main.ipynb`
- Move: `synthetic_data.ipynb` → `notebooks/synthetic_data.ipynb`
- Move: `luminose_distance.ipynb` → `notebooks/luminose_distance.ipynb`
- Move: `luminous_distanceLCDM.ipynb` → `notebooks/luminous_distanceLCDM.ipynb`

**Critical:** The notebooks' import statements need updating from `from cosmology import mu` to `from cosmographic_analysis.cosmology import mu`, etc. Since .ipynb is JSON, we edit the `source` fields in the cell JSON.

- [ ] **Step 1: Move notebooks**

```bash
mkdir -p notebooks
git mv main.ipynb notebooks/main.ipynb
git mv synthetic_data.ipynb notebooks/synthetic_data.ipynb
git mv luminose_distance.ipynb notebooks/luminose_distance.ipynb
git mv luminous_distanceLCDM.ipynb notebooks/luminous_distanceLCDM.ipynb
```

- [ ] **Step 2: Update imports in main.ipynb**

The notebook has these imports that reference root .py files. Update each cell's source code:

| Old import | New import |
|---|---|
| `from cosmology import mu` | `from cosmographic_analysis.cosmology import mu` |
| `from healpix_vectors import DecRa2Cartesian` | `from cosmographic_analysis.coordinates import DecRa2Cartesian` |
| `from healpix_vectors import get_healpix_vectors` | `from cosmographic_analysis.coordinates import get_healpix_vectors` |
| `from healpix_vectors import IndexToDecRa` | `from cosmographic_analysis.coordinates import IndexToDecRa` |
| `import hem_comp_functions` | `from cosmographic_analysis import hemispheric_comparison` |
| `hem_comp_functions.exec_map(...)` | `hemispheric_comparison.exec_map(...)` |
| `import dw_statistic` | `from cosmographic_analysis import dw_statistic` |
| `dw_statistic.hemispheric_dw(...)` | `dw_statistic.hemispheric_dw(...)` (auto, but ensure module name matches) |
| `import loadmap` | `from cosmographic_analysis.maps import load_hubble_data` |
| `from generate_map import generate_map` | `from cosmographic_analysis.maps import generate_map` |
| `import max_anisotropy` | `from cosmographic_analysis.anisotropy import get_max_anisotropy` |
| `max_anisotropy.get_max_anisotropy(...)` | `get_max_anisotropy(...)` |
| `from fit_gaussian import fit_gaussian` | `from cosmographic_analysis.statistics import fit_gaussian` |
| `import generate_histograms` | `from cosmographic_analysis.plotting import histograms` |
| `from generate_histograms import plot_histograms, plot_both_histograms` | `from cosmographic_analysis.plotting.histograms import plot_histograms, plot_both_histograms` |
| `import get_statistics` | `from cosmographic_analysis.statistics import mc_statistics` |
| `get_statistics.mc_statistics(...)` | `mc_statistics(...)` |

Use sed to replace within the JSON cell sources. The notebook cells store source as arrays of strings. The cleanest approach is a Python script that:
1. Reads the notebook JSON
2. Walks all cells
3. Replaces import strings
4. Writes back

Run this Python script:

```python
import json
from pathlib import Path

REPLACEMENTS = {
    "from cosmology import mu": "from cosmographic_analysis.cosmology import mu",
    "from healpix_vectors import DecRa2Cartesian": "from cosmographic_analysis.coordinates import DecRa2Cartesian",
    "from healpix_vectors import get_healpix_vectors": "from cosmographic_analysis.coordinates import get_healpix_vectors",
    "from healpix_vectors import IndexToDecRa": "from cosmographic_analysis.coordinates import IndexToDecRa",
    "import hem_comp_functions": "from cosmographic_analysis import hemispheric_comparison as hem_comp_functions",
    "import dw_statistic": "from cosmographic_analysis import dw_statistic",
    "import loadmap": "from cosmographic_analysis.maps import load_hubble_data as loadmap",
    "from generate_map import generate_map": "from cosmographic_analysis.maps import generate_map",
    "import max_anisotropy": "from cosmographic_analysis.anisotropy import get_max_anisotropy as max_anisotropy",
    "from fit_gaussian import fit_gaussian": "from cosmographic_analysis.statistics import fit_gaussian",
    "import generate_histograms": "from cosmographic_analysis.plotting import histograms as generate_histograms",
    "from generate_histograms import plot_histograms": "from cosmographic_analysis.plotting.histograms import plot_histograms",
    "from generate_histograms import plot_both_histograms": "from cosmographic_analysis.plotting.histograms import plot_both_histograms",
    "import get_statistics": "from cosmographic_analysis.statistics import mc_statistics as get_statistics",
}

notebook_path = Path("notebooks/main.ipynb")
with open(notebook_path) as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    new_source = []
    for line in cell["source"]:
        for old, new in REPLACEMENTS.items():
            if old in line:
                line = line.replace(old, new)
                break
        new_source.append(line)
    cell["source"] = new_source

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print("main.ipynb imports updated")
```

- [ ] **Step 3: Update imports in other notebooks**

Run a similar script for `synthetic_data.ipynb`, `luminose_distance.ipynb`, and `luminous_distanceLCDM.ipynb`. These have fewer imports but need the same treatment.

For `synthetic_data.ipynb`, it uses `from cosmology import mu` — replace with `from cosmographic_analysis.cosmology import mu`.

For `luminous_distanceLCDM.ipynb`, it doesn't import any custom modules (pure scipy/numpy) — no changes needed.

For `luminose_distance.ipynb`, it doesn't import any custom modules — no changes needed.

- [ ] **Step 4: Verify notebook parses correctly**

Run: `python -c "
import json
with open('notebooks/main.ipynb') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        text = ''.join(cell['source'])
        assert 'from cosmology' not in text or 'cosmographic_analysis' in text, f'Old import found in cell'
print('All imports verified')
"`

Expected: `All imports verified`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "infra: move notebooks to notebooks/ and update import paths"
```

---
## Task 11: Write comprehensive README

**Files:**
- Create: `README.md`
- Delete: `README.org`

- [ ] **Step 1: Write README.md**

Write `README.md`:

```markdown
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

## Citation

If you use this code, please cite the associated paper:

> [Paper reference — add when published]

## License

[License information]
```

- [ ] **Step 2: Delete old README.org**

```bash
git rm README.org
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "docs: replace README.org with comprehensive README.md"
```

---
## Task 12: Final cleanup and verification

**Files:**
- No new files — clean up any leftover root `.py` files

- [ ] **Step 1: Verify no orphaned .py files at root**

Run: `ls *.py`
Expected: empty (no `.py` files in root)

If any remain:
```bash
# Check if they were missed
git status
# Either git rm or move them into the package
```

- [ ] **Step 2: Verify end-to-end import chain**

Run: `python -c "
from cosmographic_analysis.config import load_config
from cosmographic_analysis.data_loader import load_pantheon_data, build_datos_tuple
from cosmographic_analysis.cosmology import dl, mu
from cosmographic_analysis.coordinates import get_healpix_vectors, DecRa2Cartesian, IndexToDecRa
from cosmographic_analysis.hemispheric_comparison import hem_h0, hem_q0, exec_map
from cosmographic_analysis.dw_statistic import entire_dw, hemispheric_dw, total_dw
from cosmographic_analysis.statistics import fit_gaussian, mc_statistics
from cosmographic_analysis.maps import generate_map, load_hubble_data
from cosmographic_analysis.anisotropy import get_max_anisotropy
from cosmographic_analysis.plotting.histograms import plot_histograms, plot_both_histograms
print('All package imports OK')
"`

Expected: `All package imports OK`

- [ ] **Step 3: Verify notebook JSON integrity**

Run: `python -c "
import json
for nb in ['notebooks/main.ipynb', 'notebooks/synthetic_data.ipynb']:
    with open(nb) as f:
        data = json.load(f)
    print(f'{nb}: {len(data[\"cells\"])} cells OK')
"`

Expected: Both notebooks load without JSON errors.

- [ ] **Step 4: Final git status check**

Run: `git status`
Expected: Clean working tree (no untracked files that should be tracked, no leftover root `.py` files)

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "infra: final cleanup after package migration"
```
