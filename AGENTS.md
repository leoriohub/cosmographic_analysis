# Project Instructions for AI Agents

## Environment

- Conda env: `cosmographic_analysis`
- Activate before running any project code: `conda activate cosmographic_analysis`
- Dev install: `pip install -e .` from project root
- All scripts run from project root (`/home/diego/Projects/cosmographic_analysis/`)
- Pinned deps in `environment.yml` (auto-exported from conda)

## Running Code

- Max **8 workers** for multiprocessing pools. Use `min(8, os.cpu_count() or 8)`.
- Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before any numpy/scipy import. Either at the top of every script, or ensured by importing the package first (src/cosmographic_analysis/__init__.py sets these, but only works if imported before numpy).
- Keep terminal output clean — no runaway tqdm progress bars, clear_output() for iterative loops.
- When running benchmarks: use `time.perf_counter()` for precision. Keep batch sizes small (start with N=4-8 before scaling up).

## Branch Structure

| Branch | Status | Contents |
|---|---|---|
| `main` | Clean | Original code, all files at root, notebooks import from local .py files |
| `infrastructure-overhaul` | Pushed | Package restructure (src/ layout), CLI scripts, config.yaml, BLAS/pool/precompute perf fixes |
| `perf/numba-optimization` | Pushed | Numba evaluation (proven ineffective — preserved as reference) |

## Architecture

```
src/cosmographic_analysis/     # Installable Python package
├── cosmology.py               # Distance modulus formulas
├── coordinates.py             # HEALPix coordinate transforms
├── hemispheric_comparison.py  # Core analysis (hem_h0, hem_q0, exec_map, precompute)
├── dw_statistic.py            # Durbin-Watson statistic
├── statistics.py              # Gaussian fitting, MC p-values
├── maps.py                    # Sky map generation, loading
├── anisotropy.py              # Max anisotropy direction finder
├── data_loader.py             # Pantheon+ data loading
├── config.py                  # YAML config loader (dataclasses)
└── plotting/
    ├── histograms.py          # Histogram plots
    ├── skymaps.py             # Mollweide sky map plots
    └── summary.py             # Summary tables (CSV + PNG)

scripts/                        # CLI entry points
├── run_pipeline.py             # Full analysis pipeline
├── run_synthetic_iso.py        # Isotropic MC simulations
└── run_synthetic_lcdm.py       # LCDM MC simulations

notebooks/                      # Jupyter notebooks (import from package)
├── main.ipynb
├── synthetic_data.ipynb
├── luminose_distance.ipynb
└── luminous_distanceLCDM.ipynb
```

## Pipeline Flow

1. Load Pantheon+ data → filter by redshift → build cov matrix
2. HEALPix grid → hemispheric comparison (exec_map, Pool parallel)
3. Real data: h0/q0 maps, max anisotropy, Durbin-Watson
4. ISO simulations: 500 iterations with reshuffled SN positions
5. LCDM simulations: 500 iterations with resampled distance moduli (uses precomputed hemisphere data)
6. Fit Gaussians → plot histograms → p-values → summary tables

## Output Conventions

| Directory | Contents | Naming pattern |
|---|---|---|
| `compilations/` | Numerical results (txt) | `[PREFIX][TYPE](params).txt` |
| `figures/` | Sky maps (PNG) | `[PREFIX][TYPE](params).png` |
| `histograms/` | Histogram plots (PNG) | `[TYPE](params).png` |
| `tables/` | Summary tables (CSV + PNG) | `[PREFIX][TYPE](params).csv/.png` |

Parameters in filenames use `hf={h0f}_qf={q0f}` format. PREFIX typically `[SH0ES_CALIB]`.

## Physics Context

- **Data:** Pantheon+ supernova catalog (1701 light curves, z<2.3). Main analysis cuts to 0.01<z<0.1 (~630 SNe).
- **Method:** Cosmographic (model-independent) distance expansion: `dl = c/h0 * (y + (3-q0)*y²/2)` where `y=z/(1+z)`.
- **Test:** Hemispheric Comparison — for each HEALPix direction, split the sky into two hemispheres, fit h0 (with q0 fixed) and q0 (with h0 fixed). Max Δ between opposite hemispheres measures anisotropy.
- **Simulations:** ISO = isotropic SN positions (null test), LCDM = ΛCDM-consistent distance moduli. Compare observed anisotropy against MC distributions.

## Committing

- CRITICAL: Do NOT commit, push, or create PRs unless explicitly prompted.
- When committing IS requested, inspect `git status`, `git diff`, and `git log --oneline -10` first.
- Stage only intended files. Never commit secrets, generated artifacts, or test output.
- When dispatching subagents, explicitly forbid commit/push steps in their instructions.

## Remote Execution (Lab Machine)

- Lab SSH: `ssh uni`
- Lab project path: `~/Documentos/cosmographic_analysis/`
- Sync via GitHub push/pull — never rsync the full project.

### Lab execution pattern (prevents SSH hangs)

1. Write script locally, verify with small test
2. Commit + push to GitHub
3. `ssh uni "cd ~/Documentos/cosmographic_analysis && git pull && conda run -n cosmographic_analysis nohup python scripts/run_pipeline.py > ~/pipeline.log 2>&1 & echo PID=$!"`
4. Track with SHORT timeouts (10-15s): `ssh uni "tail -5 ~/pipeline.log"`
5. Check completion: `ssh uni "ps aux | grep run_pipeline | grep -v grep | wc -l"`
6. Do NOT use `sleep N && ssh ...` — blocks indefinitely. Poll with short timeouts instead.
7. Results are in `compilations/`, `figures/`, `histograms/`, `tables/` on the lab. Pull via GitHub.

### Design patterns for scripts run remotely

**Opt-in parallel** — `n_workers` defaults to 1. Serial branch is original code verbatim. Import `concurrent.futures` inside the `if` so serial mode needs nothing extra.

```python
def run_sweep(..., n_workers=1):
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        tasks = [(arg1, arg2) for ...]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for result in executor.map(_worker_fn, tasks):
                store(result)
    else:
        for arg1, arg2 in ...:
            result = compute(arg1, arg2)
            store(result)
```

Worker function must be at module level (pickle requirement). Never nest parallelism — pick the outer loop or the inner one.

**Flat progress** — one line per item, same format serial and parallel, no tqdm:

```python
n = len(items)
print(f"Processing {n} items (n_workers={n_workers})...")
for i, item in enumerate(items, 1):
    result = compute(item)
    print(f"  [{i}/{n}] {label(item)} -> {summarize(result)}")
```

For sub-items within one job: print header with total, inline progress every 10% using `\r`.

**Errors are data** — never stop the whole run:

```python
def worker_fn(a, b):
    try:
        result = compute(a, b)
        if result['status'] != 'ok':
            return {'a': a, 'b': b, 'status': 'error', 'message': result.get('message', '')}
        return {'a': a, 'b': b, 'status': 'ok', 'value': result['value']}
    except Exception as e:
        return {'a': a, 'b': b, 'status': 'error', 'message': str(e)}
```

Output JSON includes both successes and failures — filter at analysis time.

**Lab computes, local analyzes** — heavy runs on lab, results saved as JSON/CSV, analysis and plotting local:

```
Lab:   python scripts/sweep.py --n-workers 8  →  compilations/results.json
Local: rsync -avz lab:~/Documentos/cosmographic_analysis/compilations/ compilations/
       jupyter lab notebooks/analyse.ipynb
```

**CLI design for remote scripts** — every parameter is a CLI argument with default, `--n-workers` optional (default 1), output path configurable:

```bash
python scripts/run_pipeline.py --config config.yaml --n-workers 8
```

**Inline post-processing** — if each grid point produces intermediate data that gets reduced, compute the reduction in the worker rather than writing and re-reading files:

```python
# Avoid:
result = run_pipeline(save=True)
analyze_from_file(path)

# Prefer:
result = run_pipeline(save=False)
metric = compute_metric(result["data"])
```

## .md Files Are Public

This repository is public. Do not write into .md files:
- API keys, tokens, credentials
- User-specific internal paths (usernames, home directories) unless already public
- Personal or sensitive data

## Key Constraints (Do Not Override)

- Do NOT use more than 8 multiprocessing workers
- OpenBLAS threads must be set to 1 before numpy imports
- No `sys.path` hacks — use `pip install -e .`
- Pipeline config from `config.yaml`, not hardcoded in scripts
