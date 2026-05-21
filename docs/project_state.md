# Project State

## Branches

| Branch | Base | Status | Description |
|---|---|---|---|
| `main` | — | Clean | Original code, all files at root, notebooks import from local `.py` files |
| `infrastructure-overhaul` | `main` | Pushed | Package restructuring + performance optimizations |
| `perf/numba-optimization` | `infrastructure-overhaul` | Current | Numba JIT for inner optimization loop (in progress) |

### `infrastructure-overhaul` (pushed to remote)

All modules moved into `src/cosmographic_analysis/` installable package.
Notebooks moved to `notebooks/` with updated import paths.
CLI scripts in `scripts/` reproduce the full pipeline.
`config.yaml` centralizes all pipeline parameters.

**Performance optimizations applied:**
- OpenBLAS thread limit (parallel was 13× slower than sequential — fixed)
- `mu()` vectorized (100× speedup on that call)
- LCDM precompute: hemisphere masks + cov inversions computed once for 500 iterations
- Multiprocessing Pool reused across iterations (not created fresh each time)
- Workers capped at 10 to leave desktop room
- Pipeline estimate: ~50min (was ~66h baseline)

### `perf/numba-optimization` (current branch)

Adding Numba-compiled chi² and grid-search optimizer to replace
`scipy.optimize.minimize` in the hemispheric comparison inner loop.
Target: 3-5× speedup on the optimization step, ~50min → ~15-25min total.

## Environment

Conda environment: `cosmographic_analysis`

```bash
conda activate cosmographic_analysis
pip install -e .
python scripts/run_pipeline.py
```

Pinned dependencies in `environment.yml` (auto-exported).
All changes go through `pip install -e .` so the package is always importable.
