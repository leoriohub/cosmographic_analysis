#!/usr/bin/env python3
"""
Benchmark the cosmographic analysis pipeline to identify slow code sections.

Measures wall-clock per pipeline section, isolated kernel micro-benchmarks,
cProfile traces, and scaling characteristics. Produces a ranked report of
optimization targets.

Usage:
    python scripts/benchmark_pipeline.py --mode quick          # Fast dev iteration
    python scripts/benchmark_pipeline.py --mode full           # Production accuracy
    python scripts/benchmark_pipeline.py --mode scaling        # Scaling matrix
    python scripts/benchmark_pipeline.py --mode profile        # cProfile trace
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import cProfile
import json
import pstats
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool
from typing import Callable, Optional

import numpy as np
import healpy as hp

from cosmographic_analysis.config import load_config, ensure_output_dirs
from cosmographic_analysis.data_loader import load_pantheon_data, build_datos_tuple
from cosmographic_analysis.coordinates import get_healpix_vectors, IndexToDecRa
from cosmographic_analysis.cosmology import mu_model, MODEL_CODES
from cosmographic_analysis.hemispheric_comparison import (
    exec_map_numba,
    exec_map_fixed_numba,
    precompute_hemisphere_data,
    _golden_fit,
    _chi2_1par,
)


# ---------------------------------------------------------------------------
# Timer utilities
# ---------------------------------------------------------------------------


@dataclass
class TimerStats:
    """Accumulates timing statistics across multiple runs."""
    name: str
    times: list = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(self.times)

    @property
    def mean(self) -> float:
        return np.mean(self.times) if self.times else 0.0

    @property
    def std(self) -> float:
        return np.std(self.times) if len(self.times) > 1 else 0.0

    @property
    def count(self) -> int:
        return len(self.times)

    def add(self, seconds: float):
        self.times.append(seconds)


_timers: list[TimerStats] = []
_current_timer: Optional[TimerStats] = None


@contextmanager
def measure(name: str):
    """Context manager that records elapsed time into a TimerStats accumulator."""
    global _current_timer, _timers
    ts = TimerStats(name)
    _timers.append(ts)
    saved = _current_timer
    _current_timer = ts
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        ts.add(elapsed)
        _current_timer = saved


def reset_timers():
    """Clear all accumulated timing data."""
    global _timers
    _timers = []


def get_timers() -> list[TimerStats]:
    """Return all accumulated TimerStats in insertion order."""
    return list(_timers)


def print_timer_table(timers: list[TimerStats], title: str = "Timing Results"):
    """Print a formatted table of TimerStats."""
    if not timers:
        print(f"  ({title}: no data)")
        return

    total_wall = sum(t.total for t in timers)
    if total_wall == 0:
        total_wall = 1e-12

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    header = f"{'Section':<40s} {'Calls':>6s} {'Total (s)':>10s} {'Mean (s)':>10s} {'Std (s)':>9s} {'%':>6s}"
    print(header)
    print("-" * len(header))

    for t in timers:
        pct = t.total / total_wall * 100
        print(f"{t.name:<40s} {t.count:>6d} {t.total:>10.4f} {t.mean:>10.6f} {t.std:>9.6f} {pct:>5.1f}%")

    print("-" * len(header))
    print(f"{'TOTAL':<40s} {'':>6s} {total_wall:>10.4f} {'':>10s} {'':>9s} {'100.0%':>6s}")
    print()

    return total_wall


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    mode: str  # quick | full | scaling | profile
    nside: int
    repetitions: int
    n_workers: int
    optimizer: str  # golden | scipy
    output_path: str
    config: str  # path to config.yaml


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Benchmark the cosmographic analysis pipeline."
    )
    parser.add_argument(
        "--mode", choices=["quick", "full", "scaling", "profile"],
        default="quick", help="Benchmark mode (default: quick)"
    )
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of parallel workers (default: 2 quick, 8 full)"
    )
    parser.add_argument(
        "--nside", type=int, default=None,
        help="HEALPix resolution (default: 4 quick, 8 full)"
    )
    parser.add_argument(
        "--repetitions", type=int, default=None,
        help="MC repetitions (default: 5 quick, 500 full)"
    )
    parser.add_argument(
        "--optimizer", choices=["golden", "scipy", "grid"], default="golden",
        help="Optimizer method (default: golden)"
    )
    parser.add_argument(
        "--output", type=str, default="compilations/benchmark_results.json",
        help="Path for JSON output (default: compilations/benchmark_results.json)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: config.yaml in project root)"
    )

    args = parser.parse_args()

    if args.mode == "quick":
        nside = args.nside or 4
        reps = args.repetitions or 5
        workers = args.n_workers or 2
    elif args.mode == "full":
        nside = args.nside or 8
        reps = args.repetitions or 500
        workers = args.n_workers or min(8, os.cpu_count() or 8)
    elif args.mode == "scaling":
        nside = args.nside or 8
        reps = args.repetitions or 5
        workers = args.n_workers or 16
    else:  # profile
        nside = args.nside or 8
        reps = 1  # profile just one iteration
        workers = args.n_workers or 1

    return BenchmarkConfig(
        mode=args.mode,
        nside=nside,
        repetitions=reps,
        n_workers=workers,
        optimizer=args.optimizer,
        output_path=args.output,
        config=args.config or "config.yaml",
    )


# ---------------------------------------------------------------------------
# Numba warmup
# ---------------------------------------------------------------------------


def warmup_numba():
    """Run 3 warmup iterations through a golden fit to trigger Numba JIT compilation."""
    print("  Warming up Numba JIT (3 iterations)...")
    rng = np.random.default_rng(42)
    n = 100
    z = np.sort(rng.uniform(0.01, 0.1, n)).astype(np.float64)
    mu_ceph = rng.uniform(34, 38, n).astype(np.float64)
    mu_sh0es = rng.uniform(34, 38, n).astype(np.float64)
    hostyn = rng.integers(0, 2, n).astype(np.int64)
    # Build a small SPD matrix for the inverse
    A = rng.normal(0, 1, (n, n)).astype(np.float64)
    cov_small = A.T @ A + np.eye(n)
    inv_cov = np.linalg.inv(cov_small).astype(np.float64)

    for i in range(3):
        _golden_fit(z, mu_ceph, mu_sh0es, hostyn, inv_cov, -0.574, True, 0.3, 1.5)
    print("  Numba warmup complete.")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase A: Overall pipeline wall-clock timing
# ---------------------------------------------------------------------------


def phase_a_pipeline_timing(bcfg: BenchmarkConfig, config_path: str):
    """Measure wall-clock time per major pipeline section."""
    print("\n" + "=" * 70)
    print("  PHASE A: Overall Pipeline Wall-Clock Timing")
    print(f"  Config: nside={bcfg.nside}, reps={bcfg.repetitions}, "
          f"workers={bcfg.n_workers}, optimizer={bcfg.optimizer}")
    print("=" * 70)

    reset_timers()

    # A1: Data loading
    cfg = load_config(config_path)
    zup = cfg.parameters.zup
    zdown = cfg.parameters.zdown
    model_code = MODEL_CODES[cfg.parameters.model]

    print("\n  Loading data...")
    with measure("Data loading"):
        zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, inv_cov_z, cov_numpy = \
            load_pantheon_data(cfg.data.lcparam, cfg.data.cov_matrix, zup, zdown)

    pts = hp.nside2npix(bcfg.nside)
    print(f"  {len(zz)} SNe loaded, cov matrix: {cov_mat.shape}")

    datos = build_datos_tuple(
        ra, dec, zz, mz, sigmz, muz, sigmuz, muceph, hostyn,
        cov_mat, cfg.parameters.h0f, cfg.parameters.q0f, pts, zup, zdown, cov_numpy, model_code,
    )
    r1, v1, hostyn_arr, _, _, _, _, _, _, _, _ = datos

    # HEALPix vectors
    with measure("HEALPix initialization"):
        healpix_dirs = get_healpix_vectors(bcfg.nside)
    print(f"  {len(healpix_dirs)} HEALPix directions (nside={bcfg.nside})")

    # Warmup Numba
    warmup_numba()

    # A2: Real data hemispheric comparison
    print("\n  Running real data hemispheric comparison...")
    with measure("Real data (hemispheric comparison)"):
        if bcfg.n_workers > 1:
            pool = Pool(min(bcfg.n_workers, 8, os.cpu_count() or 8))
            results_h0, results_q0 = exec_map_numba(
                healpix_dirs, datos, pool=pool, n_workers=bcfg.n_workers,
                method=bcfg.optimizer,
            )
            pool.close()
            pool.join()
        else:
            results_h0, results_q0 = exec_map_numba(
                healpix_dirs, datos, n_workers=1, method=bcfg.optimizer,
            )

    # A3: LCDM precompute
    print("\n  Precomputing LCDM hemisphere data...")
    with measure("LCDM precompute"):
        lcdm_precomputed = precompute_hemisphere_data(
            healpix_dirs, datos, n_workers=bcfg.n_workers,
        )

    # A4: ISO simulation
    print(f"\n  Running ISO simulation ({bcfg.repetitions} reps)...")
    v1_iso = np.zeros((bcfg.repetitions, len(ra), 3))
    for i in range(bcfg.repetitions):
        vecti = np.random.randn(len(ra), 3)
        vecti /= np.linalg.norm(vecti, axis=1)[:, np.newaxis]
        v1_iso[i] = vecti

    with measure(f"ISO simulation ({bcfg.repetitions} rep)"):
        iso_deltas = []
        for i in range(bcfg.repetitions):
            datos_iso = [r1, v1_iso[i], hostyn_arr, cov_mat,
                         cfg.parameters.h0f, cfg.parameters.q0f,
                         pts, zup, zdown, cov_numpy, model_code]
            res_h0, res_q0 = exec_map_numba(
                healpix_dirs, tuple(datos_iso), n_workers=1, method=bcfg.optimizer,
            )
            h0u = np.array(res_h0[0])
            h0d = np.array(res_h0[1])
            q0u = np.array(res_q0[0])
            q0d = np.array(res_q0[1])
            h0m = np.concatenate((h0u, h0d))
            q0m = np.concatenate((q0u, q0d))
            iso_deltas.append((np.max(h0m) - np.min(h0m),
                               np.max(q0m) - np.min(q0m)))
    iso_deltas = np.array(iso_deltas)

    # A5: LCDM simulation
    print(f"\n  Running LCDM simulation ({bcfg.repetitions} reps)...")
    mu_fid = np.array([mu_model(zi, cfg.parameters.h0f, cfg.parameters.q0f, model_code) for zi in zz])
    r1_lcdm = np.tile(r1, (bcfg.repetitions, 1, 1))
    sigmuz_arr = np.array(sigmuz) if isinstance(sigmuz, list) else sigmuz
    for i in range(bcfg.repetitions):
        mu_sample = np.random.normal(mu_fid, sigmuz_arr)
        r1_lcdm[i, :, 5] = mu_sample

    with measure(f"LCDM simulation ({bcfg.repetitions} rep)"):
        lcdm_deltas = []
        for i in range(bcfg.repetitions):
            datos_lcdm = [r1_lcdm[i], v1, hostyn_arr, cov_mat,
                          cfg.parameters.h0f, cfg.parameters.q0f,
                          pts, zup, zdown, cov_numpy, model_code]
            res_h0, res_q0 = exec_map_fixed_numba(
                healpix_dirs, tuple(datos_lcdm), lcdm_precomputed,
                n_workers=1, method=bcfg.optimizer,
            )
            h0u = np.array(res_h0[0])
            h0d = np.array(res_h0[1])
            q0u = np.array(res_q0[0])
            q0d = np.array(res_q0[1])
            h0m = np.concatenate((h0u, h0d))
            q0m = np.concatenate((q0u, q0d))
            lcdm_deltas.append((np.max(h0m) - np.min(h0m),
                                np.max(q0m) - np.min(q0m)))
    lcdm_deltas = np.array(lcdm_deltas)

    # A6: Plotting + I/O (light version — just file writes)
    print("\n  Writing results...")
    with measure("I/O (saving results)"):
        header_iso = f"ISO benchmark results\n"
        np.savetxt("/tmp/bench_iso.txt",
                   np.column_stack([iso_deltas[:, 0], iso_deltas[:, 1]]),
                   header=header_iso)
        np.savetxt("/tmp/bench_lcdm.txt",
                   np.column_stack([lcdm_deltas[:, 0], lcdm_deltas[:, 1]]),
                   header=header_iso)

    # Print results
    timers = get_timers()
    total = print_timer_table(timers, "Phase A: Pipeline Wall-Clock")

    # Data for later aggregation
    phase_a_data = {
        "sections": [
            {"name": t.name, "total_s": t.total, "mean_s": t.mean,
             "std_s": t.std, "calls": t.count}
            for t in timers
        ],
        "total_s": total,
        "n_directions": len(healpix_dirs),
        "n_sne": len(zz),
        "config": {
            "nside": bcfg.nside,
            "repetitions": bcfg.repetitions,
            "n_workers": bcfg.n_workers,
            "optimizer": bcfg.optimizer,
        },
    }

    return phase_a_data


# ---------------------------------------------------------------------------
# Phase B: Micro-benchmarks
# ---------------------------------------------------------------------------


def bench_inversion(sizes: list[int], n_repeats: int = 100):
    """Benchmark np.linalg.inv at various matrix sizes."""
    print(f"\n  -- Inversion micro-benchmark ({n_repeats} reps per size) --")
    results = []
    for n in sizes:
        rng = np.random.default_rng(42 + n)
        A = rng.normal(0, 1, (n, n))
        A = A.T @ A + np.eye(n)  # SPD

        # Warmup
        _ = np.linalg.inv(A)

        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            inv = np.linalg.inv(A)
            times.append(time.perf_counter() - t0)
            _ = inv[0, 0]  # prevent optimization

        mean_t = np.mean(times)
        std_t = np.std(times)
        # FLOP count for inv: ~2*n^3/3 (LU) + ~4*n^3/3 (inv) = 2*n^3
        flops = 2 * n ** 3
        gflops = flops / mean_t / 1e9 if mean_t > 0 else 0

        print(f"    inv({n:3d}×{n:3d}): {mean_t*1000:8.3f} ms ± {std_t*1000:6.3f} ms  "
              f"({gflops:5.1f} GFLOP/s)")
        results.append({"size": n, "mean_ms": mean_t * 1000, "std_ms": std_t * 1000,
                        "gflops": gflops})

    return results


def bench_chi2(sizes: list[int], n_repeats: int = 1000):
    """Benchmark the chi² evaluation: dot(resid, dot(inv_cov, resid))."""
    print(f"\n  -- Chi² evaluation micro-benchmark ({n_repeats} reps per size) --")
    results = []
    for n in sizes:
        rng = np.random.default_rng(42 + n)
        resid = rng.normal(0, 1, n).astype(np.float64)
        A = rng.normal(0, 1, (n, n)).astype(np.float64)
        inv_cov = np.linalg.inv(A.T @ A + np.eye(n)).astype(np.float64)

        # Warmup
        _ = np.dot(resid, np.dot(inv_cov, resid))

        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            val = np.dot(resid, np.dot(inv_cov, resid))
            times.append(time.perf_counter() - t0)
            _ = val  # prevent optimization

        mean_t = np.mean(times)
        std_t = np.std(times)
        # FLOP: 2*n^2 + n ≈ 2*n^2
        flops = 2 * n ** 2
        gflops = flops / mean_t / 1e9 if mean_t > 0 else 0

        print(f"    chi2({n:3d}): {mean_t * 1e6:8.1f} μs ± {std_t * 1e6:6.1f} μs  "
              f"({gflops:5.2f} GFLOP/s)")
        results.append({"size": n, "mean_us": mean_t * 1e6, "std_us": std_t * 1e6,
                        "gflops": gflops})

    return results


def bench_golden_section(n_repeats: int = 50):
    """Benchmark the golden section search on synthetic data."""
    print(f"\n  -- Golden section micro-benchmark ({n_repeats} reps) --")
    rng = np.random.default_rng(42)
    n = 315  # match typical hemisphere size

    z = np.sort(rng.uniform(0.01, 0.1, n)).astype(np.float64)
    mu_ceph = rng.uniform(34, 38, n).astype(np.float64)
    mu_sh0es = rng.uniform(34, 38, n).astype(np.float64)
    hostyn = rng.integers(0, 2, n).astype(np.int64)
    A = rng.normal(0, 1, (n, n)).astype(np.float64)
    inv_cov = np.linalg.inv(A.T @ A + np.eye(n)).astype(np.float64)

    # Warmup (3 iters) — already done globally, but do local warmup too
    for _ in range(3):
        _golden_fit(z, mu_ceph, mu_sh0es, hostyn, inv_cov, -0.574, True, 0.3, 1.5)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        val, err = _golden_fit(z, mu_ceph, mu_sh0es, hostyn, inv_cov,
                                -0.574, True, 0.3, 1.5)
        times.append(time.perf_counter() - t0)
        _ = val, err

    mean_t = np.mean(times)
    std_t = np.std(times)
    print(f"    golden section: {mean_t * 1000:.3f} ms ± {std_t * 1000:.3f} ms  "
          f"(over {n_repeats} reps, n={n})")

    return {"mean_ms": mean_t * 1000, "std_ms": std_t * 1000, "n": n,
            "n_repeats": n_repeats}


def bench_pandas_iloc(n_repeats: int = 384):
    """Benchmark cov_mat.iloc[upi, upi].values overhead."""
    print(f"\n  -- Pandas iloc micro-benchmark ({n_repeats} reps) --")
    import pandas as pd
    rng = np.random.default_rng(42)
    n_total = 630
    n_sub = 315

    data = rng.normal(0, 1, (n_total, n_total)).astype(np.float64)
    data = data.T @ data + np.eye(n_total)
    df = pd.DataFrame(data)

    times = []
    for _ in range(n_repeats):
        upi = np.sort(rng.choice(n_total, n_sub, replace=False))
        t0 = time.perf_counter()
        sub = df.iloc[upi, upi].values
        times.append(time.perf_counter() - t0)
        _ = sub[0, 0]

    mean_t = np.mean(times)
    std_t = np.std(times)
    print(f"    iloc({n_sub}×{n_sub}): {mean_t * 1000:.3f} ms ± {std_t * 1000:.3f} ms")
    return {"mean_ms": mean_t * 1000, "std_ms": std_t * 1000, "n_sub": n_sub,
            "n_repeats": n_repeats}


def _mp_dummy_worker(x):
    """Dummy worker for multiprocessing overhead benchmark."""
    return x * 2


def bench_mp_overhead(n_tasks: int = 384, worker_counts: list[int] = None):
    """Benchmark multiprocessing Pool dispatch overhead."""
    if worker_counts is None:
        worker_counts = [1, 2, 4, 8]
    print(f"\n  -- Multiprocessing overhead benchmark ({n_tasks} tasks) --")

    # Serial baseline
    t0 = time.perf_counter()
    for i in range(n_tasks):
        _ = _mp_dummy_worker(i)
    serial_time = time.perf_counter() - t0
    print(f"    serial loop ({n_tasks} tasks): {serial_time * 1000:.3f} ms")

    results = {"serial_ms": serial_time * 1000, "n_tasks": n_tasks, "workers": []}

    for nw in worker_counts:
        if nw == 1:
            # starmap with 1 worker has overhead too
            with Pool(1) as pool:
                t0 = time.perf_counter()
                _ = list(pool.starmap(_mp_dummy_worker, [(i,) for i in range(n_tasks)]))
                t = time.perf_counter() - t0
        else:
            with Pool(min(nw, n_tasks)) as pool:
                t0 = time.perf_counter()
                _ = list(pool.starmap(_mp_dummy_worker, [(i,) for i in range(n_tasks)]))
                t = time.perf_counter() - t0
        # Overhead relative to ideal linear speedup
        ideal = serial_time / nw if nw > 0 else serial_time
        overhead = t - ideal if t > ideal else 0.0
        slowdown = t / ideal if ideal > 0 else 0.0
        print(f"    Pool({nw} workers) starmap: {t * 1000:.3f} ms "
              f"(ideal: {ideal * 1000:.3f} ms, slowdown: {slowdown:.2f}x)")
        results["workers"].append({"n_workers": nw, "total_ms": t * 1000,
                                   "overhead_ms": overhead * 1000})

    return results


def phase_b_microbenchmarks():
    """Run all Phase B micro-benchmarks."""
    print("\n" + "=" * 70)
    print("  PHASE B: Kernel Micro-Benchmarks")
    print("=" * 70)

    sizes = [100, 200, 315, 400, 500]
    inv_results = bench_inversion(sizes, n_repeats=100)
    chi2_results = bench_chi2(sizes, n_repeats=1000)
    golden_results = bench_golden_section(n_repeats=50)
    iloc_results = bench_pandas_iloc(n_repeats=384)
    mp_results = bench_mp_overhead(n_tasks=384)

    return {
        "inversion": inv_results,
        "chi2": chi2_results,
        "golden_section": golden_results,
        "pandas_iloc": iloc_results,
        "mp_overhead": mp_results,
    }


# ---------------------------------------------------------------------------
# Phase C: cProfile trace
# ---------------------------------------------------------------------------


def phase_c_profile(bcfg: BenchmarkConfig):
    """Run cProfile on a single ISO MC iteration."""
    print("\n" + "=" * 70)
    print("  PHASE C: cProfile Trace (Single ISO Iteration)")
    print("=" * 70)

    cfg = load_config(bcfg.config)
    zup = cfg.parameters.zup
    zdown = cfg.parameters.zdown
    model_code = MODEL_CODES[cfg.parameters.model]

    zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, inv_cov_z, cov_numpy = \
        load_pantheon_data(cfg.data.lcparam, cfg.data.cov_matrix, zup, zdown)
    pts = hp.nside2npix(bcfg.nside)
    datos = build_datos_tuple(
        ra, dec, zz, mz, sigmz, muz, sigmuz, muceph, hostyn,
        cov_mat, cfg.parameters.h0f, cfg.parameters.q0f, pts, zup, zdown, cov_numpy, model_code,
    )
    r1, v1, hostyn_arr, _, _, _, _, _, _, _, _ = datos
    healpix_dirs = get_healpix_vectors(bcfg.nside)

    warmup_numba()

    # Create one ISO iteration
    vecti = np.random.randn(len(ra), 3)
    vecti /= np.linalg.norm(vecti, axis=1)[:, np.newaxis]
    datos_iso = [r1, vecti, hostyn_arr, cov_mat,
                 cfg.parameters.h0f, cfg.parameters.q0f,
                 pts, zup, zdown, cov_numpy, model_code]

    prof_file = ".omo/evidence/benchmark-pipeline/phase_c_profile.prof"
    print(f"  Profiling single ISO iteration...")

    profiler = cProfile.Profile()
    profiler.enable()
    res_h0, res_q0 = exec_map_numba(
        healpix_dirs, tuple(datos_iso), n_workers=1, method=bcfg.optimizer,
    )
    profiler.disable()
    profiler.dump_stats(prof_file)

    # Print top 30 functions
    print("\n  Top 30 functions by cumulative time:")
    print("  " + "-" * 60)
    stats = pstats.Stats(prof_file)
    stats.sort_stats("cumtime").print_stats(30)

    # Also top 10 by call count
    print("\n  Top 10 functions by call count:")
    print("  " + "-" * 60)
    stats.sort_stats("ncalls").print_stats(10)

    # Analyze the profile data programmatically
    from pstats import Stats
    import io
    s = io.StringIO()
    ps = Stats(prof_file, stream=s)
    ps.sort_stats("cumtime").print_stats(30)
    profile_text = s.getvalue()

    with open(".omo/evidence/benchmark-pipeline/phase_c_profile.txt", "w") as f:
        f.write(profile_text)

    print(f"  Profile saved to {prof_file}")

    return {"profile_file": prof_file, "n_directions": len(healpix_dirs)}


# ---------------------------------------------------------------------------
# Phase D: Scaling characterization
# ---------------------------------------------------------------------------


def phase_d_scaling(bcfg: BenchmarkConfig):
    """Measure pipeline runtime across nside × n_worker combinations."""
    print("\n" + "=" * 70)
    print("  PHASE D: Scaling Characterization (nside × n_workers)")
    print("=" * 70)

    nsides = [4, 6, 8]
    worker_counts = [1, 2, 4, 8, 16]

    # Direction counts (accounting for the npix/2 in get_healpix_vectors)
    nside_dirs = {4: 96, 6: 216, 8: 384}
    print(f"  Direction counts: {nside_dirs} (note: halved from full npix)")
    print(f"  Using {5} reps per measurement\n")

    cfg = load_config(bcfg.config)
    zup = cfg.parameters.zup
    zdown = cfg.parameters.zdown
    model_code = MODEL_CODES[cfg.parameters.model]

    # Load data once
    zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, inv_cov_z, cov_numpy = \
        load_pantheon_data(cfg.data.lcparam, cfg.data.cov_matrix, zup, zdown)

    warmup_numba()

    scaling_data = {"iso_per_iter": [], "lcdm_per_iter": [], "real_data": []}

    for nside in nsides:
        n_dirs = nside_dirs[nside]
        pts = hp.nside2npix(nside)
        datos = build_datos_tuple(
            ra, dec, zz, mz, sigmz, muz, sigmuz, muceph, hostyn,
            cov_mat, cfg.parameters.h0f, cfg.parameters.q0f, pts, zup, zdown, cov_numpy, model_code,
        )
        r1, v1, hostyn_arr, _, _, _, _, _, _, _, _ = datos
        healpix_dirs = get_healpix_vectors(nside)

        print(f"\n  --- nside={nside} ({n_dirs} directions) ---")

        for nw in worker_counts:
            if nw > os.cpu_count() or nw > 16:
                continue

            actual_workers = min(nw, n_dirs)

            # Real data timing
            pool = Pool(actual_workers) if actual_workers > 1 else None
            t0 = time.perf_counter()
            exec_map_numba(healpix_dirs, datos, pool=pool,
                           n_workers=actual_workers, method="golden")
            real_t = time.perf_counter() - t0
            if pool:
                pool.close()
                pool.join()

            # ISO per-iteration timing (5 reps, no pool — use n_workers=1 per iter)
            # For ISO timing, we measure per-iter with n_workers=1 inside each iter
            # but we can also measure external parallelism
            iso_times = []
            for _ in range(3):  # 3 reps enough for timing estimate
                vecti = np.random.randn(len(ra), 3)
                vecti /= np.linalg.norm(vecti, axis=1)[:, np.newaxis]
                datos_iso = [r1, vecti, hostyn_arr, cov_mat,
                             cfg.parameters.h0f, cfg.parameters.q0f,
                             pts, zup, zdown, cov_numpy, model_code]
                t0 = time.perf_counter()
                exec_map_numba(healpix_dirs, tuple(datos_iso),
                               n_workers=1, method="golden")
                iso_times.append(time.perf_counter() - t0)
            iso_mean = np.mean(iso_times)

            # LCDM per-iteration timing (with precompute, then measure per-iter)
            lcdm_pre = precompute_hemisphere_data(
                healpix_dirs, datos, n_workers=actual_workers, cov_numpy=cov_numpy,
            )
            mu_fid = np.array([mu_model(zi, cfg.parameters.h0f, cfg.parameters.q0f, model_code)
                              for zi in zz])
            lcdm_times = []
            for _ in range(3):
                mu_sample = np.random.normal(mu_fid, sigmuz)
                r1_it = r1.copy()
                r1_it[:, 5] = mu_sample
                datos_lcdm = [r1_it, v1, hostyn_arr, cov_mat,
                              cfg.parameters.h0f, cfg.parameters.q0f,
                              pts, zup, zdown, cov_numpy, model_code]
                t0 = time.perf_counter()
                exec_map_fixed_numba(healpix_dirs, tuple(datos_lcdm),
                                     lcdm_pre, n_workers=1, method="golden")
                lcdm_times.append(time.perf_counter() - t0)
            lcdm_mean = np.mean(lcdm_times)

            print(f"    workers={nw:2d} | real={real_t*1000:7.1f} ms | "
                  f"ISO/iter={iso_mean*1000:7.1f} ms | "
                  f"LCDM/iter={lcdm_mean*1000:7.1f} ms")

            scaling_data["real_data"].append({
                "nside": nside, "n_workers": nw, "time_ms": real_t * 1000,
            })
            scaling_data["iso_per_iter"].append({
                "nside": nside, "n_workers": nw, "time_ms": iso_mean * 1000,
            })
            scaling_data["lcdm_per_iter"].append({
                "nside": nside, "n_workers": nw, "time_ms": lcdm_mean * 1000,
            })

    # Print summary tables
    print("\n\n  Scaling Summary — ISO per-iteration time (ms):")
    print(f"  {'workers':>8s}", end="")
    for ns in nsides:
        print(f"  nside={ns:<6d}", end="")
    print()
    for nw in worker_counts:
        print(f"  {nw:>8d}", end="")
        for ns in nsides:
            vals = [d["time_ms"] for d in scaling_data["iso_per_iter"]
                    if d["nside"] == ns and d["n_workers"] == nw]
            if vals:
                print(f"  {vals[0]:>10.1f}", end="")
            else:
                print(f"  {'N/A':>10s}", end="")
        print()

    print(f"\n  Scaling Summary — LCDM per-iteration time (ms):")
    print(f"  {'workers':>8s}", end="")
    for ns in nsides:
        print(f"  nside={ns:<6d}", end="")
    print()
    for nw in worker_counts:
        print(f"  {nw:>8d}", end="")
        for ns in nsides:
            vals = [d["time_ms"] for d in scaling_data["lcdm_per_iter"]
                    if d["nside"] == ns and d["n_workers"] == nw]
            if vals:
                print(f"  {vals[0]:>10.1f}", end="")
            else:
                print(f"  {'N/A':>10s}", end="")
        print()

    # Save scaling JSON
    scaling_json_path = ".omo/evidence/benchmark-pipeline/phase_d_scaling.json"
    with open(scaling_json_path, "w") as f:
        json.dump(scaling_data, f, indent=2)
    print(f"\n  Scaling data saved to {scaling_json_path}")

    return scaling_data


# ---------------------------------------------------------------------------
# Phase E: Report generation
# ---------------------------------------------------------------------------


def generate_report(phase_a, phase_b, phase_c, phase_d, bcfg: BenchmarkConfig):
    """Consolidate all measurements into a ranked report."""
    print("\n" + "=" * 70)
    print("  PHASE E: Consolidated Benchmark Report")
    print("=" * 70)

    # Build ranked list of operations
    ranked = []

    # From Phase A
    if phase_a and "sections" in phase_a:
        for sec in phase_a["sections"]:
            ranked.append({
                "operation": sec["name"],
                "total_s": sec["total_s"],
                "self_s": sec["total_s"],
                "calls": sec["calls"],
                "source": "Phase A (wall-clock)",
                "pct": sec["total_s"] / phase_a.get("total_s", 1) * 100
                if phase_a.get("total_s", 0) > 0 else 0,
            })

    # From Phase B (estimate total impact from micro-benchmarks)
    if phase_b:
        n_dirs = phase_a.get("n_directions", 384) if phase_a else 384
        n_reps = bcfg.repetitions

        # Estimate total inversion time in ISO
        if "inversion" in phase_b:
            for inv in phase_b["inversion"]:
                if inv["size"] == 315:
                    inv_time_s = inv["mean_ms"] / 1000
                    total_inv_iso = inv_time_s * n_dirs * n_reps * 2  # 2 hemi
                    ranked.append({
                        "operation": f"np.linalg.inv ({inv['size']}×{inv['size']}) in ISO",
                        "total_s": total_inv_iso,
                        "self_s": total_inv_iso,
                        "calls": n_dirs * n_reps * 2,
                        "source": "Phase B (extrapolated)",
                        "pct": 0,  # will be normalized
                    })

        # Estimate total chi² time
        if "chi2" in phase_b:
            for chi2 in phase_b["chi2"]:
                if chi2["size"] == 315:
                    chi2_time_us = chi2["mean_us"]
                    # ~30 chi2 per golden fit, 4 fits per dir, 2 hemi, n_dirs, n_reps
                    n_chi2_calls = n_dirs * n_reps * 4 * 30
                    total_chi2_s = chi2_time_us / 1e6 * n_chi2_calls
                    ranked.append({
                        "operation": f"chi2 evaluation ({chi2['size']})",
                        "total_s": total_chi2_s,
                        "self_s": total_chi2_s,
                        "calls": n_chi2_calls,
                        "source": "Phase B (extrapolated)",
                        "pct": 0,
                    })

        # Golden section
        if "golden_section" in phase_b:
            gs = phase_b["golden_section"]
            gs_time_s = gs["mean_ms"] / 1000
            total_gs = gs_time_s * n_dirs * n_reps * 4  # 4 fits per dir
            ranked.append({
                "operation": "golden section fit (4 fits/dir)",
                "total_s": total_gs,
                "self_s": total_gs,
                "calls": n_dirs * n_reps * 4,
                "source": "Phase B (extrapolated)",
                "pct": 0,
            })

        # Pandas iloc
        if "pandas_iloc" in phase_b:
            il = phase_b["pandas_iloc"]
            il_time_s = il["mean_ms"] / 1000
            total_il = il_time_s * n_dirs * n_reps * 2  # 2 hemi
            ranked.append({
                "operation": f"pandas iloc indexing ({il['n_sub']}×{il['n_sub']})",
                "total_s": total_il,
                "self_s": total_il,
                "calls": n_dirs * n_reps * 2,
                "source": "Phase B (extrapolated)",
                "pct": 0,
            })

    # Normalize percentages
    total_all = sum(r["total_s"] for r in ranked) if ranked else 1
    for r in ranked:
        r["pct"] = r["total_s"] / total_all * 100

    # Sort descending
    ranked.sort(key=lambda r: r["total_s"], reverse=True)

    # Print ranked table
    print(f"\n  Ranked Optimization Targets")
    print(f"  (Total estimated: {total_all:.1f}s for {bcfg.repetitions} reps, "
          f"{n_dirs if phase_a else '?'} directions)")
    print(f"  " + "-" * 90)
    header = f"  {'RANK':>4s} {'Operation':<45s} {'Total (s)':>10s} {'%':>6s} {'Calls':>10s} {'Source'}"
    print(header)
    print(f"  " + "-" * 90)

    for i, r in enumerate(ranked, 1):
        marker = " ★" if i <= 3 else "  "
        print(f"  {f'{i}{marker}':>4s} {r['operation']:<45s} {r['total_s']:>10.1f} "
              f"{r['pct']:>5.1f}% {r['calls']:>10d} {r['source']}")

    # Identify top 3
    print(f"\n  ★ Top 3 Optimization Targets:")
    for i, r in enumerate(ranked[:3], 1):
        print(f"    {i}. {r['operation']} — {r['total_s']:.1f}s ({r['pct']:.1f}%) "
              f"— {r['source']}")

    # Cross-validation
    print(f"\n  Cross-Validation:")
    if phase_a and "total_s" in phase_a:
        wall_total = phase_a["total_s"]
        # Sum of section times from Phase A
        section_sum = sum(s["total_s"] for s in phase_a.get("sections", []))
        if wall_total > 0:
            disc = abs(section_sum - wall_total) / wall_total * 100
            print(f"    Phase A sections sum: {section_sum:.2f}s")
            print(f"    Phase A total wall-clock: {wall_total:.2f}s")
            print(f"    Discrepancy: {disc:.2f}% {'✓' if disc < 10 else '⚠ >10%'}")
        else:
            disc = 0
    else:
        disc = 0

    # Build report dict
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "mode": bcfg.mode,
            "nside": bcfg.nside,
            "repetitions": bcfg.repetitions,
            "n_workers": bcfg.n_workers,
            "optimizer": bcfg.optimizer,
        },
        "phases": {
            "wall_clock": phase_a,
            "micro_benchmarks": phase_b,
            "profile": phase_c,
            "scaling": phase_d,
        },
        "ranked_targets": ranked[:10],  # top 10
        "cross_validation": {
            "sum_of_sections": section_sum if phase_a else 0,
            "total_wallclock": wall_total if phase_a else 0,
            "discrepancy_pct": disc,
        },
    }

    # Save JSON
    json_path = bcfg.output_path
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {json_path}")

    return report


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------


def main():
    bcfg = parse_args()

    print("=" * 70)
    print("  COSMOGRAPHIC ANALYSIS — PIPELINE BENCHMARK")
    print(f"  Mode: {bcfg.mode}")
    print(f"  Config: nside={bcfg.nside}, reps={bcfg.repetitions}, "
          f"workers={bcfg.n_workers}, optimizer={bcfg.optimizer}")
    print("=" * 70)

    phase_a = None
    phase_b = None
    phase_c = None
    phase_d = None

    if bcfg.mode in ("quick", "full"):
        phase_a = phase_a_pipeline_timing(bcfg, bcfg.config)
        phase_b = phase_b_microbenchmarks()
        phase_c = phase_c_profile(bcfg)

    elif bcfg.mode == "scaling":
        phase_d = phase_d_scaling(bcfg)
        # Also run micro-benchmarks for context
        phase_b = phase_b_microbenchmarks()

    elif bcfg.mode == "profile":
        phase_c = phase_c_profile(bcfg)
        phase_b = phase_b_microbenchmarks()

    # Generate report
    report = generate_report(phase_a, phase_b, phase_c, phase_d, bcfg)

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
