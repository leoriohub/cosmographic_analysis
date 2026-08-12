#!/usr/bin/env python3
"""
Truncation-error scan: parametrize the fit bias introduced by extending zup.

For each model x redshift window [zdown, zup] it reports the noiseless
LCDM fit biases (q0 with h0 at truth, h0 with q0 at truth) over the
actual Pantheon+ SNe in the window, the dL relative error at z=zup, and
the bias introduced by extending from the anchor zup (default 0.1).
Also prints the largest safe zup per model (|q0 bias| <= tolerance).

Usage: python scripts/truncation_scan.py [--config config.yaml]
       [--models taylor2,pade21] [--zups 0.1 0.2 0.3 0.4 0.5]
       [--tolerance 0.05] [--zup-max 0.6] [--output PATH.csv]
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cosmographic_analysis.config import load_config, ensure_output_dirs
from cosmographic_analysis.cosmology import MODEL_CODES
from cosmographic_analysis.truncation import (
    H0_TRUE,
    Q0_TRUE,
    max_safe_zup,
    scan_biases,
    window_redshifts,
)


def main():
    parser = argparse.ArgumentParser(description="Truncation-error scan")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo config.yaml)")
    parser.add_argument("--models", default="taylor2,pade21",
                        help="Comma-separated model names (default: taylor2,pade21)")
    parser.add_argument("--zups", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help="Redshift cuts to scan (default: 0.1 0.2 0.3 0.4 0.5)")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Safe-zup criterion: |q0 bias| <= tolerance (default: 0.05)")
    parser.add_argument("--zup-max", type=float, default=0.6,
                        help="Upper bound for the safe-zup scan (default: 0.6)")
    parser.add_argument("--output", default=None,
                        help="CSV output path (default: compilations/truncation_scan(...).csv)")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_output_dirs(config)
    p = config.parameters
    zdown = p.zdown

    models = [m.strip() for m in args.models.split(",")]
    unknown = set(models) - set(MODEL_CODES)
    if unknown:
        raise ValueError(f"Unknown model(s): {sorted(unknown)}. Valid: {sorted(MODEL_CODES)}")
    codes = [MODEL_CODES[m] for m in models]

    # Light load: only the redshift column (same column as data_loader).
    zz_all = np.loadtxt(config.data.lcparam, skiprows=1, usecols=(2,))
    print(f"Reference: flat LCDM (Om=0.30, H0=70), h0_true={H0_TRUE}, q0_true={Q0_TRUE}")
    print(f"Redshifts: {len(zz_all)} SNe from {config.data.lcparam}, windows zdown={zdown}")

    rows = scan_biases(zz_all, zdown, args.zups, models=codes)

    # ---- Printed table ----
    print("\n" + "=" * 100)
    print("Fit bias vs zup (noiseless LCDM; q0 fit with h0 at truth, h0 fit with q0 at truth)")
    print("=" * 100)
    print(f"{'model':8s} {'zup':>6s} {'nSNe':>6s} | {'q0_bias':>9s} {'Δq0 vs anchor':>13s} | "
          f"{'h0_bias':>9s} {'Δh0 vs anchor':>13s} | {'dL err(zup)':>11s}")
    print("-" * 100)
    for r in rows:
        dq = r.get("delta_q0")
        dh = r.get("delta_h0")
        dq_s = f"{dq:+.4f}" if dq is not None else "-"
        dh_s = f"{dh:+.4f}" if dh is not None else "-"
        print(f"{MODEL_NAMES[r['model']]:8s} {r['zup']:6.2f} {r['n_sne']:6d} | "
              f"{r['q0_bias']:+9.4f} {dq_s:>13s} | {r['h0_bias']:+9.4f} {dh_s:>13s} | "
              f"{r['dl_err']:+10.4%}")

    # ---- Safe zup summary ----
    print("\n" + "=" * 100)
    print(f"Largest safe zup per model (|q0 bias| <= {args.tolerance})")
    print("=" * 100)
    for name, code in zip(models, codes):
        zup_safe, bias = max_safe_zup(zz_all, zdown, code, tolerance=args.tolerance, zup_max=args.zup_max)
        n_sne = len(window_redshifts(zz_all, zup_safe, zdown))
        print(f"  {name:8s}: zup <= {zup_safe:.3f} ({n_sne} SNe, q0 bias {bias:+.4f})")

    # ---- CSV ----
    out_path = args.output or (
        f"{config.output.compilations}truncation_scan"
        f"(zdown={zdown})(models={'+'.join(models)})(tol={args.tolerance}).csv"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["model"] = df["model"].map(MODEL_NAMES)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


MODEL_NAMES = {code: name for name, code in MODEL_CODES.items()}


if __name__ == "__main__":
    main()
