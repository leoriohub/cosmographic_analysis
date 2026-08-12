"""Truncation-error diagnostics for the distance-modulus models.

Quantifies how much error each truncated dL expansion (taylor2, pade11,
pade21) introduces when the fit window [zdown, zup] is extended, using a
noiseless flat LCDM reference and the actual redshift distribution of the
SNe in each window.

Conventions:
- h0 in code units of 100 km/s/Mpc (e.g. 0.70).
- Reference cosmology: flat LCDM, Om=0.30, H0=70 km/s/Mpc, whose
  deceleration parameter is q0 = Om/2 - (1-Om) = -0.55.
- "Bias" of a fit = fitted_parameter - true_parameter, with the OTHER
  parameter held at its true value. Holding the partner at truth isolates
  the pure truncation error (the pipeline's per-hemisphere fits fix the
  partner at the fiducial config value instead; the difference is a small
  second-order effect on the bias).
- Fits are unweighted least squares over the window's SNe (no covariance,
  no hostyn split). The covariance weighting shifts the exact bias values
  slightly but does not change the parametrization or the safe-zup
  conclusions.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares

from cosmographic_analysis.cosmology import dl_model, mu_model

OM_LCDM = 0.30
H0_TRUE = 0.70  # code units: 100 km/s/Mpc
Q0_TRUE = -0.55  # flat LCDM with Om = 0.30
C_LIGHT = 299792.458  # km/s


def dl_lcdm_exact(z, Om=OM_LCDM, H0=70.0):
    """Exact flat LCDM luminosity distance in Mpc (scalar z)."""
    integ = quad(lambda zp: 1.0 / np.sqrt(Om * (1 + zp) ** 3 + (1 - Om)), 0, z)[0]
    return (C_LIGHT / H0) * (1 + z) * integ


def mu_lcdm_exact(z, Om=OM_LCDM, H0=70.0):
    """Exact flat LCDM distance modulus (scalar z)."""
    return 5.0 * np.log10(dl_lcdm_exact(z, Om, H0)) + 25.0


def dl_rel_error(z, model, h0=H0_TRUE, q0=Q0_TRUE):
    """Relative error of the model dL vs exact LCDM at fiducial (h0, q0)."""
    return dl_model(z, h0, q0, model) / dl_lcdm_exact(z) - 1.0


def q0_fit_bias(zz, model, h0=H0_TRUE, q0_true=Q0_TRUE):
    """Noiseless q0 fit bias (h0 held at truth) over the given redshifts.

    bias = fitted_q0 - q0_true; positive means the fit overestimates q0.
    """
    zz = np.asarray(zz, dtype=np.float64)
    mu_true = np.array([mu_lcdm_exact(zi) for zi in zz])
    res = least_squares(
        lambda q0: mu_model(zz, h0, q0[0], model) - mu_true, x0=[q0_true]
    )
    return res.x[0] - q0_true


def h0_fit_bias(zz, model, q0=Q0_TRUE, h0_true=H0_TRUE):
    """Noiseless h0 fit bias (q0 held at truth) over the given redshifts."""
    zz = np.asarray(zz, dtype=np.float64)
    mu_true = np.array([mu_lcdm_exact(zi) for zi in zz])
    res = least_squares(
        lambda h0: mu_model(zz, h0[0], q0, model) - mu_true, x0=[h0_true]
    )
    return res.x[0] - h0_true


def window_redshifts(zz_all, zup, zdown):
    """SNe redshifts in (zdown, zup)."""
    zz_all = np.asarray(zz_all, dtype=np.float64)
    return zz_all[(zz_all < zup) & (zz_all > zdown)]


def scan_biases(zz_all, zdown, zups, models=(0, 2), zup_anchor=0.1):
    """Bias table over zup values for each model code.

    Returns a list of dicts (one per model x zup):
    model, zup, n_sne, q0_bias, h0_bias, dl_err (dL rel. error at z=zup).
    Also adds 'delta_q0'/'delta_h0' keys on the anchor-zup row: the bias
    introduced by extending from zup_anchor to this zup.
    """
    zz_all = np.asarray(zz_all, dtype=np.float64)
    rows = []
    for model in models:
        anchor = {}
        for zup in zups:
            zz = window_redshifts(zz_all, zup, zdown)
            row = {
                "model": model,
                "zup": zup,
                "n_sne": len(zz),
                "q0_bias": q0_fit_bias(zz, model),
                "h0_bias": h0_fit_bias(zz, model),
                "dl_err": dl_rel_error(zup, model),
            }
            if np.isclose(zup, zup_anchor):
                anchor = row
            rows.append(row)
        for row in rows:
            if row["model"] == model and anchor:
                row["delta_q0"] = row["q0_bias"] - anchor["q0_bias"]
                row["delta_h0"] = row["h0_bias"] - anchor["h0_bias"]
    return rows


def max_safe_zup(zz_all, zdown, model, tolerance=0.05, zup_max=0.6, n_steps=400):
    """Largest zup in (zdown, zup_max] with |q0 fit bias| <= tolerance.

    Bias is a step function of zup (SNe enter discretely), so this is a
    fine scan rather than a bisection. Returns (zup, bias_at_zup).
    """
    zz_all = np.asarray(zz_all, dtype=np.float64)
    zups = np.linspace(zdown + 1e-4, zup_max, n_steps)
    best = (zdown, np.inf)
    for zup in zups:
        zz = window_redshifts(zz_all, zup, zdown)
        bias = q0_fit_bias(zz, model)
        if abs(bias) <= tolerance:
            best = (float(zup), float(bias))
    return best
