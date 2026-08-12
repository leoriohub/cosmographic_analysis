"""Padé distance-modulus model tests.

Thresholds grounded on noiseless ΛCDM fits this session:
- z=0.3 relative dL error: taylor2 −10.3%, pade21 +0.7%
- q0 truncation bias over 0.01<z<0.1 (true −0.55): taylor2 −0.336, pade21 +0.004
- pade21 pole minimum over q0 ∈ [−1.5, 0.5]: z = 1.256
"""
import numpy as np
from scipy.optimize import least_squares

from cosmographic_analysis.cosmology import dl, dl_model, mu, mu_model
from cosmographic_analysis.config import ParametersConfig, load_config
from cosmographic_analysis.truncation import dl_lcdm_exact, mu_lcdm_exact


def test_model0_equals_mu():
    """model=0 must be bit-identical to the legacy taylor2 mu/dl."""
    zs = np.linspace(0.01, 0.5, 50)
    h0, q0 = 0.7304, -0.574
    np.testing.assert_array_equal(dl(zs, h0, q0), dl_model(zs, h0, q0, 0))
    np.testing.assert_array_equal(mu(zs, h0, q0), mu_model(zs, h0, q0, 0))


def test_accuracy_vs_lcdm_at_z03():
    """At z=0.3 taylor2 underestimates ΛCDM dL by >5%; pade21 is within 2%."""
    z = 0.3
    dL_true = dl_lcdm_exact(z)
    # code h0 convention: units of 100 km/s/Mpc
    err_taylor2 = dl_model(z, 0.70, -0.55, 0) / dL_true - 1.0
    err_pade21 = dl_model(z, 0.70, -0.55, 2) / dL_true - 1.0
    assert err_taylor2 < -0.05, f"taylor2 err {err_taylor2:.4f} should be below -5%"
    assert abs(err_pade21) < 0.02, f"pade21 err {err_pade21:.4f} should be within 2%"


def test_q0_truncation_bias_gain():
    """Noiseless ΛCDM over 0.01<z<0.1: taylor2 biases q0 by >0.25, pade21 by <0.05."""
    zz = np.linspace(0.01, 0.1, 400)
    mu_true = np.array([mu_lcdm_exact(zi) for zi in zz])
    biases = {}
    for model, name in [(0, "taylor2"), (2, "pade21")]:
        res = least_squares(
            lambda q0: mu_model(zz, 0.70, q0[0], model) - mu_true, x0=[-0.5]
        )
        biases[name] = res.x[0] + 0.55  # true q0 = -0.55
    assert abs(biases["taylor2"]) > 0.25, f"taylor2 bias {biases['taylor2']:.4f}"
    assert abs(biases["pade21"]) < 0.05, f"pade21 bias {biases['pade21']:.4f}"


def test_pade21_pole_outside_window():
    """pade21 pole in z stays above 1.0 for all q0 in the fit range."""
    q0s = np.linspace(-1.5, 0.5, 1000)
    f2 = (3.0 - q0s) / 2.0
    f3 = (10.0 - 5.0 * q0s + 3.0 * q0s * q0s) / 6.0
    z_pole = f2 / (f3 - f2)  # y_pole = f2/f3, z = y/(1-y)
    assert np.min(z_pole) > 1.0, f"pole z={np.min(z_pole):.4f} inside window"


def test_config_model_key(tmp_path):
    """Config loads the model key; default stays taylor2."""
    assert ParametersConfig().model == "taylor2"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("parameters:\n  model: pade21\n")
    assert load_config(str(cfg_path)).parameters.model == "pade21"
