"""Mollweide sky map plotting utilities.

Generates all-sky maps of best-fit h0 and q0 values using HEALPix.
"""

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from cosmographic_analysis.config import Config
from cosmographic_analysis.maps import generate_map


def plot_h0_q0_maps(
    nside: int,
    theta: np.ndarray,
    phi: np.ndarray,
    h0: np.ndarray,
    q0: np.ndarray,
    h0f: float,
    q0f: float,
    config: Config,
    optimizer: str = "golden",
    model: str = "taylor2",
) -> str:
    """Generate and save Mollweide sky maps for h0 and q0.

    Parameters
    ----------
    optimizer : str, optional
        Optimization method used (default 'golden'). Included in output filename.
    model : str, optional
        Distance model used (default 'taylor2'). Included in output filename.

    Returns
    -------
    str
        Path to the saved figure.

    """
    h0map, q0map = generate_map(nside, theta, phi, h0, q0)
    p = config.parameters
    o = config.output

    plt.figure(figsize=(6, 7))
    hp.mollview(
        h0map, coord="cg",
        title=rf"$h_0$ map (fixed $q_0$={q0f})",
        unit="", notext=True, norm="hist", cmap="jet",
        min=min(h0map), max=max(h0map),
        fig=1, sub=(1, 2, 1),
    )
    hp.mollview(
        q0map, coord="cg",
        title=rf"$q_0$ map (fixed $h_0$={h0f})",
        unit="", notext=True, norm="hist", cmap="jet",
        min=min(q0map), max=max(q0map),
        fig=1, sub=(2, 2, 1),
    )

    map_filename = (
        f"{o.figures}{p.prefix_name}[VERTICAL]"
        f"(hf={h0f}_qf={q0f})({p.zup}>z>{p.zdown})"
        f"(method={optimizer})(model={model}).png"
    )
    plt.savefig(map_filename, dpi=400, bbox_inches="tight")
    plt.close()
    return map_filename
