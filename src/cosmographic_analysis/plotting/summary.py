"""Summary table generation.

Saves pipeline results as CSV (machine-readable) and
renders a clean table image (for papers/documents).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_summary_tables(
    delta_h0_data_max: float,
    delta_q0_data_max: float,
    p_h0_iso_max: float,
    p_q0_iso_max: float,
    p_h0_lcdm_max: float,
    p_q0_lcdm_max: float,
    prefix_name: str,
    h0f: float,
    q0f: float,
    zup: float,
    zdown: float,
    pts: int,
    n_rep: int,
    tables_dir: str,
    optimizer: str = "golden",
    model: str = "taylor2",
):
    """Save summary tables as CSV and rendered PNG images.

    Produces two tables (h0 and q0) with observed anisotropy,
    ISO p-values, and LCDM p-values. Each table is saved as
    a CSV file and a formatted PNG image.

    Parameters
    ----------
    optimizer : str
        Optimizer name used in the analysis (e.g. 'golden' or 'brent').
        Included in the output filename suffix.
    model : str
        Distance model name (e.g. 'taylor2' or 'pade21').
        Included in the output filename suffix.

    """
    tables_path = Path(tables_dir)
    tables_path.mkdir(parents=True, exist_ok=True)

    z_range_str = f"{zup} > z > {zdown}"

    # Build data
    h0_data = {
        "z range": z_range_str,
        "delta h0": f"{delta_h0_data_max:.4f}",
        "ISO p-value": f"{p_h0_iso_max:.4f}",
        "LCDM p-value": f"{p_h0_lcdm_max:.4f}",
    }
    q0_data = {
        "z range": z_range_str,
        "delta q0": f"{delta_q0_data_max:.4f}",
        "ISO p-value": f"{p_q0_iso_max:.4f}",
        "LCDM p-value": f"{p_q0_lcdm_max:.4f}",
    }

    suffix = (
        f"({h0f}=h0f_{q0f}=q0f)"
        f"_({zup}>z>{zdown})"
        f"({pts}_pts)_({n_rep})_rep_"
        f"(method={optimizer})(model={model})"
    )

    for label, data in [("h0", h0_data), ("q0", q0_data)]:
        df = pd.DataFrame([data])

        # CSV
        csv_path = tables_path / f"{prefix_name}[{label}_table]{suffix}.csv"
        df.to_csv(csv_path, index=False)

        # Rendered PNG
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.axis("tight")
        tbl = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        for key, cell in tbl.get_celld().items():
            cell.set_edgecolor("gray")
            if key[0] == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#e8e8e8")

        png_path = tables_path / f"{prefix_name}[{label}_table]{suffix}.png"
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()
