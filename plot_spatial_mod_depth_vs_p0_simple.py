import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt

from analytic_predictors import analytic_delay_time


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Simplified spatial modulation depth check."
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Name of the patt1d input/output directory pair.",
    )
    parser.add_argument(
        "-x",
        "--xpos",
        required=True,
        help="Spatial location (simulation units) to inspect.",
    )
    return parser.parse_args()


def read_seed_metadata(seed_path):
    data = np.genfromtxt(seed_path, skip_footer=1, comments="!")
    nodes = int(data[0])
    num_crit = data[8]
    gambar = data[6]
    b0 = data[7]
    R = data[9]
    seed = data[13] if len(data) > 13 else 0.0
    return nodes, num_crit, gambar, b0, R, seed


def extract_float(filename):
    match = re.search(r"psi\d+_([\d.e-]+)", filename)
    if match:
        float_str = match.group(1).rstrip(".")
        return float(float_str)
    return 0.0


def load_sorted_files(output_dir):
    paths = glob.glob(os.path.join(output_dir, "psi*"))
    return sorted(paths, key=extract_float)


def parse_p0_from_filename(path):
    match = re.search(r"(\d+(?:\.\d+)?e[-+]\d+)", path)
    if not match:
        raise ValueError(f"Could not extract p0 from {path}")
    return float(match.group(1))


def main():
    plt.rcParams["ps.usedistiller"] = "xpdf"

    args = parse_arguments()
    x_requested = float(args.xpos)

    output_dir = os.path.join("patt1d_outputs", args.filename)
    input_dir = os.path.join("patt1d_inputs", args.filename)
    seed_path = os.path.join(input_dir, "seed.in")

    nodes, num_crit, gambar, b0, R, seed = read_seed_metadata(seed_path)

    # Map the requested physical position onto the discrete grid index.
    frac = (np.abs(x_requested + np.pi * num_crit) / (2 * np.pi * num_crit)) * nodes
    spatial_index = int(np.clip(frac, 0, nodes - 1))

    sorted_files = load_sorted_files(output_dir)
    if not sorted_files:
        raise FileNotFoundError("No psi files found in the specified output directory.")

    p_th = (2 * gambar) / (b0 * R)

    p0_vals = []
    mod_depth_vals = []
    mod_depth_t0_vals = []
    p0_vals_t0 = []

    for path in sorted_files:
        psi_data = np.loadtxt(path)

        # Column 0 holds the time array; spatial data start at column 1.
        time_series_at_x = psi_data[:, spatial_index + 1]
        peak_time_index = int(np.argmax(time_series_at_x))

        spatial_cut = psi_data[peak_time_index, 1:]
        max_val = np.max(spatial_cut)
        min_val = np.min(spatial_cut)
        denom = max_val + min_val
        mod_depth = (max_val - min_val) / denom if denom != 0 else 0.0

        p0_vals.append(parse_p0_from_filename(path))
        mod_depth_vals.append(mod_depth)

        # Evaluate modulation depth near the analytic delay time t0.
        p0_current = p0_vals[-1]
        t0_estimate = analytic_delay_time(p0_current, p_th, gambar, seed)
        if np.isfinite(t0_estimate):
            time_array = psi_data[:, 0]
            nearest_time_index = int(np.argmin(np.abs(time_array - t0_estimate)))
            spatial_cut_t0 = psi_data[nearest_time_index, 1:]
            max_val_t0 = np.max(spatial_cut_t0)
            min_val_t0 = np.min(spatial_cut_t0)
            denom_t0 = max_val_t0 + min_val_t0
            mod_depth_t0 = (max_val_t0 - min_val_t0) / denom_t0 if denom_t0 != 0 else 0.0
            p0_vals_t0.append(p0_current)
            mod_depth_t0_vals.append(mod_depth_t0)

    p0_vals = np.array(p0_vals)
    mod_depth_vals = np.array(mod_depth_vals)
    p0_vals_t0 = np.array(p0_vals_t0)
    mod_depth_t0_vals = np.array(mod_depth_t0_vals)

    if p0_vals.size:
        p0_min, p0_max = p0_vals.min(), p0_vals.max()
        if p0_min == p0_max:
            p0_samples = np.array([p0_min])
        else:
            p0_samples = np.linspace(p0_min, p0_max, max(len(p0_vals), 200))
        with np.errstate(divide="ignore", invalid="ignore"):
            m_max_vals = np.sqrt(2 * p_th * np.maximum(p0_samples - p_th, 0) / (p0_samples**2))
            m_max_vals = np.where(p0_samples > 0, m_max_vals, np.nan)
    else:
        p0_samples = np.array([])
        m_max_vals = np.array([])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(p0_vals, mod_depth_vals, color="black", marker="x", label="Simulation (peak time)")

    if mod_depth_t0_vals.size:
        ax.scatter(
            p0_vals_t0,
            mod_depth_t0_vals,
            color="tab:orange",
            marker="o",
            s=35,
            label=r"Simulation @ $t_0$",
        )

    if m_max_vals.size:
        ax.plot(p0_samples, m_max_vals, color="black", linewidth=1.0, label=r"$M_{max}$ analytic")

    ax.axvline(p_th, color="k", linestyle="--", linewidth=1.0, label=r"$p_{th}$")

    ax.set_xlabel(r"Pump strength $p_0$")
    ax.set_ylabel(r"Modulation depth $m[|\psi|^2]$")
    ax.set_title(rf"Simplified spatial modulation depth at $x = {x_requested}$")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
