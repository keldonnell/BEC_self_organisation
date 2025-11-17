import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import fourier_utils as ft_utils
import standard_data_utils as stand_utils

FLOAT_PATTERN = re.compile(r"(\d+(?:\.\d+)?e[-+]\d+)", re.IGNORECASE)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot analytic omega/Gamma versus pump strength.")
    parser.add_argument(
        "-f",
        "--filename",
        metavar="name",
        required=True,
        help="Name of the patt1d input/output directory pair (e.g. high_res).",
    )
    parser.add_argument(
        "-x",
        "--xpos",
        metavar="x_position",
        default=0.0,
        type=float,
        help="Spatial position (in simulation units) to track when extracting observed peaks.",
    )
    return parser.parse_args()


def read_seed_metadata(seed_path):
    data = np.genfromtxt(seed_path, skip_footer=1, comments="!")
    return {
        "nodes": int(data[0]),
        "omega_r": data[6],
        "b0": data[7],
        "num_crit": data[8],
        "R": data[9],
    }


def compute_x_index(x_pos, nodes, num_crit):
    """
    Convert a spatial position (in simulation units) into the column index used for traces.
    """
    if nodes <= 1:
        return 1

    if num_crit <= 0:
        return max(1, min(nodes // 2, nodes))

    domain = 2 * np.pi * num_crit
    frac = (x_pos + np.pi * num_crit) / domain
    frac = np.clip(frac, 0.0, 1.0)
    column = int(np.clip(frac * nodes, 0, nodes))
    return max(1, min(column, nodes))


def collect_psi_files(output_dir):
    psi_files = glob.glob(os.path.join(output_dir, "psi*.out"))
    filtered = [path for path in psi_files if FLOAT_PATTERN.search(os.path.basename(path))]
    if not filtered:
        raise FileNotFoundError(f"No psi*.out files with embedded p0 values found in {output_dir}")
    return sorted(filtered, key=lambda f: float(FLOAT_PATTERN.search(f).group(1)))


def extract_peak_profile(psi_path, nodes):
    data = np.loadtxt(psi_path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{psi_path} does not look like a psi output table.")

    centre_idx = max(1, min(nodes // 2, data.shape[1] - 1))
    time_index = int(np.argmax(data[:, centre_idx]))
    return data[time_index, 1:]  # discard the time column


def compute_analytic_omega_gamma(psi_files, params):
    p0_vals = stand_utils.find_p0_vals_from_filenames(psi_files)
    if p0_vals.size == 0:
        raise ValueError("Unable to extract p0 values from psi filenames.")

    omega_gamma_vals = []
    nqc_vals = []

    for file_path, p0 in zip(psi_files, p0_vals):
        psi_snapshot = extract_peak_profile(file_path, params["nodes"])
        n_qc = ft_utils.calculate_nqc_amplitude(psi_snapshot, params["num_crit"])
        # Analytic omega/Gamma definition: sqrt(b0 * R * p0 * |n^{q_c}| * omega_r)
        omega_gamma = np.sqrt(params["b0"] * params["R"] * p0 * n_qc * params["omega_r"])
        nqc_vals.append(n_qc)
        omega_gamma_vals.append(omega_gamma)

    return p0_vals, np.array(omega_gamma_vals), np.array(nqc_vals)


def plot_results(p0_vals, omega_vals, observed_p0, observed_omega, folder_name):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p0_vals, omega_vals, marker="o", linewidth=1.2, label="Analytic")

    if observed_p0.size:
        ax.plot(
            observed_p0,
            observed_omega,
            marker="x",
            linestyle="--",
            linewidth=1.2,
            label="Observed",
        )
    else:
        print("No observed omega/Gamma values passed to plotting routine.")

    ax.set_xlabel(r"$p_0$")
    ax.set_ylabel(r"$\omega / \Gamma$")
    ax.set_title(rf"Analytic vs observed $\omega / \Gamma$ for {folder_name}")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig, ax

def compute_observed_omega_gamma(psi_files, params, x_index):
    p0_vals = stand_utils.find_p0_vals_from_filenames(psi_files)
    observed_p0 = []
    observed_omega = []

    for file_path, p0 in zip(psi_files, p0_vals):
        data = np.loadtxt(file_path)

        if data.ndim != 2 or data.shape[1] < 2:
            print(f"Skipping p0={p0:.3e} ({os.path.basename(file_path)}): data shape invalid.")
            continue

        times = data[:, 0]
        spatial_idx = int(np.clip(x_index, 1, data.shape[1] - 1))
        trace = data[:, spatial_idx]

        amp_range = np.max(trace) - np.min(trace)
        if amp_range <= 0:
            print(f"Skipping p0={p0:.3e}: trace is flat, cannot detect peaks.")
            continue

        prominence = amp_range / stand_utils.PEAK_PROMINENCE_FACTOR
        peak_kwargs = {"prominence": prominence} if prominence > 0 else {}
        peaks, _ = find_peaks(trace, **peak_kwargs)

        if peaks.size < 2:
            print(f"Skipping p0={p0:.3e}: fewer than two peaks detected.")
            continue

        first_two = peaks[:2]
        gamma_t = times[first_two[1]] - times[first_two[0]]
        if gamma_t <= 0:
            print(f"Skipping p0={p0:.3e}: invalid peak spacing (gamma_t={gamma_t}).")
            continue

        omega_gamma = (2 * np.pi) / gamma_t
        observed_p0.append(p0)
        observed_omega.append(omega_gamma)

    return np.array(observed_p0), np.array(observed_omega)


def main():
    args = parse_arguments()

    output_dir = os.path.join("patt1d_outputs", args.filename)
    input_dir = os.path.join("patt1d_inputs", args.filename)
    seed_path = os.path.join(input_dir, "seed.in")

    params = read_seed_metadata(seed_path)
    psi_files = collect_psi_files(output_dir)

    x_index = compute_x_index(float(args.xpos), params["nodes"], params["num_crit"])

    p0_vals, omega_vals, _ = compute_analytic_omega_gamma(psi_files, params)
    observed_p0, observed_omega = compute_observed_omega_gamma(psi_files, params, x_index)
    plot_results(p0_vals, omega_vals, observed_p0, observed_omega, args.filename)


if __name__ == "__main__":
    main()
