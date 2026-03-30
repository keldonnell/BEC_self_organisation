from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

import fourier_utils as ft_utils


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot the first-harmonic Fourier amplitude of density over time."
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Name of the patt1d input/output directory pair (e.g. ivan_diss_params).",
    )
    parser.add_argument(
        "-i",
        "--frame_index",
        type=int,
        required=True,
        help="Index of the psi file to analyze.",
    )
    return parser.parse_args()


def read_input(seed_dir):
    data = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    return {
        "nodes": int(data[0]),
        "num_crit": float(data[8]),
        "plotnum": int(data[12]),
    }


def load_psi_data(output_dir, frame_index):
    psi_path = glob.glob(os.path.join(output_dir, f"psi{frame_index}_*"))
    if not psi_path:
        raise FileNotFoundError(
            f"No psi file found for index {frame_index} in {output_dir}"
        )
    return np.loadtxt(psi_path[0])


def build_spatial_freqs(num_nodes, num_crit):
    domain_len = 2 * np.pi * num_crit
    dx = domain_len / num_nodes
    freq_vals = np.fft.fftfreq(num_nodes, d=dx) * 2 * np.pi
    return np.abs(freq_vals[: num_nodes // 2])


def compute_first_harmonic_amplitude_over_time(psi_data, freq_vals):
    spatial_data = psi_data[:, 1:]
    num_nodes = spatial_data.shape[1]
    half_nodes = num_nodes // 2

    amp_trace = np.zeros(spatial_data.shape[0], dtype=float)
    for i, spatial_cut in enumerate(spatial_data):
        fft_vals = np.abs(np.fft.fft(spatial_cut, norm="forward")[:half_nodes])
        _, first_amp, _ = ft_utils.find_first_harmonic(fft_vals, freq_vals)
        amp_trace[i] = first_amp

    return amp_trace


def main():
    plt.rcParams["ps.usedistiller"] = "xpdf"

    args = parse_arguments()
    output_dir = os.path.join("patt1d_outputs", args.filename)
    input_dir = os.path.join("patt1d_inputs", args.filename)
    seed_dir = os.path.join(input_dir, "seed.in")

    params = read_input(seed_dir)
    psi_data = load_psi_data(output_dir, args.frame_index)

    num_nodes = psi_data.shape[1] - 1
    freq_vals = build_spatial_freqs(num_nodes, params["num_crit"])

    times = psi_data[:, 0]
    amp_trace = compute_first_harmonic_amplitude_over_time(psi_data, freq_vals)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(times, amp_trace, color="tab:blue", linewidth=1.5)
    ax.set_xlabel(r"Time $\Gamma t$", fontsize=14)
    ax.set_ylabel("1st harmonic amplitude", fontsize=14)
    ax.set_title(
        f"Density 1st harmonic amplitude vs time (index {args.frame_index})",
        fontsize=14,
    )
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
