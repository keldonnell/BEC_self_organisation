import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import fourier_utils as ft_utils
import standard_data_utils as stand_utils
from analytic_predictors import analytic_delay_time, pump_threshold


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot modulation delay time t0 versus p0 using the time of maximum "
            "first-harmonic Fourier amplitude."
        )
    )
    parser.add_argument(
        "-f",
        "--filename",
        metavar="filename",
        required=True,
        help="Name of the patt1d input/output directory pair (e.g. ivan_diss_params).",
    )
    parser.add_argument(
        "-x",
        "--xpos",
        metavar="x_position",
        required=True,
        help="Spatial location to analyse (kept for CLI compatibility).",
    )
    parser.add_argument(
        "--csv",
        nargs="?",
        const="temporal_delay_vs_p0_at_x_using_fourier_amp.csv",
        metavar="csv_path",
        help=(
            "Export plotted datapoints (p0, delay_time) to CSV. "
            "Optionally provide an output path."
        ),
    )
    return parser.parse_args()


def read_input(seed_dir):
    data0 = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    gamma_bar_val = float(data0[6])
    return {
        "nodes": int(data0[0]),
        "maxt": float(data0[1]),
        "ht": float(data0[2]),
        "width_psi": float(data0[3]),
        "p0": float(data0[4]),
        "Delta": float(data0[5]),
        "gamma_bar": gamma_bar_val,
        "omega_r": gamma_bar_val,
        "b0": float(data0[7]),
        "num_crit": float(data0[8]),
        "R": float(data0[9]),
        "gbar": float(data0[10]),
        "v0": float(data0[11]),
        "plotnum": int(data0[12]),
        "seed": float(data0[13]),
    }


def extract_float(filename):
    match = re.search(r"psi\d+_([\d.e-]+)", filename)
    if match:
        float_str = match.group(1).rstrip(".")
        return float(float_str)
    return 0.0


def load_data(output_dir):
    data_files = glob.glob(os.path.join(output_dir, "psi*"))
    return sorted(data_files, key=extract_float)


def build_spatial_freqs(num_nodes, num_crit):
    domain_len = 2 * np.pi * num_crit
    dx = domain_len / num_nodes
    freq_vals = np.fft.fftfreq(num_nodes, d=dx) * 2 * np.pi
    return np.abs(freq_vals[: num_nodes // 2])


def find_first_peak_time(psi_data, freq_vals):
    if psi_data.ndim != 2 or psi_data.shape[1] < 3:
        return np.nan

    spatial_data = psi_data[:, 1:]
    num_nodes = spatial_data.shape[1]
    half_nodes = num_nodes // 2

    first_amp_trace = np.zeros(spatial_data.shape[0], dtype=float)
    for i, spatial_cut in enumerate(spatial_data):
        fft_vals = np.abs(np.fft.fft(spatial_cut, norm="forward")[:half_nodes])
        _, first_amp, _ = ft_utils.find_first_harmonic(fft_vals, freq_vals)
        first_amp_trace[i] = first_amp

    if np.all(first_amp_trace <= 0):
        return np.nan

    prominence = 0.25 * np.max(first_amp_trace)
    peak_indices, _ = find_peaks(first_amp_trace, prominence=prominence)
    if len(peak_indices) == 0:
        return np.nan

    first_peak_idx = int(peak_indices[0])
    return float(psi_data[first_peak_idx, 0])


def compute_delay_times(sorted_files, p0_vals, freq_vals):
    delay_times = []
    valid_p0 = []

    for file, p0 in zip(sorted_files, p0_vals):
        data = np.loadtxt(file)
        t0 = find_first_peak_time(data, freq_vals)
        if np.isnan(t0):
            continue
        valid_p0.append(p0)
        delay_times.append(t0)

    return np.array(valid_p0, dtype=float), np.array(delay_times, dtype=float)


def create_plot(p0_vals, delay_times, p0_analytic_vals, t0_analytic_vals, p_th):
    fig, ax = plt.subplots()

    ax.plot(
        p0_analytic_vals,
        t0_analytic_vals,
        color="black",
        linewidth=1,
        zorder=2,
        # label="Analytic $t_0$",
    )
    ax.plot(
        p0_vals,
        delay_times,
        color="tab:red",
        marker="x",
        linestyle="None",
        markersize=5,
        linewidth=1,
        alpha=0.8,
        zorder=3,
        # label=r"Observed $t_0$",
    )

    ax.axvline(p_th, color="k", linestyle="--", linewidth=1.2, zorder=1)

    ax.set_xlabel(r"$p_0$", fontsize=18)
    ax.set_ylabel(r"$\bar{t_D}$", fontsize=18)
    ax.legend(frameon=False, loc="lower right", fontsize=16)
    # Force scientific notation with math text so ticks use ×10^n style.
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)

    if p0_vals.size:
        ax.set_xlim(p_th * 0.98, p0_vals.max() * 1.02)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig, ax


def export_delay_csv(csv_path, p0_vals, delay_times):
    out_data = np.column_stack((p0_vals, delay_times))
    np.savetxt(csv_path, out_data, delimiter=",", header="p0,delay_time", comments="")


def main():
    plt.rcParams["ps.usedistiller"] = "xpdf"

    args = parse_arguments()

    output_dir = os.path.join("patt1d_outputs", args.filename)
    input_dir = os.path.join("patt1d_inputs", args.filename)
    seed_dir = os.path.join(input_dir, "seed.in")

    params = read_input(seed_dir)

    num_crit = params["num_crit"]
    b0 = params["b0"]
    reflectivity = params["R"]
    gamma_bar = params["gamma_bar"]
    seed = params["seed"]

    sorted_files = load_data(output_dir)
    all_p0_vals = stand_utils.find_p0_vals_from_filenames(sorted_files)

    p_th = pump_threshold(gamma_bar, b0, reflectivity)
    p0_above_th_vals, sorted_files_above_th = stand_utils.find_vals_above_th(
        all_p0_vals, sorted_files, p_th
    )

    if not sorted_files_above_th:
        raise RuntimeError("No simulation files were found above threshold.")

    num_nodes = np.loadtxt(sorted_files_above_th[0]).shape[1] - 1
    freq_vals = build_spatial_freqs(num_nodes, num_crit)

    p0_vals, delay_times = compute_delay_times(
        sorted_files_above_th, p0_above_th_vals, freq_vals
    )

    if p0_vals.size == 0:
        raise RuntimeError("Unable to determine any delay times above threshold.")

    num_analytic_datapoints = 2000
    above_threshold = p0_vals[p0_vals > (p_th * 1.001)]
    p0_min_for_curve = above_threshold.min() if above_threshold.size else p0_vals.min()
    p0_analytic_vals = np.linspace(p0_min_for_curve, p0_vals.max(), num_analytic_datapoints)
    t0_analytic_vals = analytic_delay_time(p0_analytic_vals, p_th, gamma_bar, seed)

    create_plot(p0_vals, delay_times, p0_analytic_vals, t0_analytic_vals, p_th)
    if args.csv:
        export_delay_csv(args.csv, p0_vals, delay_times)
    plt.savefig("temporal_delay_vs_p0_at_x_using_fourier_amp.pdf", dpi=300)
    plt.savefig("temporal_delay_vs_p0_at_x_using_fourier_amp.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
