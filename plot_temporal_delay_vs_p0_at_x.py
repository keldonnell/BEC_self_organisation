import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import standard_data_utils as stand_utils
from analytic_predictors import analytic_delay_time, pump_threshold

RELATIVE_HEIGHT_THRESHOLD = 1e-3
MIN_PROMINENCE = 1e-6


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot modulation delay time t0 versus p0.")
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
        help="Spatial location to analyse (in the same units used by the simulation).",
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


def find_first_peak_time(psi_data, x_index):
    spatial_data = psi_data[:, 1:]
    spatial_idx = int(np.clip(x_index, 0, spatial_data.shape[1] - 1))
    trace = spatial_data[:, spatial_idx]

    baseline = trace[0]
    peak_indices, properties = find_peaks(trace, prominence=0.25 * np.max(trace))

    if len(peak_indices) > 0:
        return psi_data[peak_indices[0], 0]
    else:
        print("Didn't find any peaks")

    return np.nan


def compute_delay_times(sorted_files, p0_vals, x_index):
    delay_times = []
    valid_p0 = []

    for file, p0 in zip(sorted_files, p0_vals):
        data = np.loadtxt(file)
        t0 = find_first_peak_time(data, x_index)
        if np.isnan(t0):
            continue
        valid_p0.append(p0)
        delay_times.append(t0)

    return np.array(valid_p0, dtype=float), np.array(delay_times, dtype=float)

def create_plot(p0_vals, delay_times, p0_analytic_vals, t0_analytic_vals, p_th, x):

    fig, ax = plt.subplots(figsize=(9, 6))


    # Plotting the analtic function

    ax.plot(
        p0_analytic_vals,
        t0_analytic_vals,
        color="C0",
        linewidth=1.5,
        zorder=2,
        label="Analytic $t_0$",
    )
    ax.scatter(
        p0_vals,
        delay_times,
        color="black",
        marker="x",
        s=55,
        linewidths=1.1,
        zorder=3,
    )

    ax.axvline(p_th, color="k", linestyle="--", linewidth=1.2, label=r"$p_{th}$", zorder=1)

    ax.set_xlabel(r"Pump strength $p_0$", fontsize=13)
    ax.set_ylabel(r"Delay time $t_0$", fontsize=13)
    ax.set_title(rf"Modulation delay time at $x = {x}$", fontsize=14)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.grid(which="major", linestyle=":", alpha=0.4)
    ax.grid(which="minor", linestyle=":", alpha=0.15)
    ax.legend(frameon=False, loc="best")

    if p0_vals.size:
        ax.set_xlim(p0_vals.min() * 0.9, p0_vals.max() * 1.02)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig, ax


def main():
    plt.rcParams["ps.usedistiller"] = "xpdf"

    args = parse_arguments()

    output_dir = os.path.join("patt1d_outputs", args.filename)
    input_dir = os.path.join("patt1d_inputs", args.filename)
    seed_dir = os.path.join(input_dir, "seed.in")

    params = read_input(seed_dir)

    nodes = params["nodes"]
    num_crit = params["num_crit"]
    b0 = params["b0"]
    reflectivity = params["R"]
    gamma_bar = params["gamma_bar"]
    seed = params["seed"]

    x = float(args.xpos)
    x_index = int((np.abs(x + np.pi * num_crit) / (2 * np.pi * num_crit)) * nodes)

    sorted_files = load_data(output_dir)
    all_p0_vals = stand_utils.find_p0_vals_from_filenames(sorted_files)

    p_th = pump_threshold(gamma_bar, b0, reflectivity)

    p0_above_th_vals, sorted_files_above_th = stand_utils.find_vals_above_th(all_p0_vals, sorted_files, p_th)

    p0_vals, delay_times = compute_delay_times(sorted_files_above_th, p0_above_th_vals, x_index)


    if p0_vals.size == 0:
        raise RuntimeError("Unable to determine any delay times above threshold.")

    num_analytic_datapoints = 2000
    above_threshold = p0_vals[p0_vals > (p_th * 1.001)]
    p0_min_for_curve = above_threshold.min() if above_threshold.size else p0_vals.min()
    p0_analytic_vals = np.linspace(p0_min_for_curve, p0_vals.max(), num_analytic_datapoints)
    t0_analytic_vals = analytic_delay_time(p0_analytic_vals, p_th, gamma_bar, seed)
    print(t0_analytic_vals)

    create_plot(p0_vals, delay_times, p0_analytic_vals, t0_analytic_vals, p_th, x)
    plt.show()


if __name__ == "__main__":
    main()
