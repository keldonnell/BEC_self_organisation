import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import standard_data_utils as stand_utils


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
    return (
        data0[0].astype(int),  # nodes
        *data0[1:12],
        data0[12].astype(int),  # plotnum
        data0[13].astype(float),
    )


def extract_float(filename):
    match = re.search(r"psi\d+_([\d.e-]+)", filename)
    if match:
        float_str = match.group(1).rstrip(".")
        return float(float_str)
    return 0.0


def load_data(output_dir):
    data_files = glob.glob(os.path.join(output_dir, "psi*"))
    return sorted(data_files, key=extract_float)


def find_first_deviation_time(psi_data, x_index, tol=1e-6):
    spatial_data = psi_data[:, 1:]
    spatial_idx = int(np.clip(x_index, 0, spatial_data.shape[1] - 1))
    trace = spatial_data[:, spatial_idx]

    deviations = np.where(np.abs(trace - 1.0) > tol)[0]
    if deviations.size == 0:
        return np.nan
    idx = deviations[0]
    if idx >= psi_data.shape[0]:
        return np.nan

    return psi_data[idx, 0]


def compute_delay_times(sorted_files, p0_vals, x_index):
    delay_times = []
    valid_p0 = []

    for file, p0 in zip(sorted_files, p0_vals):
        data = np.loadtxt(file)
        t0 = find_first_deviation_time(data, x_index)
        if np.isnan(t0):
            continue
        valid_p0.append(p0)
        delay_times.append(t0)

    return np.array(valid_p0, dtype=float), np.array(delay_times, dtype=float)

def t0_analytic(p0, p_th, gambar, seed):

    M0 = seed
    return np.arccosh(np.sqrt(2) * (p_th / p0) * np.sqrt((p0 / p_th) - 1) / M0) / (np.sqrt((p0 / p_th) - 1) * gambar)

def create_plot(p0_vals, delay_times, p0_analtic_vals, t0_analtic_vals, p_th, x):

    fig, ax = plt.subplots(figsize=(9, 6))


    # Plotting the analtic function

    ax.plot(
        p0_analtic_vals,
        t0_analtic_vals,
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

    (
        nodes,
        maxt,
        ht,
        width_psi,
        p0,
        Delta,
        gambar,
        b0,
        num_crit,
        R,
        gbar,
        v0,
        plotnum,
        seed
    ) = read_input(seed_dir)

    x = float(args.xpos)
    x_index = int((np.abs(x + np.pi * num_crit) / (2 * np.pi * num_crit)) * nodes)

    sorted_files = load_data(output_dir)
    all_p0_vals = stand_utils.find_p0_vals_from_filenames(sorted_files)

    p_th = (2 * gambar) / (b0 * R)

    p0_above_th_vals, sorted_files_above_th = stand_utils.find_vals_above_th(all_p0_vals, sorted_files, p_th)

    p0_vals, delay_times = compute_delay_times(sorted_files_above_th, p0_above_th_vals, x_index)


    if p0_vals.size == 0:
        raise RuntimeError("Unable to determine any delay times above threshold.")

    num_analytic_datapoints = 2000
    above_threshold = p0_vals[p0_vals > (p_th * 1.001)]
    p0_min_for_curve = above_threshold.min() if above_threshold.size else p0_vals.min()
    p0_analytic_vals = np.linspace(p0_min_for_curve, p0_vals.max(), num_analytic_datapoints)
    t0_analtic_vals = t0_analytic(p0_analytic_vals, p_th, gambar, seed)
    print(t0_analtic_vals)

    create_plot(p0_vals, delay_times, p0_analytic_vals, t0_analtic_vals, p_th, x)
    plt.show()


if __name__ == "__main__":
    main()
