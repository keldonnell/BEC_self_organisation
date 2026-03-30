from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np

import standard_data_utils as stand_utils

plt.rcParams["ps.usedistiller"] = "xpdf"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot S(x, t) evolution at a chosen spatial point.")
    parser.add_argument("-f", "--filename", required=True, help="Base name of the patt1d input/output directories.")
    parser.add_argument(
        "-x",
        "--xpos",
        type=float,
        required=True,
        help="Spatial coordinate to inspect.",
    )
    parser.add_argument("-i", "--frame_index", type=int, help="Specific frame index to load (if multiple snapshots exist).")
    parser.add_argument(
        "-t",
        "--trail_x_max",
        required=True,
        choices=["True", "False"],
        help="Whether to follow the peak that starts near xpos (True/False).",
    )
    return parser.parse_args()


def read_input(seed_path):
    data = np.genfromtxt(seed_path, skip_footer=1, comments="!")
    return int(data[0]), float(data[8])


def load_s_data(output_dir, frame_index):
    if frame_index is None:
        s_files = glob.glob(f"{output_dir}s*")
        if len(s_files) != 1:
            raise RuntimeError("Multiple S-files found; specify --frame_index.")
        return np.loadtxt(s_files[0])

    matches = glob.glob(f"{output_dir}s{frame_index}_*")
    if not matches:
        raise FileNotFoundError(f"No S file matching frame index {frame_index}.")
    return np.loadtxt(matches[0])


def compute_x_index(x_pos, nodes, num_crit):
    if num_crit == 0:
        return max(1, min(int(round(x_pos)), nodes - 1))
    frac = (x_pos + np.pi * num_crit) / (2 * np.pi * num_crit)
    frac = np.clip(frac, 0.0, 1.0)
    column = int(frac * nodes)
    return max(1, min(column, nodes - 1))


def main():
    args = parse_arguments()

    output_dir = f"patt1d_outputs/{args.filename}/"
    input_dir = f"patt1d_inputs/{args.filename}/"
    nodes, num_crit = read_input(f"{input_dir}seed.in")

    s_data = load_s_data(output_dir, args.frame_index)

    x_index = compute_x_index(args.xpos, nodes, num_crit)
    x_index = int(x_index)

    if args.trail_x_max.upper() == "TRUE":
        s_cut_vals = stand_utils.find_temporal_cut_of_x_peaks(s_data, x_index)
    elif args.trail_x_max.upper() == "FALSE":
        s_cut_vals = s_data[:, x_index]
    else:
        raise ValueError("Invalid --trail_x_max flag; use True or False.")

    t_vals = s_data[:, 0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Evolution of S at x = {args.xpos}")
    ax.set_xlabel(r"$\Gamma t$", fontsize=14)
    ax.set_ylabel("S", fontsize=14)
    ax.plot(t_vals, s_cut_vals)
    plt.show()


if __name__ == "__main__":
    main()
