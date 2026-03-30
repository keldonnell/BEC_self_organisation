import argparse
import glob
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["ps.usedistiller"] = "xpdf"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a single snapshot of s and |Psi|^2 at a given time index."
    )
    parser.add_argument(
        "-f",
        "--filename",
        metavar="filename",
        required=True,
        help="Simulation folder name under patt1d_outputs/ and patt1d_inputs/",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        required=True,
        help=(
            "If output uses sN_*/psiN_* files: N file index to load. "
            "If output uses s.out/psi.out: time-row index to plot."
        ),
    )
    parser.add_argument(
        "-t",
        "--time-index",
        type=int,
        default=-1,
        help=(
            "Time-row index inside the selected file (used with sN_*/psiN_* layout). "
            "Default -1 (last row)."
        ),
    )
    return parser.parse_args()


def read_input(seed_dir):
    data0 = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    nodes = int(data0[0])
    num_crit = data0[8]
    return nodes, num_crit


def resolve_data_paths(output_dir, index):
    s_out = os.path.join(output_dir, "s.out")
    psi_out = os.path.join(output_dir, "psi.out")
    if os.path.exists(s_out) and os.path.exists(psi_out):
        return "single", s_out, psi_out

    s_matches = glob.glob(os.path.join(output_dir, f"s{index}_*"))
    psi_matches = glob.glob(os.path.join(output_dir, f"psi{index}_*"))
    if len(s_matches) == 1 and len(psi_matches) == 1:
        return "indexed", s_matches[0], psi_matches[0]

    raise FileNotFoundError(
        f"Could not find s.out/psi.out or matching s{index}_*/psi{index}_* in {output_dir}"
    )


def main():
    args = parse_args()

    output_dir = f"patt1d_outputs/{args.filename}/"
    input_dir = f"patt1d_inputs/{args.filename}/"
    mode, s_path, psi_path = resolve_data_paths(output_dir, args.index)
    seed_path = input_dir + "seed.in"

    print("Loading " + s_path)
    s_data = np.loadtxt(s_path)
    print("Loading " + psi_path)
    psi_data = np.loadtxt(psi_path)

    nodes, num_crit = read_input(seed_path)
    t = s_data[:, 0]
    s = s_data[:, 1:]
    prob = psi_data[:, 1:]

    if mode == "single":
        t_idx = args.index
    else:
        t_idx = args.time_index

    if t_idx < 0:
        t_idx = len(t) + t_idx
    if t_idx < 0 or t_idx >= len(t):
        raise IndexError(f"Time index {t_idx} out of range. Valid range is 0 to {len(t) - 1}.")

    xco = np.linspace(-np.pi * num_crit, np.pi * num_crit, nodes)

    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(r"$\Gamma_2$ t=" + str("%.2e" % t[t_idx]), fontsize=12)

    ax1 = plt.subplot(211)
    ax1.plot(xco, s[t_idx, :])
    ax1.set_xlabel(r"$q_c x$", fontsize=16)
    ax1.set_ylabel("Light intensity (s)", fontsize=16)

    ax2 = plt.subplot(212)
    ax2.plot(xco, prob[t_idx, :])
    ax2.set_xlabel(r"$q_c x$", fontsize=16)
    ax2.set_ylabel(r"BEC ($|\Psi|^2)$", fontsize=16)

    plt.tight_layout()

    if mode == "single":
        image_path = os.path.join(output_dir, f"snapshot_t{t_idx:04d}.png")
    else:
        image_path = os.path.join(output_dir, f"snapshot_i{args.index:04d}_t{t_idx:04d}.png")
    fig.savefig(image_path, dpi=200)
    plt.close(fig)
    print(f"Saved snapshot: {image_path}")


if __name__ == "__main__":
    main()
