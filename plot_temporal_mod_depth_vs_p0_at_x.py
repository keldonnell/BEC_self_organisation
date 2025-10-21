import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

import standard_data_utils as stand_utils


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot temporal modulation depth versus p0.")
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
    """Read simulation metadata from the seed file."""
    data0 = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    return (
        data0[0].astype(int),  # nodes
        *data0[1:12],
        data0[12].astype(int),  # plotnum
    )


def extract_float(filename):
    """Extract the floating point value from a psi filename."""
    match = re.search(r"psi\d+_([\d.e-]+)", filename)
    if match:
        float_str = match.group(1).rstrip(".")
        return float(float_str)
    return 0.0


def load_data(output_dir):
    """Load and sort psi files from the output directory."""
    data_files = glob.glob(os.path.join(output_dir, "psi*"))
    return sorted(data_files, key=extract_float)


def calc_mod_depth_vs_p0(sorted_files, nodes, x_index):
    """Compute modulation depth for every psi file and pair it with its p0 value."""
    if not sorted_files:
        raise FileNotFoundError("No psi files found in the specified output directory.")

    mod_depth_vals = stand_utils.calc_modulation_depth(sorted_files, nodes, True, x_index)
    p0_vals = stand_utils.find_p0_vals_from_filenames(sorted_files)

    return np.array(p0_vals), np.array(mod_depth_vals) / 100.0


def create_plot(p0_vals, mod_depth_vals, x, p0_samples, m0_vals, p_th):
    """Create the modulation depth vs p0 scatter plot."""
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        p0_vals,
        mod_depth_vals,
        color="C0",
        edgecolor="black",
        linewidth=0.6,
        s=30,
        label="Simulation",
        zorder=3,
    )

    if m0_vals.size:
        ax.plot(
            p0_samples,
            m0_vals,
            color="C1",
            linewidth=2.2,
            label=r"$M_0$ analytic",
            zorder=4,
        )

    ax.axvline(p_th, color="k", linestyle="--", linewidth=1.2, label=r"$p_{th}$", zorder=2)

    ax.set_xlabel(r"Pump strength $p_0$", fontsize=13)
    ax.set_ylabel(r"Modulation depth $m[|\psi|^2]$", fontsize=13)
    ax.set_title(rf"Temporal modulation depth at $x = {x}$", fontsize=14)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.grid(which="major", linestyle=":", alpha=0.4)
    ax.grid(which="minor", linestyle=":", alpha=0.15)
    ax.legend(frameon=False, loc="lower right")

    if p0_vals.size:
        ax.set_xlim(p0_vals.min() * 0.98, p0_vals.max() * 1.02)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig, ax


def main():
    """Main entry point."""
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
    ) = read_input(seed_dir)

    x = float(args.xpos)
    frac = (np.abs(x + np.pi * num_crit) / (2 * np.pi * num_crit)) * nodes
    x_index = int(np.clip(frac, 0, nodes - 1))

    sorted_files = load_data(output_dir)
    p0_vals, mod_depth_vals = calc_mod_depth_vs_p0(sorted_files, nodes, x_index)

    p_th = (2 * gambar) / (b0 * R)

    if p0_vals.size:
        p0_min, p0_max = p0_vals.min(), p0_vals.max()
        if p0_min == p0_max:
            p0_samples = np.array([p0_min])
        else:
            p0_samples = np.linspace(p0_min, p0_max, max(len(p0_vals), 200))
        with np.errstate(divide="ignore", invalid="ignore"):
            m0_vals = np.sqrt(2 * p_th * np.maximum(p0_samples - p_th, 0) / (p0_samples**2))
            m0_vals = np.where(p0_samples > 0, m0_vals, np.nan)
    else:
        p0_samples = np.array([])
        m0_vals = np.array([])

    create_plot(p0_vals, mod_depth_vals, x, p0_samples, m0_vals, p_th)
    plt.show()


if __name__ == "__main__":
    main()
