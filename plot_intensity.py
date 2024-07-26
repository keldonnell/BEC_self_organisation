import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import standard_data_utils as stand_utils

# Set matplotlib parameters for better quality output
plt.rcParams["ps.usedistiller"] = (
    "xpdf"  # Improves quality of .eps figures for use with LaTeX
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot PSI and S data from simulation outputs."
    )
    parser.add_argument(
        "-f", "--filename", required=True, help="The name of the file to save to"
    )
    parser.add_argument(
        "-i", "--frame_index", type=int, help="The index of the frame to plot"
    )
    return parser.parse_args()


def read_input(seed_dir):
    """Read input data from seed file."""
    data = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    return tuple(data[:13])


def load_data(output_dir, frame_index):
    """Load PSI and S data from files."""
    psi_files = glob.glob(output_dir + "psi*")
    s_files = glob.glob(output_dir + "s*")

    if len(psi_files) > 1 and frame_index is None:
        raise ValueError(
            "You must specify a frame index as there is more than one file"
        )

    if len(psi_files) == 1:
        psi_data = np.loadtxt(psi_files[0])
        s_data = np.loadtxt(s_files[0])
    else:
        psi_data = np.loadtxt(glob.glob(output_dir + f"psi{frame_index}_*")[0])
        s_data = np.loadtxt(glob.glob(output_dir + f"s{frame_index}_*")[0])
        print(f"PSI file: {glob.glob(output_dir + f'psi{frame_index}_*')[0]}")
        print(f"S file: {glob.glob(output_dir + f's{frame_index}_*')[0]}")

    return psi_data, s_data


def plot_psi_s_heatmaps(psi_vals, s_vals, t_vals, num_crit):
    """Plot heatmaps for PSI and S values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.3)

    extent = [-np.pi * num_crit, np.pi * num_crit, 0, t_vals.max()]

    # PSI heatmap
    f1 = ax1.imshow(psi_vals, extent=extent, origin="lower", aspect="auto", cmap="hot")
    fig.colorbar(f1, ax=ax1, orientation="horizontal")
    ax1.set_xlabel(r"$q_c x$", fontsize=14)
    ax1.set_ylabel(r"$\Gamma t$", fontsize=14)
    ax1.set_title(r"BEC density $|\Psi|^2$", fontsize=14)

    # S heatmap
    f2 = ax2.imshow(s_vals, extent=extent, origin="lower", aspect="auto", cmap="hot")
    fig.colorbar(f2, ax=ax2, orientation="horizontal")
    ax2.set_xlabel(r"$q_c x$", fontsize=14)
    ax2.set_ylabel(r"$\Gamma t$", fontsize=14)
    ax2.set_title("Intensity (s)", fontsize=14)


def plot_temporal_cut(t_vals, psi_t_cut):
    """Plot temporal cut along maximum starting at x = 0."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(r"Temporal cut along maximum starting at x = 0")
    ax.set_xlabel(r"$\Gamma t$", fontsize=14)
    ax.set_ylabel(r"${|\psi|}^2$", fontsize=14)
    ax.plot(t_vals, psi_t_cut)


def main():
    args = parse_arguments()

    output_dir = f"patt1d_outputs/{args.filename}/"
    input_dir = f"patt1d_inputs/{args.filename}/"
    seed_dir = input_dir + "seed.in"

    # Load data
    psi_data, s_data = load_data(output_dir, args.frame_index)

    # Read input parameters
    params = read_input(seed_dir)
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
    ) = params

    # Extract data
    psi_vals = psi_data[:, 1:]
    s_vals = s_data[:, 1:]
    t_vals = psi_data[:, 0]
    psi_t_cut = stand_utils.find_temporal_cut_of_x_peaks(psi_vals, 0)

    # Create plots
    plot_psi_s_heatmaps(psi_vals, s_vals, t_vals, num_crit)
    plot_temporal_cut(t_vals, psi_t_cut)

    plt.show()


if __name__ == "__main__":
    main()
