import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob

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
    """Plot density and intensity heatmaps over time."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.3)
    for axis in ax:
        axis.set_box_aspect(1)

    extent = [-np.pi * num_crit, np.pi * num_crit, 0, t_vals.max()]

    # PSI heatmap
    f1 = ax[0].imshow(
        psi_vals, extent=extent, origin="lower", aspect="auto", cmap="hot"
    )
    fig.colorbar(f1, ax=ax[0], orientation="horizontal")
    ax[0].set_xlabel(r"$q_c x$", fontsize=14)
    ax[0].set_ylabel(r"$\Gamma t$", fontsize=14)
    ax[0].set_title(r"BEC density $|\Psi|^2$", fontsize=14)

    # S heatmap
    f2 = ax[1].imshow(
        s_vals, extent=extent, origin="lower", aspect="auto", cmap="hot"
    )
    fig.colorbar(f2, ax=ax[1], orientation="horizontal")
    ax[1].set_xlabel(r"$q_c x$", fontsize=14)
    ax[1].set_ylabel(r"$\Gamma t$", fontsize=14)
    ax[1].set_title("Intensity (s)", fontsize=14)


def main():
    args = parse_arguments()

    output_dir = f"patt1d_outputs/{args.filename}/"
    input_dir = f"patt1d_inputs/{args.filename}/"
    seed_dir = input_dir + "seed.in"

    # Load data
    psi_data, s_data = load_data(output_dir, args.frame_index)

    # Read input parameters
    params = read_input(seed_dir)
    num_crit = params[8]

    # Extract data
    psi_vals = psi_data[:, 1:]
    s_vals = s_data[:, 1:]
    t_vals = psi_data[:, 0]
    # Create plots
    plot_psi_s_heatmaps(psi_vals, s_vals, t_vals, num_crit)

    plt.show()


if __name__ == "__main__":
    main()
