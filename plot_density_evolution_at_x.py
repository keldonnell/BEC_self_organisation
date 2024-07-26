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
    parser = argparse.ArgumentParser(description="Analyze and plot psi data.")
    parser.add_argument(
        "-f", "--filename", required=True, help="The name of the file to save to"
    )
    parser.add_argument(
        "-x",
        "--xpos",
        type=float,
        required=True,
        help="The x-position coordinate to inspect the amplitude evolution at",
    )
    parser.add_argument(
        "-i", "--frame_index", type=int, help="The index of the frame to plot"
    )
    parser.add_argument(
        "-t",
        "--trail_x_max",
        required=True,
        choices=["True", "False"],
        help="Specifies whether to follow the peaks that start at the required x arg",
    )
    return parser.parse_args()


def read_input(seed_dir):
    """Read input data from file."""
    data = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    return tuple(data[:13])  # Return the first 13 elements as a tuple


def load_data(output_dir, frame_index):
    """Load psi and s data from files."""
    if frame_index is None:
        psi_files = glob.glob(output_dir + "psi*")
        s_files = glob.glob(output_dir + "s*")
        if len(psi_files) > 1 or len(s_files) > 1:
            raise Exception(
                "You must specify a frame index as there is more than one file"
            )
        psi_data = np.loadtxt(psi_files[0])
        s_data = np.loadtxt(s_files[0])
    else:
        psi_data = np.loadtxt(glob.glob(output_dir + f"psi{frame_index}_*")[0])
        s_data = np.loadtxt(glob.glob(output_dir + f"s{frame_index}_*")[0])
    return psi_data, s_data


def main():
    args = parse_arguments()

    output_dir = f"patt1d_outputs/{args.filename}/"
    input_dir = f"patt1d_inputs/{args.filename}/"
    seed_dir = input_dir + "seed.in"

    # Load data
    psi_data, s_data = load_data(output_dir, args.frame_index)

    # Read input parameters
    params = read_input(seed_dir)
    nodes, num_crit = params[0], params[8]

    # Calculate x_index
    x_index = int(
        (np.abs(args.xpos + np.pi * num_crit) / (2 * np.pi * num_crit)) * nodes
    )

    # Get psi values
    if args.trail_x_max.upper() == "TRUE":
        psi_cut_vals = stand_utils.find_temporal_cut_of_x_peaks(psi_data, x_index)
    elif args.trail_x_max.upper() == "FALSE":
        psi_cut_vals = psi_data[:, x_index]
    else:
        raise ValueError(
            "Invalid input for -t. The input should be either 'True' or 'False'"
        )

    t_vals = psi_data[:, 0]

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(r"Evolution of $|\psi|^2$ at x = {}".format(args.xpos))
    ax.set_xlabel(r"$\Gamma t$", fontsize=14)
    ax.set_ylabel(r"$|\psi|^2$", fontsize=14)
    ax.plot(t_vals, psi_cut_vals)
    plt.show()


if __name__ == "__main__":
    main()
