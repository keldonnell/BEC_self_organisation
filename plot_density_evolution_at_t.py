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
        description="Plot the amplitude of psi data through space at a specific time."
    )
    parser.add_argument(
        "-f", "--filename", required=True, help="The name of the file to save to"
    )
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        required=True,
        help="The time to inspect the amplitude evolution at",
    )
    parser.add_argument(
        "-i", "--index", type=int, required=False, help="The index of the frame to plot"
    )
    return parser.parse_args()


def read_input(seed_dir):
    """Read input data from seed file."""
    data = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    return (data[0].astype(int), *data[1:13])


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

    # Read input parameters
    nodes, maxt, _, _, _, _, _, _, num_crit, _, _, _, plotnum = read_input(seed_dir)

    # Load data for the specified index
    psi_data, s_data = load_data(output_dir, args.index)

    # Calculate x values
    x_vals = np.linspace(-np.pi * num_crit, np.pi * num_crit, nodes)

    # Extract psi values (excluding the time value if present)
    t_index = int((float(args.time) / maxt) * plotnum)
    psi_vals = psi_data[t_index, 1:]

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Evolution of $|\\psi|^2$ at t = {args.time}")
    ax.set_xlabel(r"$q_c x$", fontsize=14)
    ax.set_ylabel(r"$|\psi|^2$", fontsize=14)
    ax.plot(x_vals, psi_vals)
    plt.show()


if __name__ == "__main__":
    main()
