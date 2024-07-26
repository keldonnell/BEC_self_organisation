# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.signal import find_peaks
import glob
import standard_data_utils as stand_utils


# Set matplotlib parameters for better quality output
plt.rcParams["ps.usedistiller"] = (
    "xpdf"  # Improves quality of .eps figures for use with LaTeX
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze PATT1D output data")
    parser.add_argument(
        "-f", "--filename", required=True, help="The name of the file to save to"
    )
    parser.add_argument(
        "-x", "--xpos", metavar="x_position", required=True, help="The x co-ord to inpect the fourier transform at",
    )
    parser.add_argument(
        "-i", "--frame_index", type=int, help="The index of the frame to plot"
    )
    return parser.parse_args()


def load_data(output_dir, frame_index):
    """Load psi and s data from files."""
    psi_files = glob.glob(f"{output_dir}psi*")
    s_files = glob.glob(f"{output_dir}s*")

    if len(psi_files) > 1 and frame_index is None:
        raise ValueError(
            "You must specify a frame index as there is more than one file"
        )

    if frame_index is None:
        psi_data = np.loadtxt(psi_files[0])
        s_data = np.loadtxt(s_files[0])
    else:
        psi_data = np.loadtxt(glob.glob(f"{output_dir}psi{frame_index}_*")[0])
        s_data = np.loadtxt(glob.glob(f"{output_dir}s{frame_index}_*")[0])
        print(f"Loaded files: {glob.glob(f'{output_dir}psi{frame_index}_*')[0]}")
        print(f"              {glob.glob(f'{output_dir}s{frame_index}_*')[0]}")

    return psi_data, s_data


def read_input(seed_dir):
    """Read input data from seed file."""
    data = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    return {
        "nodes": int(data[0]),
        "maxt": data[1],
        "ht": data[2],
        "width_psi": data[3],
        "p0": data[4],
        "Delta": data[5],
        "gambar": data[6],
        "b0": data[7],
        "num_crit": data[8],
        "R": data[9],
        "gbar": data[10],
        "v0": data[11],
        "plotnum": int(data[12]),
    }




def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Set up directory paths
    output_dir = f"patt1d_outputs/{args.filename}/"
    input_dir = f"patt1d_inputs/{args.filename}/"
    seed_dir = f"{input_dir}seed.in"

    # Read input parameters from seed file
    params = read_input(seed_dir)

    # Load data for the specified frame index
    psi_data, s_data = load_data(output_dir, args.frame_index)

    # Calculate x values and time index
    x = float(args.xpos)
    x_index = int((np.abs(x + np.pi * params['num_crit']) / (2 * np.pi * params['num_crit'])) * params['nodes'])
    t_vals = psi_data[:, 0]

    psi_temporal_cut = stand_utils.find_temporal_cut_of_x_peaks(psi_data, x_index)

    # Apply Fourier transform
    fft_psi_vals = np.abs(np.fft.fft(psi_temporal_cut, norm="forward"))
    freq_vals = np.fft.fftfreq(len(t_vals), np.diff(t_vals)[0])

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(r"Fourier transform of $|\psi|^2$ at x = " + str(x))
    ax.set_xlabel(r"$\Gamma \nu$", fontsize=14)
    ax.set_ylabel(r"$\mathcal{F}[|\psi(t)|^2]$", fontsize=14)
    ax.plot(freq_vals, np.abs(fft_psi_vals))
    # ax.plot(k_vals[:nodes//2], np.abs(fft_psi_vals[:nodes//2]))
    plt.show()


if __name__ == "__main__":
    main()
