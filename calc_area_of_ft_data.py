# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy import integrate
import glob

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Analyze Fourier transforms of patt1d outputs.")
    parser.add_argument("-f", "--filename", metavar="filename", required=True, help="The name of the file to save to")
    parser.add_argument("-x", "--xpos", metavar="x_position", required=True, help="The x co-ord to inspect the Fourier transform at")
    parser.add_argument("-s", "--start_f", metavar="starting_frequency_position", required=True, help="The starting frequency limit of integration")
    parser.add_argument("-e", "--end_f", metavar="ending_frequency_position", required=True, help="The ending frequency limit of integration")
    parser.add_argument("-i", "--frame_index", metavar="frame_index", required=False, help="The index of the frame to plot")
    return parser.parse_args()

def read_input(seed_dir):
    """
    Read input data from the seed file.
    
    Args:
        seed_dir (str): Path to the seed file
    
    Returns:
        tuple: Various parameters read from the seed file
    """
    data0 = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    return (
        data0[0].astype(int),  # nodes
        *data0[1:12],
        data0[12].astype(int)  # plotnum
    )

def load_data(output_dir, frame_index=None):
    """
    Load data files from the output directory.
    
    Args:
        output_dir (str): Path to the output directory
        frame_index (str, optional): Index of the frame to load
    
    Returns:
        tuple: Loaded psi and s data
    """
    if frame_index is None:
        psi_files = glob.glob(output_dir + "psi*")
        s_files = glob.glob(output_dir + "s*")
        if len(psi_files) > 1 or len(s_files) > 1:
            raise ValueError("You must specify a frame index as there is more than one file")
        data1 = np.loadtxt(psi_files[0])
        data2 = np.loadtxt(s_files[0])
    else:
        psi_file = glob.glob(output_dir + f"psi{frame_index}_*")[0]
        s_file = glob.glob(output_dir + f"s{frame_index}_*")[0]
        print(f"Loading psi file: {psi_file}")
        print(f"Loading s file: {s_file}")
        data1 = np.loadtxt(psi_file)
        data2 = np.loadtxt(s_file)
    return data1, data2

def calculate_fft(t_vals, psi_vals):
    """
    Calculate the FFT of the psi values.
    
    Args:
        t_vals (np.array): Time values
        psi_vals (np.array): Psi values
    
    Returns:
        tuple: FFT values and corresponding frequency values
    """
    fft_psi_vals = np.fft.fft(psi_vals)
    freq_vals = np.fft.fftfreq(len(t_vals), np.diff(t_vals)[0])
    
    # Get the first positive half of the values
    N = len(fft_psi_vals)
    fft_psi_vals_plus = fft_psi_vals[:N//2]
    freq_vals_plus = freq_vals[:N//2]
    
    return fft_psi_vals_plus, freq_vals_plus

def integrate_fft(freq_vals, fft_vals, f_start, f_end):
    """
    Integrate the FFT values within the specified frequency range.
    
    Args:
        freq_vals (np.array): Frequency values
        fft_vals (np.array): FFT values
        f_start (float): Start frequency for integration
        f_end (float): End frequency for integration
    
    Returns:
        float: Integrated area
    """
    f_max = np.max(freq_vals)
    f_start_index = int((np.abs(f_start) / f_max) * len(freq_vals))
    f_end_index = int((np.abs(f_end) / f_max) * len(freq_vals))
    
    freq_vals_to_integrate = np.abs(freq_vals[f_start_index:f_end_index])
    fft_vals_to_integrate = np.abs(fft_vals[f_start_index:f_end_index])
    
    return integrate.simps(fft_vals_to_integrate, freq_vals_to_integrate)

def main():
    """
    Main function to run the script.
    """
    # Set matplotlib parameters
    plt.rcParams["ps.usedistiller"] = "xpdf"
    
    # Parse command-line arguments
    args = parse_arguments()

    # Define directories
    output_dir = f"patt1d_outputs/{args.filename}/"
    input_dir = f"patt1d_inputs/{args.filename}/"
    seed_dir = f"{input_dir}seed.in"

    # Read input data
    nodes, maxt, ht, width_psi, p0, Delta, gambar, b0, num_crit, R, gbar, v0, plotnum = read_input(seed_dir)

    # Load data
    data1, data2 = load_data(output_dir, args.frame_index)

    # Calculate x index
    x = float(args.xpos)
    x_index = int((np.abs(x + np.pi * num_crit) / (2 * np.pi * num_crit)) * nodes)

    # Calculate time values and psi values at the specified x
    t_vals = np.linspace(0, maxt, plotnum)
    psi_vals_at_x = data1[:, x_index]

    # Calculate FFT
    fft_psi_vals_plus, freq_vals_plus = calculate_fft(t_vals, psi_vals_at_x)

    # Integrate FFT
    area = integrate_fft(freq_vals_plus, fft_psi_vals_plus, float(args.start_f), float(args.end_f))

    print(f"Integrated area: {area}")

if __name__ == "__main__":
    main()