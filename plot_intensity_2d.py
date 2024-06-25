import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import os

def readinput(seed_dir):
    data0 = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    nodes_x, nodes_y = data0[0].astype(int), data0[1].astype(int)
    maxt = data0[2]
    ht = data0[3]
    width_psi_x, width_psi_y = data0[4], data0[5]
    p0 = data0[6]
    Delta = data0[7]
    omega_r = data0[8]
    b0 = data0[9]
    num_crit = data0[10]
    R = data0[11]
    gbar = data0[12]
    v0_x, v0_y = data0[13], data0[14]
    plotnum = data0[15].astype(int)
    noise_val = data0[16]
    
    return (nodes_x, nodes_y, maxt, ht, width_psi_x, width_psi_y, p0, Delta, omega_r, b0, num_crit, R, gbar, v0_x, v0_y, plotnum, noise_val)

def plot_2d_data(filename):
    output_dir = f"patt2d_outputs/{filename}/"
    input_dir = f"patt2d_inputs/{filename}/"
    seed_dir = os.path.join(input_dir, "seed.in")

    # Read input parameters
    nodes_x, nodes_y, maxt, ht, width_psi_x, width_psi_y, p0, Delta, omega_r, b0, num_crit, R, gbar, v0_x, v0_y, plotnum, noise_val = readinput(seed_dir)

    # Get all output files
    psi_files = sorted(glob.glob(os.path.join(output_dir, "psi_*.npy")))
    s_files = sorted(glob.glob(os.path.join(output_dir, "s_*.npy")))

    # Create x and y coordinates
    L_dom_x, L_dom_y = 2.0 * np.pi * num_crit, 2.0 * np.pi * num_crit
    x = np.linspace(-L_dom_x/2, L_dom_x/2, nodes_x)
    y = np.linspace(-L_dom_y/2, L_dom_y/2, nodes_y)
    X, Y = np.meshgrid(x, y)

    # Plot for each time step
    for i, (psi_file, s_file) in enumerate(zip(psi_files, s_files)):
        t = float(psi_file.split('_')[-1].split('.')[0])
        
        psi = np.load(psi_file)
        s = np.load(s_file)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot BEC density
        im1 = ax1.imshow(psi, extent=[-L_dom_x/2, L_dom_x/2, -L_dom_y/2, L_dom_y/2], 
                         origin='lower', aspect='equal', cmap='viridis')
        ax1.set_title(r'BEC density $|\Psi|^2$')
        ax1.set_xlabel(r'$q_c x$')
        ax1.set_ylabel(r'$q_c y$')
        plt.colorbar(im1, ax=ax1)

        # Plot optical intensity
        im2 = ax2.imshow(s, extent=[-L_dom_x/2, L_dom_x/2, -L_dom_y/2, L_dom_y/2], 
                         origin='lower', aspect='equal', cmap='plasma')
        ax2.set_title('Optical Intensity (s)')
        ax2.set_xlabel(r'$q_c x$')
        ax2.set_ylabel(r'$q_c y$')
        plt.colorbar(im2, ax=ax2)

        plt.suptitle(f'Time: {t:.2f}')
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(output_dir, f'plot_{i:04d}.png'), dpi=300)
        plt.close()

    print(f"Plots saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 2D BEC density and optical intensity")
    parser.add_argument("-f", "--filename", required=True, help="The name of the file to process")
    args = parser.parse_args()

    plot_2d_data(args.filename)