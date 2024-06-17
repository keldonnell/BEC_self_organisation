# Calculates the fourier transform of the BEC density at a x-pos, through time

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import argparse

# fname = raw_input("Enter filename: ")
plt.rcParams["ps.usedistiller"] = (
    "xpdf"  # improves quality of .eps figures for use with LaTeX
)

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "-f",
    "--filename",
    metavar="filename",
    required=True,
    help="The name of the file to save to",
)

parser.add_argument(
    "-x",
    "--xpos",
    metavar="x_position",
    required=True,
    help="The x co-ord to inpect the fourier transform at",
)

args = parser.parse_args()


output_dir = "patt1d_outputs/" + args.filename + "/"
input_dir = "patt1d_inputs/" + args.filename + "/"
s_dir = output_dir + "s.out"
psi_dir = output_dir + "psi.out"
seed_dir = input_dir + "seed.in"

data1 = np.loadtxt(psi_dir)  # load dataset in the form t, amplitude


# Read input data from file
def readinput():
    data0 = np.genfromtxt(seed_dir, skip_footer=1, comments="!")  # load input data file

    nodes = data0[0].astype(int)
    maxt = data0[1]
    ht = data0[2]
    width_psi = data0[3]
    p0 = data0[4]
    Delta = data0[5]
    gambar = data0[6]
    b0 = data0[7]
    num_crit = data0[8]
    R = data0[9]
    gbar = data0[10]
    v0 = data0[11]
    plotnum = data0[12].astype(int)

    return (
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
    )


nodes, maxt, ht, width_psi, p0, Delta, gambar, b0, num_crit, R, gbar, v0, plotnum = (
    readinput()
)

"""
mid_x_index = int(np.round(len(data1[0, 1:]) / 2))

t_vals = data1[:, 0]
psi_vals_at_0 = data1[:, mid_x_index]        #All the psi values along x=0
"""

x = float(args.xpos)
x_index = int(
    (np.abs(x + np.pi * num_crit) / (2 * np.pi * num_crit)) * nodes
)
t_vals = np.linspace(0, maxt, plotnum)
psi_vals_at_x = data1[:, x_index]

fft_psi_vals = np.fft.fft(psi_vals_at_x)
omega_vals = np.fft.fftfreq(len(t_vals), np.diff(t_vals)[0])


# Plotting the graph
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title(r"Fourier transform of $|\psi|^2$ at x = " + str(x))
ax.set_xlabel(r"$\Gamma \omega$", fontsize=14)
ax.set_ylabel(r"$\mathcal{F}[|\psi(t)|^2]$", fontsize=14)
ax.plot(omega_vals, np.abs(fft_psi_vals))
# ax.plot(k_vals[:nodes//2], np.abs(fft_psi_vals[:nodes//2]))
plt.show()
