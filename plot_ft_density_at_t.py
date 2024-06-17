# Calculates the fourier transform of the BEC density at a specific time

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
    "-t",
    "--time",
    metavar="time",
    required=True,
    help="The time co-ord to inpect the fourier transform at",
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

# t = 0.8e7
t = float(args.time)
t_index = int((t / maxt) * plotnum)
x_vals = np.linspace(-np.pi * num_crit, np.pi * num_crit, nodes)
psi_vals_at_t = data1[t_index, 1:]

fft_psi_vals = np.fft.fft(psi_vals_at_t)
k_vals = np.fft.fftfreq(len(x_vals), np.diff(x_vals)[0])


# This code can definitely be made more clean/simple
# Finds the start and end index of the 1st fourier mode
'''
mode_1st_index = []
for i in range(len(fft_psi_vals) // 2, len(fft_psi_vals)):
    counter = 0 
    print(fft_psi_vals[i-1])
    if (fft_psi_vals[i-1] == 0 and fft_psi_vals[i] != 0):
        if counter == 1 or counter == 2:
            mode_1st_index.append[i-1]
        counter += 1

print(len(mode_1st_index))
integral_1st_modes = 2 * np.trapz(fft_psi_vals[mode_1st_index[0]:mode_1st_index[1]], k_vals[mode_1st_index[0]:mode_1st_index[1]])

print("Integral 1st modes: " + integral_1st_modes)
'''

# Plotting the graph
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title(r"Fourier transform of $|\psi|^2$ at \Gamma t = " + str(t))
ax.set_xlabel(r"$\Gamma k$", fontsize=14)
ax.set_ylabel(r"$\mathcal{F}[|\psi|^2]$", fontsize=14)
ax.scatter(k_vals, np.abs(fft_psi_vals))
# ax.plot(k_vals[:nodes//2], np.abs(fft_psi_vals[:nodes//2]))
plt.show()
