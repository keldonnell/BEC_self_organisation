
# Calculates the energy from the fourier transform of the BEC density at a specific x coord 


# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy import integrate


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

parser.add_argument(
    "-s",
    "--start_f",
    metavar="starting_frequency_position",
    required=True,
    help="The starting frequency limit of integration",
)

parser.add_argument(
    "-e",
    "--end_f",
    metavar="ending_frequency_position",
    required=True,
    help="The ending frequency limit of integration",
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


x = float(args.xpos)
x_index = int(
    (np.abs(x + np.pi * num_crit) / (2 * np.pi * num_crit)) * nodes
)
t_vals = np.linspace(0, maxt, plotnum)
psi_vals_at_x = data1[:, x_index]

fft_psi_vals = np.fft.fft(psi_vals_at_x)
freq_vals = np.fft.fftfreq(len(t_vals), np.diff(t_vals)[0])

#Gets the first positive half of the values as the -ve half is obtained by x_n = x_(-n)* (so for real values it should be mirrored)
N = len(fft_psi_vals)
fft_psi_vals_plus = fft_psi_vals[:N//2]
freq_vals_plus = freq_vals[:N//2]

f_min = np.min(freq_vals)
f_max = np.max(freq_vals)

f_start_index = int((np.abs(float(args.start_f)) / f_max) * len(freq_vals_plus))
f_end_index = int((np.abs(float(args.end_f)) / f_max) * len(freq_vals_plus))
print(f_start_index)
print(f_end_index)
print(freq_vals_plus)

#THIS BIT IS INCORRECT \/ as the freq_vals list is not ordered as expected, so the indexing ..:.. doesnt work correctly. See the freq_vals array in print.
freq_vals_to_integrate = np.abs(freq_vals_plus[f_start_index:f_end_index])
fft_psi_vals_to_integrate = np.abs(fft_psi_vals_plus[f_start_index:f_end_index])
#print(freq_vals_to_integrate)
#print(fft_psi_vals_to_integrate)
energy = integrate.simps(fft_psi_vals_to_integrate, freq_vals_to_integrate)

print(energy)


