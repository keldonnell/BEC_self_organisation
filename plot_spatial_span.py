# Plots the spatial span (Through time) of the density (|psi|^2) data 

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
        plotnum,7
    )


nodes, maxt, ht, width_psi, p0, Delta, gambar, b0, num_crit, R, gbar, v0, plotnum = (
    readinput()
)


t_vals = data1[:, 0]
span_vals = [np.abs(np.max(data1[i, 1:]) - np.min(data1[i, 1:])) for i in range(plotnum)]

# Plotting the graph
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title(r"Span of $|\psi|^2$")
ax.set_xlabel(r"$\Gamma t$", fontsize=14)
ax.set_ylabel(r"$Span[|\psi|^2]$", fontsize=14)
ax.plot(t_vals, span_vals)
plt.show()
