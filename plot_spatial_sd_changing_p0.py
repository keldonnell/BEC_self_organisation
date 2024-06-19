# Plots the spatial standard deviation for different pump parameters at a specific time/peak of the density (|psi|^2) data

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import re

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

t = float(args.time)
t_index = int((t / maxt) * plotnum)

data1_files = glob.glob(output_dir + "psi*")
sd_vals = [
    np.sqrt(
        np.mean(
            (
                np.loadtxt(data1_files[i])[t_index, 1:]
                - np.mean(np.loadtxt(data1_files[i])[t_index, 1:])
            )
            ** 2
        )
    )
    for i in range(len(data1_files))
]
rms_vals = [
    np.sqrt(np.mean((np.loadtxt(data1_files[i])[t_index, 1:]) ** 2))
    for i in range(len(data1_files))
]
span_vals = [
    np.abs(
        (
            np.max(np.loadtxt(data1_files[i])[t_index, 1:])
            - np.min(np.loadtxt(data1_files[i])[t_index, 1:])
        )
    )
    for i in range(len(data1_files))
]

p0_vals = np.array(
    [
        float(re.findall(r"(\d+\.\d+e[-+]\d+)", data1_files[j])[0])
        for j in range(len(data1_files))
    ]
)
print(p0_vals)

# Plotting the graph
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].set_title(r"Standard deviation of $|\psi|^2$ at different pump parameters")
ax[0].set_xlabel(r"$p_0$", fontsize=14)
ax[0].set_ylabel(r"$\sigma[|\psi|^2]$", fontsize=14)
ax[0].scatter(p0_vals, sd_vals)

ax[1].set_title(r"RMS of $|\psi|^2$ at different pump parameters")
ax[1].set_xlabel(r"$p_0$", fontsize=14)
ax[1].set_ylabel(r"RMS $[|\psi|^2]$", fontsize=14)
ax[1].scatter(p0_vals, rms_vals)

ax[2].set_title(r"Span of $|\psi|^2$ at different pump parameters")
ax[2].set_xlabel(r"$p_0$", fontsize=14)
ax[2].set_ylabel(r"Span $[|\psi|^2]$", fontsize=14)
ax[2].scatter(p0_vals, span_vals)
plt.show()
