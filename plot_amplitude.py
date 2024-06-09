#Plots the amplitude of the psi data along one of the stable (not changing in time) amplitude lines

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import time
import subprocess  # For issuing commands to the OS.
import os
import sys  # For determining the Python version.

# fname = raw_input("Enter filename: ")
plt.rcParams["ps.usedistiller"] = (
    "xpdf"  # improves quality of .eps figures for use with LaTeX
)

fname1 = "psi.out"

data1 = np.loadtxt(fname1)  # load dataset in the form t, amplitude


# Read input data from file
def readinput():
    fname0 = "patt1d_q_sfm.in"
    data0 = np.genfromtxt(fname0, skip_footer=1, comments="!")  # load input data file

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

mid_x_index = int(np.round(len(data1[0, 1:]) / 2))

t_vals = data1[:, 0]
psi_vals_at_0 = data1[mid_x_index, 1:]        #All the psi values along x=0


#Plotting the graph
fig, ax = plt.subplots(figsize=(6,6))
ax.set_title(r"Evolution of $\psi$ at x = 0")
ax.set_xlabel(r'$\Gamma t$',fontsize=14)
ax.set_ylabel(r'$\psi$',fontsize=14)
ax.plot(t_vals, psi_vals_at_0)
plt.show()


