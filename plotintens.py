# Plots output from code PART1D_Q_SFM_FFT.F90
# Shows image of optical intensity and BE density vs x and t

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
fname2 = "s.out"

data1 = np.loadtxt(fname1)  # load dataset in the form t, intensity
data2 = np.loadtxt(fname2)  # load dataset in the form t, intensity


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


# Nx=np.sqrt(np.size(data1,axis=1)-1).astype(int)                              #No. of points in each row for field and BEC

# print 'Nx =',Nx
# print 'Size of data1 =',np.size(data1,axis=0),np.size(data1,axis=1)
nodes, maxt, ht, width_psi, p0, Delta, gambar, b0, num_crit, R, gbar, v0, plotnum = (
    readinput()
)

tvec = data1[:, 0]
# plotnum=len(tvec)
psi = data1[:, 1:]
s = data2[:, 1:]


pi = 4.0 * np.arctan(1.0)
print(pi)
xco = np.linspace(-pi * num_crit, pi * num_crit, nodes)

# xmat,tmat=np.meshgrid(xco,tvec)


fig = plt.figure()

ax1 = plt.subplot(121, aspect="auto")
f1 = ax1.imshow(
    psi,
    extent = [-pi * num_crit, pi * num_crit, 0, tvec.max()],
    origin = "lower",
    vmin = psi.min(),
    vmax = psi.max(),
    aspect = "auto",
    cmap = "hot",
)
cb1 = fig.colorbar(f1, orientation="horizontal")
ax1.set_xlabel(r"$q_c x$", fontsize=14)
ax1.set_ylabel(r"$\Gamma t$", fontsize=14)
ax1.set_title(r"BEC density $|\Psi|^2$", fontsize=14)


ax2 = plt.subplot(122, aspect="auto")
f2 = ax2.imshow(
    s,
    extent = [-pi * num_crit, pi * num_crit, 0, tvec.max()],
    origin = "lower",
    vmin = s.min(),
    vmax = s.max(),
    aspect = "auto",
    cmap = "hot",
)
cb1 = fig.colorbar(f2, orientation="horizontal")
ax2.set_xlabel(r"$q_c x$", fontsize=14)
ax2.set_ylabel(r"$\Gamma t$", fontsize=14)
ax2.set_title("Intensity (s)", fontsize=14)

plt.show()
