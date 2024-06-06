# Plots output from code PATT1D_Q_SFM_FFT_S
# Plots intensities and phases of Optical field and BEC wavefunction.
# Generates sequence of .png files

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os

# fname = raw_input("Enter filename: ")
plt.rcParams["ps.usedistiller"] = (
    "xpdf"  # improves quality of .eps figures for use with LaTeX
)

fname1 = "s.out"
fname2 = "psi.out"

print("Loading " + fname1)
data1 = np.loadtxt(fname1)  # load dataset in the form t, intensity
print("Loading " + fname2)
data2 = np.loadtxt(fname2)  # load dataset in the form t, |Psi|^2


framedir = "frames/"  # Folder where .png images are to be located
if not os.path.exists(framedir):  # Create folder if it does not aleady exist
    os.makedirs(framedir)


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
t = data1[:, 0]
plotnum = len(t)
s = data1[:, 1:]
prob = data2[:, 1:]

# plt.ion()
plt.ioff()

pi = 4.0 * np.arctan(1.0)
xco = np.linspace(-pi * num_crit, pi * num_crit, nodes)

fig = plt.figure()
step = 1
count = 0
for j in np.arange(0, plotnum - 1, step):
    count = count + 1
    print("Generating frame " + str(count))

    fig.suptitle(r"$\Gamma_2$ t=" + str("%.2e" % t[j]), fontsize=12)

    ax1 = plt.subplot(211)
    f1 = ax1.plot(xco, s[j, :])
    ax1.set_xlabel(r"$q_c x$", fontsize=16)
    ax1.set_ylabel("Light intensity (s)", fontsize=16)

    ax2 = plt.subplot(212)
    f2 = ax2.plot(xco, prob[j, :])
    ax2.set_xlabel(r"$q_c x$", fontsize=16)
    ax2.set_ylabel(r"BEC ($|\Psi|^2)$", fontsize=16)

    #    plt.tight_layout()

    plt.draw()
    #    plt.show()
    #    time.sleep(0.02)
    filename = framedir + str("%03d" % count) + ".png"
    fig.savefig(filename, dpi=200)
    plt.clf()

# plt.ioff()
