
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy import integrate
import argparse
import glob
import standard_data_utils as stand_utils


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
    help="The time to inpect the amplitude evolution at",
)

parser.add_argument(
    "-i",
    "--frame_index",
    metavar="frame_index",
    required=False,
    help="The index of the frame to plot",
)

args = parser.parse_args()

output_dir = "patt1d_outputs/" + args.filename + "/"
input_dir = "patt1d_inputs/" + args.filename + "/"
s_dir = output_dir + "s.out"
psi_dir = output_dir + "psi.out"
seed_dir = input_dir + "seed.in"


if ((len(glob.glob(output_dir + "psi*")) > 1 or len(glob.glob(output_dir + "psi*")) > 1) and args.frame_index == None):
    raise Exception("You must specify a frame index as there is more than one file")

if (len(glob.glob(output_dir + "psi*")) == 1):
    data1 = np.loadtxt(glob.glob(output_dir + f"psi*")[0])
    data2 = np.loadtxt(glob.glob(output_dir + f"s*")[0])
else:
    data1 = np.loadtxt(glob.glob(output_dir + f"psi{args.frame_index}_*")[0])
    data2 = np.loadtxt(glob.glob(output_dir + f"s{args.frame_index}_*")[0])
    print(glob.glob(output_dir + f"psi{args.frame_index}_*")[0])
    print(glob.glob(output_dir + f"s{args.frame_index}_*")[0])




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

data1 = np.loadtxt(psi_dir)  # load dataset in the form t, amplitude

t = float(args.time)
t_index = int((t / maxt) * plotnum)
x_vals = np.linspace(-np.pi * num_crit, np.pi * num_crit, nodes)

Q_0 = stand_utils.calc_legett(data1, x_vals, nodes, t_index)

print(f"The legget criteria Q0 = {Q_0}")



