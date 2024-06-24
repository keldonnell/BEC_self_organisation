import matplotlib.pyplot as plt
import numpy as np
import argparse

plt.rcParams["ps.usedistiller"] = "xpdf"

parser = argparse.ArgumentParser(description="Plot intensity and BEC density from 2D simulation outputs at a specified time.")
parser.add_argument("-f", "--filename", metavar="filename", required=True, help="The name of the file to plot")
parser.add_argument("-t", "--time", metavar="time", type=float, required=False, help="The time at which to display the plot")

args = parser.parse_args()

output_dir = f"patt2d_outputs/{args.filename}/"
input_dir = f"patt2d_inputs/{args.filename}/"
seed_dir = f"{input_dir}seed.in"

def read_input():
    data0 = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    return {
        'nodes_x': int(data0[0]),
        'nodes_y': int(data0[1]),
        'maxt': data0[2],
        'ht': data0[3],
        'num_crit': data0[9],
        'plotnum': int(data0[14])
    }

input_params = read_input()

psi_data = np.load(f"{output_dir}psi.npy")
s_data = np.load(f"{output_dir}s.npy")

nodes_x, nodes_y = input_params['nodes_x'], input_params['nodes_y']
x = np.linspace(-np.pi * input_params['num_crit'], np.pi * input_params['num_crit'], nodes_x)
y = np.linspace(-np.pi * input_params['num_crit'], np.pi * input_params['num_crit'], nodes_y)
X, Y = np.meshgrid(x, y)

time_steps = int(input_params['maxt'] / input_params['ht'])
times = np.linspace(0, input_params['maxt'], time_steps)

if args.time is None:
    time_index = 0  # Default to the first time step if no time is specified
else:
    time_index = np.argmin(np.abs(times - args.time))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

print(np.shape(psi_data))

im1 = ax1.imshow(psi_data[time_index], extent=[-np.pi * input_params['num_crit'], np.pi * input_params['num_crit'],
                                               -np.pi * input_params['num_crit'], np.pi * input_params['num_crit']],
                 origin='lower', cmap='hot')
im2 = ax2.imshow(s_data[time_index], extent=[-np.pi * input_params['num_crit'], np.pi * input_params['num_crit'],
                                             -np.pi * input_params['num_crit'], np.pi * input_params['num_crit']],
                 origin='lower', cmap='hot')

ax1.set_xlabel(r"$q_c x$", fontsize=14)
ax1.set_ylabel(r"$q_c y$", fontsize=14)
ax2.set_xlabel(r"$q_c x$", fontsize=14)
ax2.set_ylabel(r"$q_c y$", fontsize=14)

ax1.set_title(r"BEC density $|\Psi|^2$ at t = {:.2f}".format(times[time_index]), fontsize=14)
ax2.set_title("Intensity (s) at t = {:.2f}".format(times[time_index]), fontsize=14)

plt.colorbar(im1, ax=ax1, orientation='horizontal')
plt.colorbar(im2, ax=ax2, orientation='horizontal')

plt.tight_layout()
plt.show()