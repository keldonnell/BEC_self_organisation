import numpy as np
import os
import argparse

pump_params = np.linspace(
    1.5e-7,
    4.5e-7,
    200)

print(pump_params[18])
print(pump_params[18 + 19])
print(pump_params[18 + 2*19])
print(pump_params[18 + 3*19])