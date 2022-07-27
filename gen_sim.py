import numpy as np
import shutil
import sys
import cv2
import os
import pathlib
from fluidsim.solvers.ns2d.solver import Simul
from fluidsim import load_sim_for_plot, load_state_phys_file
from matplotlib import pyplot as plt
from PIL import Image

params = Simul.create_default_params()

# Delete the results folder
#shutil.rmtree(sim.output.path_run)

# air viscosity at 25 Cº
# (https://www.engineersedge.com/physics/viscosity_of_air_dynamic_and_kinematic_14483.htm)
#params.nu_2 = 1.562e-3 #
#params.nu_4 = 2.468e-5
#params.nu_8 = 1.032e-4
#params.nu_m4 = 4.562e-5

# Simulation workspace and simulation time
params.time_stepping.t_end = 120
params.time_stepping.it_end = 50
params.oper.nx = 256
params.oper.ny = 256
params.oper.Lx = 10
params.oper.Ly = 10

params.forcing.enable = True
params.forcing.type = 'tcrandom_anisotropic'
#if params.forcing.type == 'tcrandom_anisotropic':
#    params.forcing.tcrandom_anisotropic.angle = np.pi * np.random.rand() / 2
params.forcing.nkmin_forcing = 0
params.forcing.nkmax_forcing = 2

#params.forcing.type = 'milestone'
#params.forcing.milestone.objects.number = np.random.randint(2, 5)
#params.forcing.milestone.objects.diameter = np.random.rand() + 1.5

'''forcing_type = ['proportional', 'tcrandom', 'tcrandom_anisotropic']
#forcing_type = ['tcrandom', 'tcrandom_anisotropic']
#forcing_idx = np.random.randint(len(forcing_type))
#params.forcing.type = forcing_type[forcing_idx]
number_cylinders = 3
params.forcing.type = 'milestone'
params.forcing.milestone.nx_max = min(
    params.oper.nx, round(32 * number_cylinders * params.oper.nx / params.oper.ny)
)
objects = params.forcing.milestone.objects
objects.number = number_cylinders'''

# Physics parameters
Re = np.random.rand() * 10 ** (np.random.randint(12, 14))
V0 = 10.0
L = 1
params.nu_2 = V0 * L / Re # Kinematic Viscosity
delta_x = params.oper.Lx / params.oper.nx
exp_f = 2 * np.random.rand() + 6.0
params.forcing.forcing_rate = 1.0
params.nu_8 = 0.2 / 3.0 * delta_x**exp_f

# Init conditions
params.init_fields.type = 'dipole'
params.init_fields.noise.velo_max = 12
params.init_fields.noise.length = 2 * np.random.rand() + 4.0

params.init_fields.modif_after_init = True

# Output parameters
params.output.periods_save.spatial_means = 1.0
params.output.periods_save.spectra = 1.
params.output.periods_save.phys_fields = 0.1

sim = Simul(params)

if sys.argv[1] == 'sim':
    sim.time_stepping.start()
else:
    sim = load_state_phys_file(sim.output.path_run)

FPS = 10
BASE_FOLDER = pathlib.Path(f'{sim.output.path_run}_video')
BASE_FOLDER.mkdir(parents=True, exist_ok=True)
video_name = BASE_FOLDER / 'video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#for t in range(params.time_stepping.it_end + 1):
for i, t in enumerate(np.linspace(0.0, params.time_stepping.t_end, params.time_stepping.t_end * FPS)):
    img_filename = BASE_FOLDER / f'step_{t:.4f}.png'
    sim.output.phys_fields.plot(time=t)

    plt.savefig(img_filename)
    frame = Image.open(img_filename)

    if i == 0:
        video = cv2.VideoWriter(str(video_name.resolve()), fourcc, FPS, frame.size)

    video.write(np.array(frame)[..., 0:3])
    plt.close()

    print(f'Processed frame {t}.')

cv2.destroyAllWindows()
video.release()

