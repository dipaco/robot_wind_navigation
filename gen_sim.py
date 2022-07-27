import numpy as np
import shutil
import sys
import cv2
import os
from fluidsim.solvers.ns2d.solver import Simul
from fluidsim import load_sim_for_plot, load_state_phys_file
from matplotlib import pyplot as plt
from PIL import Image

BASE_FOLDER = '/home/diegopc/tmp/flow_sim'

params = Simul.create_default_params()

#import pdb
#pdb.set_trace()

# Delete the results folder
#shutil.rmtree(sim.output.path_run)

# air viscosity at 25 Cº
# (https://www.engineersedge.com/physics/viscosity_of_air_dynamic_and_kinematic_14483.htm)
params.nu_2 = 1.562e-5 #
params.forcing.enable = False

params.init_fields.type = 'noise'
params.init_fields.noise.velo_max = 15

params.output.periods_save.spatial_means = 1.
params.output.periods_save.spectra = 1.
params.output.periods_save.phys_fields = 0.1

params.time_stepping.t_end = 20
params.time_stepping.it_end = 50
params.oper.nx = 256
params.oper.ny = 256
params.oper.Lx = 10
params.oper.Ly = 10

sim = Simul(params)

if sys.argv[1] == 'sim':
    sim.time_stepping.start()
else:
    sim = load_state_phys_file(sim.output.path_run)

FPS = 10
video_name = os.path.join(BASE_FOLDER, 'video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#for t in range(params.time_stepping.it_end + 1):
for i, t in enumerate(np.linspace(0.0, params.time_stepping.t_end, params.time_stepping.t_end * FPS)):
    img_filename = os.path.join(BASE_FOLDER, f'step_{t:.4f}.png')
    sim.output.phys_fields.plot(time=t)

    plt.savefig(img_filename)
    frame = Image.open(img_filename)

    if i == 0:
        video = cv2.VideoWriter(video_name, fourcc, FPS, frame.size)

    video.write(np.array(frame)[..., 0:3])
    plt.close()

    print(f'Processed frame {t}.')

cv2.destroyAllWindows()
video.release()

