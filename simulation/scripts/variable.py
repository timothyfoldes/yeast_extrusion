import numpy as np
import sys
sys.path.append('./scripts')
import helper_functions_simulation as helper

gpu_num = 0


N = 1600
probe_positions = np.arange(180, 1600, 240)
# N = 1000
M = 10
replicas = [0]

confinement_radius = 4*N**(0.6)
rho = 0.2
L = helper.get_box_size(M, N, rho)
dt = 0.05
gamma = 1
kt = 1
cuttoff = [1.5, 2, 2.5, 3]

# Truth array of FORCES to be used in simulation.py
attraction = False
extrusion = False
confinement = False
repulsion = True
bonded = True


tune = False
thermalize = True

Nthermalize = 1_000_000
Nsteps = 200_000_000
Nframes = 10_000
period = 50

# probe_dist = 542
# probe_position_1 = 250
# probe_position_2 = probe_position_1 +  probe_dist