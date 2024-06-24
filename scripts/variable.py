import numpy as np
gpu_num = 1

N = 1000
replicas = np.arange(50)

confinement_radius = 4*N**(0.6)
L = confinement_radius*2
dt = 0.05
gamma = 1
kt = 1
cuttoff = [1.5, 2, 2.5, 3]

# Truth array of FORCES to be used in simulation.py
attraction = True
extrusion = False
confinement = False
repulsion = True
bonded = True


tune = False
thermalise = False


Nsteps = 8_600_000*4
Nframes = 10_000
period = 10_000


probe_position_1 = 250
probe_position_2 = probe_position_1 + 542 