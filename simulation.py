#imports

import os
import json
import hoomd
import cooltools

import numpy as np
import matplotlib.pyplot as plt

import polychrom_hoomd.log as log
import polychrom_hoomd.build as build
import polychrom_hoomd.forces as forces
import polychrom_hoomd.render as render
import polychrom_hoomd.extrude as extrude
import sys


from lattice_translocators.generators.Translocator import Translocator
from lattice_translocators.engines.SymmetricExtruder import SymmetricExtruder

from polykit.analysis import polymer_analyses, contact_maps
from polykit.generators.initial_conformations import create_random_walk
sys.path.append('./scripts')
import helper_functions_simulation as helper
import variable as var


### FORCES

print('attraction:', var.attraction)
print('extrusion:', var.extrusion)
print('confinement:', var.confinement)
print('repulsion:', var.repulsion)
print('bonded:', var.bonded) 


print('period =', var.period)
print('Nsteps =', var.Nsteps)
print('Nframes =', var.Nframes)

replicate = sys.argv[1]

### Initialize empty simulation object
gpu_num = var.gpu_num
system = hoomd.Simulation(device=hoomd.device.GPU(gpu_ids=[gpu_num]), seed=helper.rng())
print("Using random seed: %d" % system.seed)
print('Running on GPU number:', gpu_num)

### Build initial conformation and snapshot
snapshot = build.get_simulation_box(box_length=var.L, pad=1.)
print('Setting box size to =', var.L)

# Create initial positions for N monomers
monomer_positions = create_random_walk(step_size=1, N=var.N)

### Read input force parameters
with open("scripts/force_dict_full.json", 'r') as dict_file:
    force_dict = json.load(dict_file)

### Read bond and monomer types from force_dict
bond_types = force_dict['Bonded forces'].keys()
monomer_types = force_dict['Non-bonded forces']['Attraction']['Matrix'].keys()

build.set_chains(snapshot, monomer_positions, var.N, monomer_type_list=list(monomer_types), bond_type_list=list(bond_types))


### Setup sticky probe positions by assigning monomers to a type 0 backbone or 1 probes
typeid = np.zeros(var.N)
typeid[var.probe_position_1 -10: var.probe_position_1], typeid[var.probe_position_2 : var.probe_position_2+10] = 1, 2 
snapshot.particles.typeid[:] = typeid

print('probe_position_1 =', var.probe_position_1)
print('probe_position_2 =', var.probe_position_2)

system.create_state_from_snapshot(snapshot)

### FORCES

### Setup neighbor list
# nl = hoomd.md.nlist.Cell(buffer=3.5)
nl = hoomd.md.nlist.Tree(buffer=3)


### Set forces from parameters in force dict

repulsion_forces = forces.get_repulsion_forces(nl, **force_dict)

bonded_forces = forces.get_bonded_forces(**force_dict)

force_dict['Non-bonded forces']['Attraction']['cutoff'] = sys.argv[2]
attraction_forces = forces.get_attraction_forces(nl, **force_dict)

force_dict['External forces']['Confinement']['Spherical'] = dict(R=var.confinement_radius)
confinement_forces = forces.get_confinement_forces(**force_dict)


force_list = [repulsion_forces, bonded_forces, attraction_forces, confinement_forces]
force_truth_array = [var.repulsion, var.bonded, var.attraction, var.confinement]
force_field = [element[0] for element, truth in zip(force_list, force_truth_array) if truth]

### Initialize integrators and Langevin thermostat

gamma=var.gamma
langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=var.kt)
langevin.gamma.default = gamma
integrator = hoomd.md.Integrator(dt=0.05, methods=[langevin], forces=force_field)
logger = log.get_logger(system)

system.operations.integrator = integrator

###WRITTER
system.operations.writers.append(log.table_formatter(logger, period=2e4))

### THERMALIZATION
print('thermalize system without the attractive forces to avoid getting looped while thermalization')
var.Nthermalize=50e4
system.run(var.Nthermalize)

### NEIGHBOR LIST TUNE
if var.tune:
    helper.tune_rbuff(system, nl, steps=100_000, buffer_min=1, buffer_max=3, Nbins=10, set_r_buff=True, set_check_period= False)

### ADD ATTRACTION FORCES
if var.attraction:
    print('Adding attraction forces')
    integrator.forces = force_field + attraction_forces
else :
    print('No attraction forces')

snapshot = system.state.get_snapshot()  
N = snapshot.particles.N
eps = force_dict['Non-bonded forces']['Attraction']['Matrix']['B']['B']
cutoff = force_dict['Non-bonded forces']['Attraction']['Cutoff']

dist = var.probe_position_2-var.probe_position_1
filename = sys.argv[3]

print(filename)
gsd_writer = hoomd.write.GSD(filename=filename, trigger=hoomd.trigger.Periodic(var.period), mode='wb')
system.operations.writers.append(gsd_writer)

if var.attraction:
    while helper.get_probe_distance(system, var.probe_position_1, var.probe_position_2)>6:
        print('Start search for 100_000 steps')
        system.run(100_000)
        print('Distance between probes:', helper.get_probe_distance(system, var.probe_position_1, var.probe_position_2))
    print("Target distance reached")
else :
    print('No attraction forces, probes are not sticky')
    print('Simulation will run for ', var.Nsteps)
    system.run(var.Nsteps)


# Writes a txt file including all simulation parameters.

