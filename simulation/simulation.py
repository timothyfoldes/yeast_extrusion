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
from polykit.generators.initial_conformations import create_random_walk, grow_cubic
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

print('N =', var.N)
print('M =', var.M)
print('total number of particles =', var.N*var.M)


replicate = sys.argv[1]

### Initialize empty simulation object
gpu_num = var.gpu_num
system = hoomd.Simulation(device=hoomd.device.GPU(gpu_ids=[gpu_num]), seed=helper.rng())
print("Using random seed: %d" % system.seed)
print('Running on GPU number:', gpu_num)

### Build initial conformation and snapshot
snapshot = build.get_simulation_box(box_length=var.L, pad=1.)
print('Setting box size to =', var.L)
print('Box sizes fixes density to =', var.rho, 'particles per unit volume')

# Create initial positions for N monomers
monomer_positions = grow_cubic(N=var.N*var.M, boxSize=int(var.L-1))


### Read input force parameters
with open("scripts/force_dict_full.json", 'r') as dict_file:
    force_dict = json.load(dict_file)

### Read bond and monomer types from force_dict
bond_types = force_dict['Bonded forces'].keys()
monomer_types = force_dict['Non-bonded forces']['Attraction']['Matrix'].keys()

build.set_chains(snapshot, monomer_positions, [var.N]*var.M, monomer_type_list=list(monomer_types), bond_type_list=list(bond_types))


### Set particle types for probes

probe_postions = var.probe_positions
if probe_postions[-1] > var.N:
    print('Probe positions exceed number of particles')
print('probe positions are located at :', probe_postions)

typeid = np.zeros(var.N)
typeid[probe_postions] = 1

snapshot.particles.typeid[:] = np.tile(typeid, (var.M, 1)).reshape(var.N*var.M)

system.create_state_from_snapshot(snapshot)

### FORCES

### Setup neighbor list
# nl = hoomd.md.nlist.Cell(buffer=3.5)
nl = hoomd.md.nlist.Tree(buffer=3)


### Set forces from parameters in force dict

repulsion_forces = forces.get_repulsion_forces(nl, **force_dict)

bonded_forces = forces.get_bonded_forces(**force_dict)
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
if var.thermalize:
    print(f'thermalize system for {var.Nthermalize} steps')
    var.Nthermalize=50e4
    system.run(var.Nthermalize)

### NEIGHBOR LIST TUNE
if var.tune:
    helper.tune_rbuff(system, nl, steps=100_000, buffer_min=2, buffer_max=4, Nbins=10, set_r_buff=True, set_check_period= False)

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


filename = sys.argv[2]

print(filename)

filter = hoomd.filter.Type('B')
gsd_writer = hoomd.write.GSD(filename=filename, filter=filter, trigger=hoomd.trigger.Periodic(var.period), mode='wb', dynamic = ['property', 'momentum'])

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

