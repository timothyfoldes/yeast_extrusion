#imports

import os
import json
import hoomd
import codecs
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

def tune_rbuff(sim, nl, steps, buffer_min, buffer_max, Nbins, set_r_buff, set_check_period):
    r_buffs = np.round(np.linspace(buffer_min, buffer_max, Nbins), 3)
    TPSs = []
    shortest_rebuilds = []
    print(f'Staring tune of buffer size for the neighboring \n')
    print(f'r-buff in {r_buffs} \n')

    for r_buff in r_buffs:
        print(f'buffer size = {r_buff} \n')
        nl.buffer = r_buff
        print(f'starting run for {steps} steps \n')
        sim.run(steps)
        print(f'for buffer size = {r_buff}, TPS = {sim.tps} \n, shortest rebuild = {nl.shortest_rebuild} \n')
        TPSs.append(sim.tps)
        shortest_rebuilds.append(nl.shortest_rebuild)

    best_r_buff = r_buffs[TPSs.index(max(TPSs))]
    best_shortest_rebuild = shortest_rebuilds [TPSs.index(max(TPSs))]
    print(f'The best buffer size is: {best_r_buff} with a TPS of {max(TPSs)} and shortest rebuild = {best_shortest_rebuild} \n')
    if set_r_buff == True:
        print(f'setting buffer size to {best_r_buff} \n')
        nl.buffer = best_r_buff
    if set_check_period:
        print(f'setting rebuuld check delay to {best_shortest_rebuild - 2} \n')
        nl.rebuild_check_delay = best_shortest_rebuild - 2
    return best_r_buff, best_shortest_rebuild



r_buff = 4
N = sys.argv[1]
L = 1000
Nthermalize=5e4
Nsteps = 10e6
Nframes = 5_000
period = int(Nsteps/Nframes)


# Initialize empty simulation object
hoomd_device = build.get_hoomd_device()
# Generate RNG seed
rng_seed = os.urandom(2)
rng_seed = int(codecs.encode(rng_seed, 'hex'), 16)

print("Using entropy-harvested random seed: %d" % rng_seed)
system = hoomd.Simulation(device=hoomd_device, seed=rng_seed)

#Create simulation from snapshot: Setup bonds, monomer types and initial positions 

snapshot = build.get_simulation_box(box_length=L)
# Build random, dense initial conformations
monomer_positions = create_random_walk(step_size=1, N=N)
print(f'Number of monomers: {N} \n')


# Read input force parameters
with open("force_dict_full.json", 'r') as dict_file:
    force_dict = json.load(dict_file)


#read bond and monomer types from force_dict
bond_types = force_dict['Bonded forces'].keys()
monomer_types = force_dict['Non-bonded forces']['Attraction']['Matrix'].keys()
print(f'Bond types: {bond_types} \n')
print(f'Monomer types: {monomer_types} \n')

build.set_chains(snapshot, monomer_positions, N, monomer_type_list=list(monomer_types), bond_type_list=list(bond_types))

# Setup sticky probe positions by assigning monomers to a type 0 backbone or 1 probes
probe_position_1 = 50
probe_position_2 = 450

# set position of sticky probes
typeid = np.zeros(N)
typeid[probe_position_1 : probe_position_1 +10], typeid[probe_position_2 : probe_position_2 +10] = 1, 1 
snapshot.particles.typeid[:] = typeid
print(f'setting probe positions at {probe_position_1} and {probe_position_2}\n')

system.create_state_from_snapshot(snapshot)

### Setup forces

# Setup neighbor list
nl = hoomd.md.nlist.Tree(buffer=r_buff)

# Set chromosome excluded volume
repulsion_forces = forces.get_repulsion_forces(nl, **force_dict)
bonded_forces = forces.get_bonded_forces(**force_dict)
attraction_forces = forces.get_attraction_forces(nl, **force_dict)

force_field = repulsion_forces + bonded_forces + attraction_forces

# Initialize integrators and Langevin thermostat
gamma=1
langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
langevin.gamma.default = gamma
integrator = hoomd.md.Integrator(dt=0.05, methods=[langevin], forces=force_field)
logger = log.get_logger(system)

system.operations.integrator = integrator
system.operations.writers.append(log.table_formatter(logger, period=1e4))

#thermalize system

system.run(Nthermalize)

# tune_rbuff(system, nl, steps = 50_000, buffer_min=0.5, buffer_max=1.5, Nbins=10, set_r_buff=True, set_check_period=False)


filename = f'data/trajectory_N{system.state.get_snapshot().particles.N}.gsd'
gsd_writer = hoomd.write.GSD(filename=filename, trigger=hoomd.trigger.Periodic(period), mode='wb')
system.operations.writers.append(gsd_writer)

system.run(Nsteps)