import hoomd
import numpy as np
import codecs
import os

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

def plateau(x,t1,t2):
    t1,t2 = np.sort([t1,t2])
    return np.heaviside(x-t1, 1)-np.heaviside(x-t2, 0)

def std_bridge(t, T, H):

    return (t**(2*H) + T**(2*H) - np.abs(t-T)**(2*H)) / (2*T**(2*H)) 

def normalised_bridge(t, t1,t2, H):
    T = t2 - t1

    return plateau(t,t1,t2) * std_bridge(plateau(t,t1,t2)*(t-t1), T, H) +  np.heaviside(t-t2, 0)


def walk_gen(N):
    walk = np.zeros((N,3))
    walk[1:,:] = np.random.normal(size = (N-1,3))
    return np.cumsum( walk, axis = 0 )


def bridge_gen( N, l, L ):
    walk = walk_gen(N)

    # V = walk[L+l] - walk[l]
    V = walk[L+l-1] - walk[l]


    bridge = normalised_bridge(np.arange(0,N), l,l+L, 1/2)

    return walk - np.array([ bridge* V[0], bridge* V[1], bridge* V[2] ]).T

def generate_initial_configuration_bridged(N, probe_position_1, probe_position_2):
    walk1 = walk_gen(probe_position_1)[::-1]
    walk2 = bridge_gen(probe_position_2-probe_position_1, 0, probe_position_2-probe_position_1)
    walk3 = walk_gen(N-(probe_position_2))
    init_pos = np.concatenate((walk1, walk2, walk3))
    return init_pos

def rng():
    # Generate RNG seed
    rng_seed = os.urandom(2)
    rng_seed = int(codecs.encode(rng_seed, 'hex'), 16)
    return rng_seed

def get_probe_distance(system, probe_position_1, probe_position_2):
    snapshot = system.state.get_snapshot()
    positions = snapshot.particles.position
    return np.linalg.norm(positions[probe_position_1+5] - positions[probe_position_2+5])

