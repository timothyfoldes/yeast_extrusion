import numpy as np
from lefs_cython.simple import LEFSimulator, constants
import lefs_cython
import multiprocessing as mp
import networkx as nx
import os
import logging
import pickle

lefs_cython.LEFSimulator = LEFSimulator
def shortest_path_with_shortcuts(shortcuts, start, end, processivity):
    G = nx.Graph()
    """
    Compute the shortest path between two points (start and end) with shortcuts (loop anchors).
    """
    
    # Collect all unique points including start and end
    points = set()
    points.add(start)
    points.add(end)
    adjusted_shortcuts = []
    for shortcut in shortcuts:
        # discard shortcuts that are far from the start and end
        if (shortcut[1]<start - 1.5*processivity) or (shortcut[0]>end + 1.5*processivity): 
            pass
        else:
            adjusted_shortcuts.append(shortcut)
            points.update(shortcut)

    shortcuts = adjusted_shortcuts
    # Convert the set to a sorted list
    points = sorted(points)
    
    # Create a mapping from points to indices
    point_index = {point: idx for idx, point in enumerate(points)}
    
    # Add edges for direct distances between consecutive points
    for i in range(len(points) - 1):
        distance = abs(points[i] - points[i + 1])
        G.add_edge(point_index[points[i]], point_index[points[i + 1]], weight=distance)
    
    # Add shortcut edges using precomputed indices
    for xi, yi in shortcuts:
        G.add_edge(point_index[xi], point_index[yi], weight=0)
    
    # Find shortest path using Dijkstra's algorithm
    index_start = point_index[start]
    index_end = point_index[end]
    length, path = nx.single_source_dijkstra(G, index_start, index_end)
    
    # Return the shortest distance and the path taken
    # To return the actual points in the path, uncomment the following line:
    # return length, [points[node] for node in path]
    return length, len(path)


def compute_average_loop_size(position):
    return np.diff(position, axis = -1).mean().mean()

def compute_fpt(positions, target_size):
    fpts = []
    for _ in range(5000):
        t = np.random.randint(1000, len(positions))
        positions_conditionned = positions[t:]
        if np.argmax(positions_conditionned < target_size).sum() == 0: # if the target is never reached pass
            pass
        else:
            fpts.append(np.argmax(positions_conditionned < target_size))

    
    if len(fpts) == 0:
        mean_fpts =  len(positions)
    else:
        mean_fpts = np.mean(fpts)
    return mean_fpts, np.array(fpts)


def run_sim(N, N_LEFS, LEF_steps, processivity):

    load_array = 1 * np.ones((N, 5))
    unload_array = np.ones((N, 5))/(0.5*processivity)  
    capture_array = np.zeros((N, 2))  # no CTCF
    release_array = np.zeros(N)
    pause_array = np.zeros(N)  # no pausing
    
    positions = np.zeros((LEF_steps, N_LEFS, 2))
    LEF = LEFSimulator(N_LEFS, N, load_array, unload_array, capture_array, release_array, pause_array, skip_load=False)
    for k in range(LEF_steps):
        LEF.steps(k,k+1)
        positions[k,:,:] = LEF.get_LEFs()
    return np.array(positions)

def get_shortest_paths(LEF_positions, pos, sep, processivity):
    shortest = np.zeros(LEF_positions.shape[0])
    path = np.zeros(LEF_positions.shape[0])
    for k in range(LEF_positions.shape[0]):
        shortest[k], path[k] = shortest_path_with_shortcuts(LEF_positions[k,:,:], pos, pos + sep, processivity)
    return np.array(shortest), np.array(path)



# Function to process each combination of parameters
def process_combination(args):
    l, k, i, j, N, counter = args
    pos = replicate_positions[l]
    sep = separation[k]
    N_LEFS = N_LEFSs[i]
    processivity = processivities[j]
    
    positions = run_sim(N, N_LEFS, LEF_steps, processivity) 
    shortest, path = get_shortest_paths(positions, pos, sep, processivity)
    shortest_mean = shortest.mean()
    number_of_loops = path.mean()
    average_loop_size = compute_average_loop_size(positions)

    fpt = np.zeros(len(target_sizes))
    for t, target_size in enumerate(target_sizes):
        fpt[t] = compute_fpt(shortest, target_size = target_size)[0]

    if counter % 10 == 0:
        print(f'Completed {counter} out of {len(replicate_positions)*len(separation)*len(N_LEFSs)*len(processivities)} tasks.')

    filename = f'/net/levsha/share/foldes/projects/yeast_extrusion/simulation_1D/data/1D_fpt/28-06/29-06/result_{l}_{k}_{i}_{j}.pkl'
    with open(filename, 'wb') as f:
        result = (l, k, i, j, shortest_mean, fpt, average_loop_size, number_of_loops)
        pickle.dump(result, f)
    
    return l, k, i, j, shortest_mean, fpt, average_loop_size, number_of_loops


# Configure logging
logging.basicConfig(filename='example.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Given values
N = 10000
LEF_steps = 1000000
processivities = np.linspace(50, 500, 20)
N_LEFSs = np.arange(10, 100, 10)

separation = [363, 473, 542]
target_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50]
replicate_positions = np.arange(1000, N - 1000, 1000)
Nreplicates = len(replicate_positions)

print('N_LEFSs', N_LEFSs)
print('processivities', processivities)
print('separation', separation)
print('target_sizes', target_sizes)
print('replicate_positions', replicate_positions)

# Prepare arguments for parallel processing
tasks = []
counter = 0
print(f'Running {len(replicate_positions)*len(separation)*len(N_LEFSs)*len(processivities)} tasks.')
for l, pos in enumerate(replicate_positions):
    for k, sep in enumerate(separation):
        for j, processivity in enumerate(processivities): 
            for i, N_LEFS in enumerate(N_LEFSs):
                tasks.append((l, k, i, j, N, counter))
                counter += 1

# Execute tasks in parallel using multiprocessing
with mp.Pool(mp.cpu_count()) as pool:
    results = pool.map(process_combination, tasks)


shortests = np.zeros((len(replicate_positions), len(separation), len(N_LEFSs), len(processivities)))
fpts = np.zeros((len(target_sizes), len(replicate_positions), len(separation), len(N_LEFSs), len(processivities)))
average_loop_sizes = np.zeros((len(replicate_positions), len(separation), len(N_LEFSs), len(processivities)))
number_of_loops = np.zeros((len(replicate_positions), len(separation), len(N_LEFSs), len(processivities)))
loop_cooperations = np.zeros((len(replicate_positions), len(separation), len(N_LEFSs), len(processivities)))
# Store the results

for result in results:
    l, k, i, j, shortest_mean, fpt, average_loop_size, loop_cooperation = result
    shortests[l, k, i, j] = shortest_mean
    fpts[:, l, k, i, j] = fpt
    average_loop_sizes[l, k, i, j] = average_loop_size
    loop_cooperations[l, k, i, j] = loop_cooperation
    # print(N_LEFSs[i], processivities[j], shortests[k, i, j])

results_dic = {'shortests': shortests,
            'fpts': fpts,
            'N_LEFSs': N_LEFSs,
            'processivities': processivities,
            'separation': separation,
            'target_sizes': target_sizes,
            'average_loop_size' : average_loop_sizes,
            'number_of_loops' : number_of_loops
                }

print(results_dic)

# At this point, shortests and fpts arrays are populated with the desired values

np.save(f'./results.npy', results_dic, allow_pickle=True)
os.makedirs(f'./data/1D_fpt/no_ctcf_steps{LEF_steps}/', exist_ok=True)
np.save(f'./data/1D_fpt/no_ctcf_steps{LEF_steps}/results.npy', results_dic, allow_pickle=True)
np.save(f'./data/1D_fpt/no_ctcf_steps{LEF_steps}/replicate_positions.npy', replicate_positions, allow_pickle=True)
np.save(f'./data/1D_fpt/no_ctcf_steps{LEF_steps}/processivities.npy', processivities, allow_pickle=True)
np.save(f'./data/1D_fpt/no_ctcf_steps{LEF_steps}/N_LEFSs.npy', N_LEFSs, allow_pickle=True)
np.save(f'./data/1D_fpt/no_ctcf_steps{LEF_steps}/separation.npy', separation, allow_pickle=True)
np.save(f'./data/1D_fpt/no_ctcf_steps{LEF_steps}/target_sizes.npy', target_sizes, allow_pickle=True)
np.save(f'./data/1D_fpt/no_ctcf_steps{LEF_steps}/average_loop_size.npy', average_loop_sizes, allow_pickle=True)
np.save(f'./data/1D_fpt/no_ctcf_steps{LEF_steps}/shortests.npy', shortests, allow_pickle=True)
np.save(f'./data/1D_fpt/no_ctcf_steps{LEF_steps}/fpts.npy', fpts, allow_pickle=True)