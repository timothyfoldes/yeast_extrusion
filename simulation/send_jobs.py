import subprocess
import sys
import shlex
sys.path.append('.')
sys.path.append('./scripts')
import variable as var
import os

# Initialize the replicate
replicate = 0
cutoff = var.cuttoff
Nsteps = var.Nsteps


for replicas in var.replicas:
    data_path = '/net/levsha/share/foldes/projects/yeast_extrusion/data/'

    directory = f'without_attraction/period{var.period}'
    os.makedirs(f'./data/{directory}/', exist_ok=True)
    spacing = var.probe_positions[1] - var.probe_positions[0]
    filename = f'trajectory_N{var.N}_M{var.M}_period{var.period}_Nsteps_{Nsteps}_spacing{spacing}'
    filepath = f'data/{directory}/{filename}.gsd'
    print(f'Using GPU number: {var.gpu_num}')
    print(f"Running simulation number {replicate} with parameter")

    # log_file = f"./log/simulation_probedist_542_{replicate}.log"
    os.makedirs(f'./log/{directory}/', exist_ok=True)
    logpath = f'log/{directory}/{filename}.log'
    
    # Run the simulation.py script and redirect output to the log file
    with open(logpath, 'w') as log:
        command = f'python simulation.py {replicate} {filepath}'
        command_list = shlex.split(command)
        result = subprocess.run(command_list, stdout=log, stderr=subprocess.STDOUT)

    # Optional: Add a sleep interval if you want to control the execution rate
    # import time
    # time.sleep(1)
    
    print(replicate)
    # Increment the replicate
    replicate += 1

