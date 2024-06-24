import subprocess
import sys
import shlex
sys.path.append('.')
sys.path.append('./scripts')
import variable as var
import os

# Initialize the replicate
replicate = 0
cutoffs = var.cuttoff

for cutoff in cutoffs:
    for replicas in var.replicas:
        dist = int(var.probe_position_2 - var.probe_position_1)
        directory = f'test_cuttoff/probedist_{dist}/{cutoff}'
        os.makedirs(f'./data/{directory}/', exist_ok=True)
        filename = f'trajectory_N{var.N}_cuttoff{cutoff}_period{var.period}_probe1{var.probe_position_1}_dist{dist}_{replicate}'
        filepath = f'data/{directory}/{filename}.gsd'
        # Execute the Python script with the current replicate value
        print(f'Using GPU number: {var.gpu_num}')
        print(f"Running simulation number {replicate} with parameter: cutoff {cutoff} and probe distance {dist}")

        # log_file = f"./log/simulation_probedist_542_{replicate}.log"
        os.makedirs(f'./log/{directory}/', exist_ok=True)
        logpath = f'log/{directory}/{filename}.log'
        
        # Run the simulation.py script and redirect output to the log file
        with open(logpath, 'w') as log:
            command = f'python simulation.py {replicate} {cutoff} {filepath}'
            command_list = shlex.split(command)
            result = subprocess.run(command_list, stdout=log, stderr=subprocess.STDOUT)

        # Optional: Add a sleep interval if you want to control the execution rate
        # import time
        # time.sleep(1)
        
        print(replicate)
        # Increment the replicate
        replicate += 1

