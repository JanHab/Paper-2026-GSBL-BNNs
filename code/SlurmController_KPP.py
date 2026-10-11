
"""
This script creates a slurm job which will execute the "command" defined below.
Therefore, a run_simulation.sh file will be created in the current working directory.
    The "slurm/default_jobscript" file is used and manipulated to create the run_simulation.sh file.
This is executed by slurm.
"""


from slurm.clusterrun import createsbatch
import os

# SLURM job parameters
threads = 1  # multi-threading
nnodes = 1  # multi-node
time = '25:00:00'   # HH:MM:SS -> allocated time for the job. Relevant for scheduling. 
memory_request = '32G' # RAM 

# Parameter analysis
N_ENSEMBLES = 10
script = 'Run_KPP2D.py'

SEED = 42
NX = 64

beta = 1.0
kappa = 10.0
M = 484
vartheta1 = 10.0
network_architecture = 1  # 0: 64-64-64-64, 1: 128-128-128-128, 2: 80-80-1000, 3: 100-100-1200, 4: 256-256-256-256


for i in range(N_ENSEMBLES):
    seed = SEED + i * 1_000
    # Command to run the simulation
    command = f"python code/{script} {seed} {kappa} {beta} {M} {NX} {vartheta1} {network_architecture}"
    print(f"Running command: {command}")

    # For debuggin
    # os.system(command)

    # Slurm job submission 
    createsbatch(
            command, # The command line argument which we will use
            nproc=threads, nnodes=nnodes, 
            time=time, mem=memory_request, 
            output_file=f"job_{os.path.basename(script).replace('.py', '')}_KPP_NX{NX}_M{M}_kappa{kappa}_beta{beta}_vartheta1{vartheta1}_net_arc{network_architecture}_SEED{seed}.out" # console log
    )
