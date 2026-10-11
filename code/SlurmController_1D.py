
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
time = '03:30:00'   # HH:MM:SS -> allocated time for the job. Relevant for scheduling. 
memory_request = '16G' # RAM 

# Parameter analysis
N_ENSEMBLES = 50
scripts = [
    'Run_PiecewiseConstant_Noisy.py', 
    'Run_PiecewiseSmooth_Blur.py',
]

# Start with Gaussian and Cauchy priors
models = [
    'Gaussian',
    'Cauchy'
]
initializer_vec = [
    'Default',
    'Prior'
]

optimizer = 'muon'

kappa, beta, vartheta1 = 0.0, 0.0, 0.0 # Just something for the moment

for script in scripts:
    for model in models:
        for initializer in initializer_vec:
            if initializer == 'Prior' and model == 'Gaussian':
                initializer = 'Gaussian'
            elif initializer == 'Prior' and model == 'Cauchy':
                initializer = 'Cauchy'
            elif initializer != 'Default':
                raise ValueError(f"Invalid combination of model and initializer: {model}, {initializer}. Must be one of ['Gaussian', 'Cauchy'] for the Prior initializer.")
            # Command to run the simulation
            command = f"python code/{script} {N_ENSEMBLES} {model} {optimizer} {kappa} {vartheta1} {initializer}"
            print(f"Running command: {command}")

            # For debuggin
            # os.system(command)

            # Slurm job submission 
            createsbatch(
                    command, # The command line argument which we will use
                    nproc=threads, nnodes=nnodes, 
                    time=time, mem=memory_request, 
                    output_file=f"job_{os.path.basename(script).replace('.py', '')}_{model}_kappa{kappa}_beta{beta}_{optimizer}_{initializer}.out" # console log
            )



# GSBL parameter analysis
kappa_vec = [0.1, 1.0, 10.0, 100.0, 1_000.0]
vartheta1_vec = [0.1, 1.0, 10.0, 100.0, 1_000.0]
models = ['GSBL_Independent', 'GSBL_Shared']
initializer = 'GG'
for script in scripts:
    for model in models:
        for vartheta1 in vartheta1_vec:
            for kappa in kappa_vec:
                # Command to run the simulation
                command = f"python code/{script} {N_ENSEMBLES} {model} {optimizer} {kappa} {vartheta1} {initializer}"
                print(f"Running command: {command}")

                # For debuggin
                # os.system(command)

                # Slurm job submission 
                createsbatch(
                        command, # The command line argument which we will use
                        nproc=threads, nnodes=nnodes, 
                        time=time, mem=memory_request, 
                        output_file=f"job_{os.path.basename(script).replace('.py', '')}_{model}_kappa{kappa}_vartheta1_{vartheta1}_{optimizer}_{initializer}.out" # console log
                )


# GSBL with the best parameter and the default initialization
initializer = 'Default'
kappa, vartheta1 = 10.0, 100.0
for script in scripts:
    for model in models:
        # Command to run the simulation
        command = f"python code/{script} {N_ENSEMBLES} {model} {optimizer} {kappa} {vartheta1} {initializer}"
        print(f"Running command: {command}")

        # For debuggin
        # os.system(command)

        # Slurm job submission 
        createsbatch(
                command, # The command line argument which we will use
                nproc=threads, nnodes=nnodes, 
                time=time, mem=memory_request, 
                output_file=f"job_{os.path.basename(script).replace('.py', '')}_{model}_kappa{kappa}_vartheta1_{vartheta1}_{optimizer}_{initializer}.out" # console log
        )


# Now the combined learning steps
model = 'GSBL_Shared_CombinedOptimization'
kappa, vartheta1 = 10.0, 100.0
initializer_vec = ['Default', 'GG']
for script in scripts:
    for initializer in initializer_vec:
        # Command to run the simulation
        command = f"python code/{script} {N_ENSEMBLES} {model} {optimizer} {kappa} {vartheta1} {initializer}"
        print(f"Running command: {command}")

        # For debuggin
        # os.system(command)

        # Slurm job submission 
        createsbatch(
                command, # The command line argument which we will use
                nproc=threads, nnodes=nnodes, 
                time=time, mem=memory_request, 
                output_file=f"job_{os.path.basename(script).replace('.py', '')}_{model}_kappa{kappa}_vartheta1_{vartheta1}_{optimizer}_{initializer}.out" # console log
        )
