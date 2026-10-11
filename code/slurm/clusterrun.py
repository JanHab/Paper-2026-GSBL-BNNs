from pathlib import Path
import os

# Saves a given command as an sbatch file for SLURM and executes it
def createsbatch(command, nproc=1, nnodes=1, time='00:15:00', mem='8G', output_file='output.out'):
    """createsbatch Creates a slurm batch script that can be executed on high performance clusters

    Takes user-defined properties and generates a corresponding slurm batch script 

    Parameters
    ----------
    command : str
        The command that should be executed by the slurm script
    nproc : int, optional
        number of processing units, by default 24
    nnodes : int, optional
        number of nodes, by default 1
    time : str, optional
        maximal time for the slurm job, by default '24:00:00'
    mem : str, optional
        memory requirement for the slurm job, by default '8G'
    output_file : str, optional
        path to the output file for the slurm job, by default 'output.out'
    """    
    sbatch_path = (Path(__file__).parent / 'default_jobscript.sh').resolve()
    # Change SLURM parameters
    with open(sbatch_path, 'rb') as sbatch_default:
        data = sbatch_default.read()
        data = data.replace(b'jn', bytes(str(output_file), 'utf-8'))
        data = data.replace(b'cppt', bytes(str(nproc), 'utf-8'))
        data = data.replace(b'numno', bytes(str(nnodes), 'utf-8'))
        data = data.replace(b'aot', bytes(time, 'utf-8'))
        data = data.replace(b'mem_req', bytes(mem, 'utf-8'))
        data = data.replace(b'cmd', bytes(str(command), 'utf-8'))
        data = data.replace(b'outfile', bytes(str(output_file), 'utf-8'))
    save_path = (Path(os.getcwd())/'run_simulation.sh').resolve()
    # Save sbatch file to cwd
    with open(save_path, 'wb') as sbatch:
        sbatch.write(data)
    runsbatch(Path(os.getcwd()))

def runsbatch(loc):
    os.system('sbatch {}/run_simulation.sh'.format(str(loc)))        
