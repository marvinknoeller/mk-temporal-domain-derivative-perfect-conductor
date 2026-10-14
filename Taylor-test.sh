#!/bin/bash
#SBATCH --job-name=taylor_test
#SBATCH --account=project_2008159
#SBATCH --partition=medium
#SBATCH --time=8:00:00
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=24 --cpus-per-task=16  # The product should be 384

# Taylor test on refinement 5 (12288 RWG dofs); prints the table and writes TaylorTest_ref5.mat.
# Every solve keeps 603 LU factors of 2.4 GB each (1.5 TB), hence the 6 nodes.

# Set the number of threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Place and bind threads to single cores
# Comment the following lines if binding is not desired
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Run the program (python -m mpi4py stops all ranks if one of them fails)
source bempp-env/bin/activate

srun python -m mpi4py Taylor_test_TD.py --refinement 5
