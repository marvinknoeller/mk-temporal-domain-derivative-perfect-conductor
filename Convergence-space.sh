#!/bin/bash
#SBATCH --job-name=conv_space
#SBATCH --account=project_2008159
#SBATCH --partition=medium
#SBATCH --time=3:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24 --cpus-per-task=16  # The product should be 384

# Space study: refinement 2, 3, 4 with M = 1600, compared with Convergence_reference.mat
# (submit after Convergence-reference.sh has finished). Writes Convergence_space.mat after every run.
# The largest run (refinement 4) keeps 1760 LU factors of 0.15 GB each (about 270 GB).

# Set the number of threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Place and bind threads to single cores
# Comment the following lines if binding is not desired
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Run the program (python -m mpi4py stops all ranks if one of them fails)
source bempp-env/bin/activate

srun python -m mpi4py Convergence_test_TD.py --study space
