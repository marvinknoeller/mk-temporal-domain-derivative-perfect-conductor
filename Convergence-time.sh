#!/bin/bash
#SBATCH --job-name=conv_time
#SBATCH --account=project_2008159
#SBATCH --partition=medium
#SBATCH --time=8:00:00
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=12 --cpus-per-task=32  # The product should be 384

# Time study: refinement 5, M = 32, 40, ..., 800, compared with Convergence_reference.mat
# (submit after Convergence-reference.sh has finished). Writes Convergence_time.mat after every run.
# The largest run (M = 800) keeps 1040 LU factors of 2.4 GB each (2.5 TB, about 420 GB per node).
# 12 ranks per node leave enough room for the matrices being assembled; if memory still runs out,
# add --recompute_lu to the srun line (twice the factorization work, almost no stored factors).

# Set the number of threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Place and bind threads to single cores
# Comment the following lines if binding is not desired
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Run the program (python -m mpi4py stops all ranks if one of them fails)
source bempp-env/bin/activate

srun python -m mpi4py Convergence_test_TD.py --study time
