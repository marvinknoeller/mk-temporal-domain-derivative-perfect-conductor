#!/bin/bash
#SBATCH --job-name=conv_reference
#SBATCH --account=project_2008159
#SBATCH --partition=medium
#SBATCH --time=8:00:00
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=24 --cpus-per-task=16  # The product should be 384

# Reference solution for both studies: refinement 5 (12288 RWG dofs), M = 1600, written to
# Convergence_reference.mat. Keeping all 1760 LU factors (2.4 GB each) would need 4.3 TB, more than
# 6 nodes hold. With --recompute_lu they are computed again for the derivative instead of being kept:
# twice the assembly and factorization work, but only a few dense matrices per rank in memory.

# Set the number of threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Place and bind threads to single cores
# Comment the following lines if binding is not desired
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Run the program (python -m mpi4py stops all ranks if one of them fails)
source bempp-env/bin/activate

srun python -m mpi4py Convergence_test_TD.py --study reference --recompute_lu
