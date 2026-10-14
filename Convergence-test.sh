#!/bin/bash
#SBATCH --job-name=convergence_test
#SBATCH --account=project_2008159
#SBATCH --partition=medium
#SBATCH --time=8:00:00
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=24 --cpus-per-task=16  # The product should be 384

# Usage: compute the reference solution first; both studies read it from Convergence_reference.mat.
#   jid=$(sbatch --parsable --partition=large --nodes=10 Convergence-test.sh reference)
#   sbatch --dependency=afterok:$jid Convergence-test.sh time
#   sbatch --dependency=afterok:$jid Convergence-test.sh space
# The reference (refinement 5, M = 1600) keeps 1760 LU factors of 2.4 GB each (4.3 TB), more than the
# 6 nodes of the medium partition hold. The time study needs at most 2.5 TB (M = 800).

# Set the number of threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Place and bind threads to single cores
# Comment the following lines if binding is not desired
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Run the program (python -m mpi4py stops all ranks if one of them fails)
source bempp-env/bin/activate

srun python -m mpi4py Convergence_test_TD.py --study $1
