#!/bin/bash
#SBATCH --job-name=example3_30
#SBATCH --account=project_...
#SBATCH --partition=medium
#SBATCH --time=17:00:00
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=24 --cpus-per-task=16  # The product should be 384

# Set the number of threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Place and bind threads to single cores
# Comment the following lines if binding is not desired
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Run the program
source bempp-env/bin/activate

srun python IP_efie_TD.py --scatterer 0 --example 6 --direction_case 2 --noise 30.0 --N 10 --refinement_rec 4 --alpha 1e-2
