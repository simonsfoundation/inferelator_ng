#!/bin/sh

#SBATCH --nodes=3
#SBATCH --tasks-per-node=20
#SBATCH --mem=64GB
#SBATCH --time=40:00:00
#SBATCH --job-name=InfereCLaDR_full
#SBATCH --output=InfereCLaDR_64GB_15_taus_8_seeds_30_bs_clust1_60tasks.out

module purge
module load r/intel/3.4.2 python/intel/2.7.12 bedtools/intel/2.26.0
source /home/kmt331/inferelator_ng/py2.7/bin/activate

cd /home/kmt331/inferelator_ng
export PYTHONPATH=$PYTHONPATH:$(pwd)/kvsstcp

time python ~/inferelator_ng/kvsstcp/kvsstcp.py --execcmd 'srun -n '${SLURM_NTASKS}' python yeast_inferecladr_workflow_runner.py'

