#!/bin/sh

#SBATCH --nodes=12
#SBATCH --tasks-per-node=5
#SBATCH --mem=64GB
#SBATCH --time=20:00:00
#SBATCH --job-name=InfereCLaDR_full
#SBATCH --output=InfereCLaDR_64GB_12_taus_2_seeds_20_bs_60tasks.out

module purge
module load r/intel/3.4.2 python/intel/2.7.12 bedtools/intel/2.26.0
source /home/kmt331/inferelator_ng/py2.7/bin/activate

cd /home/kmt331/inferelator_ng
export PYTHONPATH=$PYTHONPATH:$(pwd)/kvsstcp

time python ~/inferelator_ng/kvsstcp/kvsstcp.py --execcmd 'srun -n '${SLURM_NTASKS}' python yeast_inferecladr_workflow_runner.py'

