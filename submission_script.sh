#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 1
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 10080
### Partition or queue name
#SBATCH -p general
### memory per cpu, in MB
#SBATCH --mem-per-cpu=4000
### Job name
#SBATCH -J 'mock_BD'
### output and error logs
#SBATCH -o mock_BD_%a.out
#SBATCH -e mock_BD_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=sandro.tacchella@cfa.harvard.edu
source activate pro
export WDIR=/n/home03/stacchella/CANDELS/prospector_fits/mock_BD/
srun -n 1 python $APPS/prospector/scripts/prospector_dynesty.py \
--param_file=$WDIR/parameter_files/mock_BD_params.py \
--outfile=$WDIR/results/mock_BD_"${SLURM_ARRAY_TASK_ID}" \
--i_comp="${SLURM_ARRAY_TASK_ID}"
