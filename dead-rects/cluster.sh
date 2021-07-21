#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=bayes_tiny_tiny
#SBATCH --mail-type=END
#SBATCH --mail-user=hhs4@nyu.edu
#SBATCH --output=slurm-output/slurm_%A_%a.out

index=$SLURM_ARRAY_TASK_ID
job=$SLURM_JOB_ID
ppn=$SLURM_JOB_CPUS_PER_NODE
module purge
module load anaconda3/5.3.1

source activate test

cd dead-leaves

echo $index
echo $job
echo $ppn

python bayes_tiny_tiny.py -i $SLURM_ARRAY_TASK_ID -p $HOME/tinytinydeadrects/validation
