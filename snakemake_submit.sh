#!/bin/bash -l
#SBATCH --partition cpu_long
#SBATCH --mem 6G
#SBATCH --time 27-23:59:59
#SBATCH --job-name snakeautocluster
#SBATCH --cpus-per-task=6
#SBATCH -e logs/sbatchSnakefile_progress_err.log
#SBATCH -o logs/sbatchSnakefile_progress_out.log


module purge
module add slurm
source activate autocluster
cd /gpfs/home/lmb529/ruggleslabHome/autocluster
mkdir -p logs/slurm/

snakemake -j 999 -p --verbose -s autocluster.smk \
--keep-going \
--cluster-config cluster.json \
--cluster "sbatch --mem={cluster.mem} -t {cluster.time} -o {cluster.output} -p {cluster.partition}"
