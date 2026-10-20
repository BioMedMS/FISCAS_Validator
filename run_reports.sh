#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500GB
#SBATCH --partition zen3
#SBATCH --time=24:00:00
#SBATCH --job-name="FISCAS-Validator"
#SBATCH --mail-type=ALL
#SBATCH --output /scratch/tmp/j_schw74/validator.log
#SBATCH --mail-user=j_schw74@uni-muenster.de

cd /scratch/tmp/j_schw74/fiscas-validator/
module load uv
uv run run_reports.py
echo -n "Total amount of memory used (in bytes): "
cat /sys/fs/cgroup/$(cat /proc/self/cgroup | awk -F ':' '{print $3}')/memory.peak
