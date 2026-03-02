#!/bin/bash
#SBATCH --job-name=gui360_data
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=scripts/GUI_360/prepare_gui360_rl_data.log

cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1
python3 scripts/GUI_360/prepare_gui360_rl_data.py --max-val 50
