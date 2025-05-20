#!/bin/bash

##SBATCH --output=/scratch/gpfs/kf1298/
#SBATCH --job-name=rfdiff    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)
##SBATCH --mem=60G                # total memory per node (4 GB per cpu-core is default)
##SBATCH --partition=pli
##SBATCH --constraint=gpu80
##SBATCH --gres=gpu:1          # number of gpus per node
#SBATCH --time=0:59:59          # total run time limit (HH:MM:SS)
##SBATCH --mail-type=begin        # send email when job begins
##SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=thufkd21@gmail.com


module load anaconda3/2024.6

export LD_LIBRARY_PATH=/scratch/gpfs/kf1298/anaconda3/envs/rfnew/lib:$LD_LIBRARY_PATH

conda activate /scratch/gpfs/kf1298/anaconda3/envs/rfnew
cd /scratch/gpfs/kf1298/HyLIP

python parallel_predict_csv.py --input data/All_results.csv --output data/All_results_with_predictions.csv --debug --num_processes 100