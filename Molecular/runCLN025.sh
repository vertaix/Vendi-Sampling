#!/bin/bash
#SBATCH -N -1
#SBATCH --gres=gpu:1 --ntasks-per-node=1 -N 1
#SBATCH -t 9:00:00
#SBATCH --mem=30G
#SBATCH -A vertaix
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=ap7956@princeton.edu
source ~/.bashrc
conda activate MM_env
python -u runCLN025.py --replicas 32 --vendi --stop 80000 --simu-time 400 --nu 250 --recording-interval 2 --part 0 --output-path 'Vendi_CLN025/output'
python -u runCLN025.py --replicas 32 --simu-time 600 --recording-interval 2 --part 1 --output-path 'Vendi_CLN025/output'
for i in {2..50};
do python runCLN025.py --replicas 32 --simu-time 1000 --recording-interval 2 --part $i --output-path 'Vendi_CLN025/output';
done