#!/bin/bash
#SBATCH -N -1
#SBATCH --gres=gpu:1 --ntasks-per-node=1 -N 1
#SBATCH -t 3:00:00
#SBATCH --mem=20G
#SBATCH -A vertaix
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=ap7956@princeton.edu
source ~/.bashrc
conda activate testingMM_env
python runAla2.py --replicas 32 --vendi --stop 7500 --simu-time 100 --nu 100 --part 0 --output-path 'Vendi_Ala2/output'
python runAla2.py --replicas 32 --simu-time 900 --part 1 --output-path 'Vendi_Ala2/output'
for i in {2..30};
    do python runAla2.py --replicas 32 --simu-time 1000 --recording-interval 0.1 --part $i --output-path 'Vendi_Ala2/output';
done
