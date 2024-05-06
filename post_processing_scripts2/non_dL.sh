#!/bin/bash

#SBATCH --job-name=non_dL_NGC3115
#SBATCH --output=new_NGC3115_non_dL.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1

which python

python NGC3115_non_dL.py

echo "Run complete."
