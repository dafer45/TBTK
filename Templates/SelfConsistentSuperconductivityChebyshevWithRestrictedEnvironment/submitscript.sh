#!/bin/bash
#SBATCH -A snic2016-1-19
#SBATCH -J SCCS
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kristofer.bjornson@physics.uu.se

./build/a.out
