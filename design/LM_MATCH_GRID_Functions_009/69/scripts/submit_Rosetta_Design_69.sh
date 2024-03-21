#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_69
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/69/scripts/AI_Rosetta_Design_69.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/69/scripts/AI_Rosetta_Design_69.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/69
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/69/scripts/Rosetta_Design_69.sh
