#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_36
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/36/scripts/AI_Rosetta_Design_36.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/36/scripts/AI_Rosetta_Design_36.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/36
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/36/scripts/Rosetta_Design_36.sh
