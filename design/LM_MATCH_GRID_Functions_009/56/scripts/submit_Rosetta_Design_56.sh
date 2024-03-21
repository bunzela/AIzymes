#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_56
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/56/scripts/AI_Rosetta_Design_56.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/56/scripts/AI_Rosetta_Design_56.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/56
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/56/scripts/Rosetta_Design_56.sh
