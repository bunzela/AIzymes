#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_81
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/81/scripts/AI_Rosetta_Design_81.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/81/scripts/AI_Rosetta_Design_81.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/81
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/81/scripts/Rosetta_Design_81.sh
