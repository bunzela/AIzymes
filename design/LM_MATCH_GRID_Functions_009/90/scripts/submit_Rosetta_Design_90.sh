#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_90
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/90/scripts/AI_Rosetta_Design_90.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/90/scripts/AI_Rosetta_Design_90.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/90
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/90/scripts/Rosetta_Design_90.sh
