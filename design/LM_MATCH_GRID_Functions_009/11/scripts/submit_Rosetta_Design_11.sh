#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_11
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/11/scripts/AI_Rosetta_Design_11.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/11/scripts/AI_Rosetta_Design_11.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/11
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/11/scripts/Rosetta_Design_11.sh
