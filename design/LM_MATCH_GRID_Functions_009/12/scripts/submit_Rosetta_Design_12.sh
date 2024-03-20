#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_12
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/12/scripts/AI_Rosetta_Design_12.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/12/scripts/AI_Rosetta_Design_12.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/12
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/12/scripts/Rosetta_Design_12.sh
