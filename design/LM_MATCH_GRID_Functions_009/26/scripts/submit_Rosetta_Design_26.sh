#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_26
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/26/scripts/AI_Rosetta_Design_26.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/26/scripts/AI_Rosetta_Design_26.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/26
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/26/scripts/Rosetta_Design_26.sh
