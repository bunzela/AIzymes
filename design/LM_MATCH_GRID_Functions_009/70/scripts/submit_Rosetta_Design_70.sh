#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_70
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/70/scripts/AI_Rosetta_Design_70.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/70/scripts/AI_Rosetta_Design_70.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/70
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/70/scripts/Rosetta_Design_70.sh
