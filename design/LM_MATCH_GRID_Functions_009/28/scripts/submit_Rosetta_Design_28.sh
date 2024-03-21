#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_28
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/28/scripts/AI_Rosetta_Design_28.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/28/scripts/AI_Rosetta_Design_28.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/28
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/28/scripts/Rosetta_Design_28.sh
