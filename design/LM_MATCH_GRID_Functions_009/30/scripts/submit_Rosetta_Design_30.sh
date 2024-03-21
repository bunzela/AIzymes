#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_30
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/30/scripts/AI_Rosetta_Design_30.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/30/scripts/AI_Rosetta_Design_30.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/30
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/30/scripts/Rosetta_Design_30.sh
