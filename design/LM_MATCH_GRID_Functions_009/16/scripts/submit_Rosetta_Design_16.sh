#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_16
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/16/scripts/AI_Rosetta_Design_16.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/16/scripts/AI_Rosetta_Design_16.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/16
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/16/scripts/Rosetta_Design_16.sh
