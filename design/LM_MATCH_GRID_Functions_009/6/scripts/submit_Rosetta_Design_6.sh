#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_6
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/6/scripts/AI_Rosetta_Design_6.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/6/scripts/AI_Rosetta_Design_6.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/6
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/6/scripts/Rosetta_Design_6.sh
