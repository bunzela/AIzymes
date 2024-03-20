#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_14
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/14/scripts/AI_Rosetta_Design_14.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/14/scripts/AI_Rosetta_Design_14.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/14
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/14/scripts/Rosetta_Design_14.sh
