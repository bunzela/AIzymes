#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_29
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/29/scripts/AI_Rosetta_Design_29.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/29/scripts/AI_Rosetta_Design_29.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/29
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/29/scripts/Rosetta_Design_29.sh
