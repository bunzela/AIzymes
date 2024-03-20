#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_10
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/10/scripts/AI_Rosetta_Design_10.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/10/scripts/AI_Rosetta_Design_10.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/10
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/10/scripts/Rosetta_Design_10.sh
