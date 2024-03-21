#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_23
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/23/scripts/AI_Rosetta_Design_23.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/23/scripts/AI_Rosetta_Design_23.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/23
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/23/scripts/Rosetta_Design_23.sh
