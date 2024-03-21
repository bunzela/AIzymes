#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_54
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/54/scripts/AI_Rosetta_Design_54.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/54/scripts/AI_Rosetta_Design_54.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/54
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/54/scripts/Rosetta_Design_54.sh
