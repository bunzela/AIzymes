#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_87
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/87/scripts/AI_Rosetta_Design_87.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/87/scripts/AI_Rosetta_Design_87.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/87
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/87/scripts/Rosetta_Design_87.sh
