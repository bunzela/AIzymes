#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_77
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/77/scripts/AI_Rosetta_Design_77.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/77/scripts/AI_Rosetta_Design_77.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/77
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/77/scripts/Rosetta_Design_77.sh
