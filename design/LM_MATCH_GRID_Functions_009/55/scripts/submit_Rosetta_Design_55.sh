#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_55
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/55/scripts/AI_Rosetta_Design_55.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/55/scripts/AI_Rosetta_Design_55.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/55
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/55/scripts/Rosetta_Design_55.sh
