#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_47
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/47/scripts/AI_Rosetta_Design_47.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/47/scripts/AI_Rosetta_Design_47.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/47
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/47/scripts/Rosetta_Design_47.sh
