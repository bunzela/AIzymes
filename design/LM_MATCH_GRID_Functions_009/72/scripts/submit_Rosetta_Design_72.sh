#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_72
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/72/scripts/AI_Rosetta_Design_72.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/72/scripts/AI_Rosetta_Design_72.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/72
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/72/scripts/Rosetta_Design_72.sh
