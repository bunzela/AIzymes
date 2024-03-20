#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_40
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/40/scripts/AI_Rosetta_Design_40.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/40/scripts/AI_Rosetta_Design_40.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/40
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/40/scripts/Rosetta_Design_40.sh
