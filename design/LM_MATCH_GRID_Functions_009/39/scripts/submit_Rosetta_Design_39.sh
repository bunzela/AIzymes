#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_39
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/39/scripts/AI_Rosetta_Design_39.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/39/scripts/AI_Rosetta_Design_39.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/39
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/39/scripts/Rosetta_Design_39.sh
