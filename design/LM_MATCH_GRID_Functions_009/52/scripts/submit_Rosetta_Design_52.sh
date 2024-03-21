#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_52
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/52/scripts/AI_Rosetta_Design_52.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/52/scripts/AI_Rosetta_Design_52.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/52
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/52/scripts/Rosetta_Design_52.sh
