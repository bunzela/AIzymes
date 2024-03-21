#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_99
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/99/scripts/AI_Rosetta_Design_99.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/99/scripts/AI_Rosetta_Design_99.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/99
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/99/scripts/Rosetta_Design_99.sh
