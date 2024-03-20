#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_46
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/46/scripts/AI_Rosetta_Design_46.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/46/scripts/AI_Rosetta_Design_46.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/46
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/46/scripts/Rosetta_Design_46.sh
