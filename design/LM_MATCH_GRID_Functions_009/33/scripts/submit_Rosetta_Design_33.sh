#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_33
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/33/scripts/AI_Rosetta_Design_33.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/33/scripts/AI_Rosetta_Design_33.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/33
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/33/scripts/Rosetta_Design_33.sh
