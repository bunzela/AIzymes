#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_45
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/45/scripts/AI_Rosetta_Design_45.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/45/scripts/AI_Rosetta_Design_45.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/45
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/45/scripts/Rosetta_Design_45.sh
