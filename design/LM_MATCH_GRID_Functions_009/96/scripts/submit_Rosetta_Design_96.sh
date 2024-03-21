#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_96
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/96/scripts/AI_Rosetta_Design_96.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/96/scripts/AI_Rosetta_Design_96.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/96
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/96/scripts/Rosetta_Design_96.sh
