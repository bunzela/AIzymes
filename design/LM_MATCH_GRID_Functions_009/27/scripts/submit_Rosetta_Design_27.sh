#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_27
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/27/scripts/AI_Rosetta_Design_27.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/27/scripts/AI_Rosetta_Design_27.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/27
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/27/scripts/Rosetta_Design_27.sh
