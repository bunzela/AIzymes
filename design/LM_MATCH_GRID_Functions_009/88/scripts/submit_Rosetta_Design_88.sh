#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_88
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/88/scripts/AI_Rosetta_Design_88.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/88/scripts/AI_Rosetta_Design_88.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/88
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/88/scripts/Rosetta_Design_88.sh
