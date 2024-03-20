#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_76
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/76/scripts/AI_Rosetta_Design_76.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/76/scripts/AI_Rosetta_Design_76.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/76
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/76/scripts/Rosetta_Design_76.sh
