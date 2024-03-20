#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_93
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/93/scripts/AI_Rosetta_Design_93.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/93/scripts/AI_Rosetta_Design_93.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/93
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/93/scripts/Rosetta_Design_93.sh
