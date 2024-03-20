#!/bin/bash
#$ -V
#$ -cwd
#$ -N MATCH_Rosetta_Design_86
#$ -hard -l mf=16G
#$ -o /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/86/scripts/AI_Rosetta_Design_86.out
#$ -e /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/86/scripts/AI_Rosetta_Design_86.err

# Output folder
cd /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/86
pwd
bash /home/lmerlicek/AIzymes/design/LM_MATCH_GRID_Functions_009/86/scripts/Rosetta_Design_86.sh
