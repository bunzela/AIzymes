#!/bin/bash
#SBATCH --job-name=TEST_controller
#SBATCH --partition=scc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/controller.log
#SBATCH --error=/mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/controller.log

set -e  # Exit script on any error

cd /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/..

echo "Current Working Directory:" > test.txt
pwd >> test.txt
echo "Timestamp:" >> test.txt
date >> test.txt

python /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/start_controller_parallel.py

echo "Timestamp:" >> test.txt
date >> test.txt

