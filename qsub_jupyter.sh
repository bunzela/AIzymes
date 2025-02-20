#!/bin/bash
#$ -V
#$ -cwd
#$ -N qsub_jupyter
#$ -hard -l mf=10G
#$ -o qsub_jupyter.out
#$ -e qsub_jupyter.err
#$ -q regular.q
cd

source ~/.bashrc

pwd

module list

conda env list

echo jupyter is running on $(hostname -i) > IP_jupyter.IP
jupyter notebook --no-browser --ip=$(hostname -i) --port=7564 >> IP_jupyter.IP 

