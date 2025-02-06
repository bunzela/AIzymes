#!/bin/bash
#$ -V
#$ -cwd
#$ -N qsub_jupyter
#$ -hard -l mf=10G
#$ -o qsub_jupyter.out
#$ -e qsub_jupyter.err

echo jupyter is running on $(hostname -i) > IP_jupyter.IP
jupyter notebook --no-browser --ip=$(hostname -i) --port=23900 >> IP_jupyter.IP 

