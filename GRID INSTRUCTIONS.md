Instructions to run AIzmymes from compute node

1. generate ssh key

ssh-keygen -t rsa -b 4096 #(here do not enter anyhting (no password!) only standard settings)
ssh-copy-id $USER@bs-submit04.ethz.ch

2. start jupyter with

qsub qsub_jupyter.sh

3. use Bitvise to open new tunnel to connect with IP in IP_jupyter.IP 