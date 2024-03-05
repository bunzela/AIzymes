Hi All,

This will be the AIzymes Github. Lets use the readme to collect notes.

> [!IMPORTANT]
> Track all versions of AIzyme_Functions in ./src/README.md

> [!TIP]
> Read AIzymes Manual.docx for detailed information
> Feel free to change the Manual, but always track changes!

# Requirments
AIzymes requires ESMfold through HuggingFace (https://huggingface.co/facebook/esmfold_v1)
```
pip install --upgrade transformers py3Dmol accelerate
```

# How to push/pull from HPC (in progress, please adjust)
On the cluster, run:
```
ssh-keygen
```
Add the ssh key on this site
```
https://github.com/settings/keys
```
On the cluster, run:
```
git clone https://github.com/bunzela/AIzymes
```

what next?

# To Do
> [!IMPORTANT]
> @Abbie, please adjust design/AIzyme_Run_Abbie_1.ipynb so that the paths point to the correct file in ./src
- ProteinMPNN starting to be implemented but not yet completed
- BACKGROUND_JOB starting to be implemented but not yet completed
- Add penalty if design results in a sequence that already exists? 
