# To Do
- Add propper usage instructions to each function
- ProteinMPNN starting to be implemented but not yet completed
- BACKGROUND_JOB starting to be implemented but not yet completed
- Add penalty if design results in a sequence that already exists? 

# AIzymes
This will be the AIzymes Github. Lets use the readme to collect notes.

> [!IMPORTANT]
> Track all versions of AIzyme_Functions in ./src/README.md

> [!TIP]
> Read [AIzymes Manual.docx](https://github.com/bunzela/AIzymes/blob/main/AIzymes%20Manual.docx) for detailed information on AIzymes
> 
> Feel free to change the Manual, but always track changes!

> [!TIP]
> See [How_to_push_and_pull.txt](https://github.com/bunzela/AIzymes/blob/main/How_to_push_and_pull.txt) for a quick manual how to push and pull changes to GitHub.

# Requirments
Download and install Rosetta from the following link, and set ROSETTA_PATH to point to rosetta_XXX/main/source/
```
https://www.rosettacommons.org/software/license-and-download
```
ESMfold through HuggingFace (https://huggingface.co/facebook/esmfold_v1)
```
pip install --upgrade transformers py3Dmol accelerate
```
ProteinMPNN might be used in the future and can be installed like this:
```
git clone https://github.com/dauparas/ProteinMPNN.git
pip install --upgrade transformers py3Dmol accelerate torch
```
