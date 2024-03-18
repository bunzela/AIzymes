# To Do
> [!IMPORTANT]
> @Abbie <br>
> Generate a large dataset for Tudor for training <br>
> Establish scoring for electric fields. This should add an electrostatic_score to all_scores.csv. In addition, please make a seperate dataframes electric_fields.csv that contains the apparent per-residue electric field for each residue for all variants. We might use these information for ML training. <br>
> Establish MD for analysis

> [!IMPORTANT]
> @Lucas <br>
> establish Rosetta match <br>
> Generate a large dataset for Tudor for training <br>
> Establish ProteinMPNN. Has been started to be implemented but not yet completed <br>

> [!IMPORTANT]
> @Tudor <br>
> Train ML model to guide mutagenesis in AI.zymes <br>

- Add penalty if design results in a sequence that already exists? 
- Add propper usage instructions to each function
- Modularize AIzymes. Ideally make individual modules for main parts (Design functions, main functions, helper functions, plotting functions, etc) and load with one master module
- BACKGROUND_JOB starting to be implemented but not yet completed

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


To install everything necessary for AIzymes in a new conda environment (no ProteinMPNN):
```
conda create -n AIzymes python=3.9 matplotlib scipy jupyter pandas networkx
conda activate AIzymes
conda install conda-forge::biopython conda-forge::accelerate conda-forge::transformers
```
