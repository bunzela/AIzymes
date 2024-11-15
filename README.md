# To Do
| **Action Item**             | **Description**                                                                                                  | **Person** |
|-----------------------------|------------------------------------------------------------------------------------------------------------------|------------|
| `MS` Finalize variant charact | pH-dependence & MM Kin of variants for preprint                                                                  | Vivian     |
| `MS` Finalize variant charact | Chrystallography (for future full paper)                                                                         | ???        |
| `MS` Finalize manual        | Detailed description, setup, and usage instructions. Includes docstrings in code                                 | Vivian     |
| `MS` Plots                  | What is missing both in terms of parameter testing and from experiment                                           | Lucas      |
| MDMin Design Option         | Work on an MDMin design option to run between ESMfold and RosettaRelax. Updates are on the branch `MD_minimise`. | Abbie      |
| Running MD Option           | To be done after MDMin                                                                                           | Abbie      |
| Enable AIzymes for CATPHOTO | Establish heme & design protocols to make AI.zymes accessible to everyone in the group                           | ???        |
| Finalize Plot Functions     | Complete the tree plot and UMAP plotting functions.                                                              | Vivian     |
| Optimize `cst_weight`       | Optimize the `cst_weight` to increase the total score and smooth hamming distance over generations.              | Vivian     |
| Optimize `cst_weight`       | Optimize the `boltzman_weight`                                                                                   | Vivian     |
| Test New Design Runs        | Test whether starting a new design run from a previous design provides improvement.                              | Vivian     |
| Test New Design Runs        | Test whether starting a new design run from a previous design provides improvement.                              | Vivian     |

| **Packages**                | **Description**                                                                                                  | **Person** |
|-----------------------------|------------------------------------------------------------------------------------------------------------------|------------|
| LigandMPNN                  | .                                                                                                                | ???        |
| ChemNet                     | [AI tool to generate ensembles](https://doi.org/10.1101/2024.09.25.614868)                                       | ???        |
| Additional Base Filter      | [Base vdW energy in Rosetta seems to correlate with activity?](https://onlinelibrary.wiley.com/doi/full/10.1002/pro.4400) | ???        |
| ElektroNN                   | Not yet published, currently developed by Jens Meiler, NN tool generate electron density of full protein structure | ???        |

RiffDiff – Rotamer Inverted Fragment Finder Diffiusion

# Related work

[REvoLd: Ultra-Large Library Screening with an Evolutionary Algorithm in Rosetta](https://pubmed.ncbi.nlm.nih.gov/38711437/)
put other links here, please

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

# Requirements
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


To install everything necessary for AIzymes (except Rosetta) in a new conda environment (no ProteinMPNN) from Abbie:
```
conda env create -f environment.yml
```
