# Note to Reviewers
In this code repository, you find the code for our manuscript: AI.zymes – A modular platform for evolutionary enzyme design.

You can access the code used for the submitted manuscript under src/AIzymes_013.py. 

Example notebooks to test AI.zymes are given under AIzymes/design/TEST/

Upon peer-reviewed publications, a separate repository will be uploaded to the ETH Research Collection. This upload will be unmodifiable and associated with a DOI. Because the upload will contain data that may change during peer-review (e.g. additional analyses requested by reviewers), the upload to the ETH Research Collection will only be completed after peer review. 

# Current Development
Tudor is working on this branch to add de novo enzyme generation using ZymCTRL and to integrate Boltz-1 as a structure predictor.

# Future To Do
| **Packages**                | **Description**                                                                                                  | **Person** |
|-----------------------------|------------------------------------------------------------------------------------------------------------------|------------|
| LigandMPNN                  | .                                                                                                                | ???        |
| ChemNet                     | [AI tool to generate ensembles](https://doi.org/10.1101/2024.09.25.614868)                                       | ???        |
| Additional Base Filter      | [Base vdW energy in Rosetta seems to correlate with activity?](https://onlinelibrary.wiley.com/doi/full/10.1002/pro.4400) | ???        |
| ElektroNN                   | Not yet published, currently developed by Jens Meiler, NN tool generate electron density of full protein structure | ???        |

RiffDiff – Rotamer Inverted Fragment Finder Diffusion

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
