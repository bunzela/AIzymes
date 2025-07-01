# Available Tools

AI.zymes works by wrapping around a range of bioengineering tools. While some of these tools can be readily installed using *pip* or *conda*, others need to be installed as stand-alone programs. Furthermore, some programs need to be registered in the **config** file genereatd during the first run of AI.zymes

Rosetta         <d>Protein design and relaxation <br>        
LigandMPNN      <d>Protein design with ligands <br>    
SolubleMPNN     <d>Protein design without ligands <br>  
AlphaFold3      <d>Structure prediciton with ligands <br> 
ESMFold         <d>Structure prediciton without ligands <br>
AmberTools      <d>Structure minimization <br>
FieldTools      <d>Electric field calculation <br>        
BioDC           <d>Redox Potential calculation <br>

## Protein Design Tolls

### Rosetta

Rosetta is a comprehensive suite of software for physics-based macromolecular modeling and design. In AI.zymes, it is used for (1) structure relaxation and repacking, and (2) enzyme design.

<br> **Available at:** https://rosettacommons.org/software/download/


### LigandMPNN

LigandMPNN is a deep-learning design tool that focuses on optimizing protein–ligand interactions. Within AI.zymes, it can be integrated to tailor the active-site environment by adjusting residues that directly interact with the chemical transition state or substrate.

<br> **Available at:** https://github.com/dauparas/LigandMPNN


### SolubleMPNN and ProteinMPNN

SolubleMPNN is a deep-learning design tool used to generate protein sequences that fold into a given structural scaffold, often enhancing the stability of the input structure in the process. In AI.zymes, ProteinMPNN is used to remodel the protein scaffold, focusing on regions away from the active site to improve stability without compromising enzyme activity.

<br> **Available at:** https://github.com/dauparas/ProteinMPNN

## Structure Prediction Tools

### AlphaFold3

AlphaFold3 is a deep learning-based structure prediction tool that can include a range of small-molecule ligands during prediction. In AI.zymes, AlphaFold3 is used to predict the three-dimensional structure of designed enzymes, especially those bound to ligands such as heme.

<br> **Available at:** https://github.com/google-deepmind/alphafold3

### ESMFold

ESMFold leverages large protein language models for structure prediction. It is usually faster than AlphaFold3 but cannot include small-molecule ligands. In AI.zymes, ESMFold is used in combination with RosettaRelax and MDMin to generate ligand-bound structural models. This combination offers a fast route to assess design variants for proper folding and active-site organization without the computational overhead of traditional methods.

<br> **Available at:** https://github.com/facebookresearch/esm
<br> **Installed using:** pip install transformers fair-esm

### AmberTools

AmberTools is a collection of programs for molecular mechanics and dynamics, which help parameterize molecules and perform energy calculations. In the context of AI.zymes, AmberTools is used to minimize enzyme–ligand complexes (MDMin).

<br> **Available at:** https://ambermd.org/AmberTools.php
<br> **Installed using:** conda install conda-forge::ambertools


## Scoring Tools

### FieldTools

FieldTools is a computational tool for calculating electric fields in and around enzyme active sites. Since electrostatic interactions can greatly influence enzyme catalysis, AI.zymes incorporates FieldTools to quantitatively assess the catalytic electric field along a target bond. This evaluation is integrated into the multi-objective scoring system, enabling selection of variants that not only stabilize the protein and its active site but also optimize the electric field for enhanced catalysis.

<br> **Available at:** https://github.com/bunzela/FieldTools

> [!NOTE] 
> FieldTools has been hard-coded into AI.zymes and does not need to be installed seperately.

### BioDC

Work in progress...
