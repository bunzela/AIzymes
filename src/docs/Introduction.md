# Introduction
## Coding philosophy
AI.zymes is a modular program that seamlessly combines computational methods for design, structure prediction, and machine learning in a coherent enzyme design workflow (Fig. 1). To that end, AI.zymes employs a controller function that orchestrates the design process. The controller collects information from the designs and stores them in a shared database, selects which variants to submit for design, and decides what type of design or structure prediction algorithms to run with the selected variant. The controller is designed to manage a user-defined number of parallel design jobs, automatically starting a new job as soon as a previous one is completed. 
 
![General Flow Chart of AI.zymes](images/Figure1_AIzymes_Sheme.pdf)
<sub>**Fig. 1 | General Flow Chart of AI.zymes.** Based on a set of input variables (grey), AI.zymes will be set up and started (yellow). The main program of AI.zymes is the controller (blue), which controls the overall workflow, decides what action to take (salmon, red), and writes information into the databases (green).</sub>

## Basic concepts

### Evolutionary "rounds" in AI.zymes

The evolutionary algorithm in AI.zymes is not based on performing classical evolutionary rounds comprising mutagenesis and screening. Instead, the algorithm aims to maximize performance by constantly running a set number of design jobs set by **max_jobs**. Whenever the number of running jobs falls below **max_jobs**, AI.zymes spawns another design or structure prediction job. To choose the parent variant for design, AI.zymes performs multi-objective Boltzmann selection. Importantly, all designs are treated equally during Boltzmann selection, which selects from the growing pool of designs irrespective of which round the designs stemmed. Thus, design is not limited by a maximum number of rounds but a maximum number of designs (**max_designs**).

### Scoring metrics

AI.zymes employs various scoring metrics to assess various properties relevant to enzyme activity. The **total_score** corresponds to the total energy of the system. This score reflects the **total_score** of a structure in Rosetta and is referred to as "stability score" in the main text. The **interface_score** corresponds to the binding energy of the ligand to the protein. This score was calculated using Rosetta's InterfaceScoreCalculator mover and is referred to as the "interface score" in AI.zymes. The **catalytic_score** corresponds to the score of the catalytic interaction. It is calculated from the ideal interaction geometry described in a Rosetta_Match constraint file format. Finally, the **efield_score** describes the electric fields along the scissile C-H bond. Fields were calculated using FieldTools, and this score is referred to as the "electric field" in the main text.

Note that each variant in AI.zymes can have two sets of scores. All variants have scores from their initial design run. Variants selected for redesign also undergo structure prediction and relaxation, generating an updated set of scores. If available, Boltzmann selection is based on these updated scores; otherwise, the original design scores are used.

Furthermore, to run evolution in a forward-thinking manner, where the scores of the direct descendants of a variant can be included during screening, a metric called "potential" was introduced. The potential is the average of the score of the current variant and the scores of all its direct descendants.

### System startup

To start AI.zymes, the system must be set up using **setup_aizymes()** to create the overall file structure and stores all design-relevant parameters in the variables.json file. Among others, these include the name of the protein, ligand, and which residues can be designed. Furthermore, general settings can be set, such as how many jobs may run in parallel and how many designs will be made in total. Following the initial setup, **startup_controller()** reads these parameters from the variables.json file and initializes all other required variables. Thus, **setup_aizymes()** is only rone once, whereas **startup_controller()** is executed every time the code is restarted. 

### Controller

The controller is the central program of AI.zymes. It constantly cycles between the following steps:

1)	**check_running_jobs()** determines how many jobs are currently running on the system. If **MAX_JOBS** are running, the controller sleeps for 20 s and rechecks the number of running jobs, or else the controller initiates a new design cycle.

2)	**update_scores()** iterates through all designs and updates the **all_scores.csv** database. **update_scores()** also unblocks all indices for which the structure prediction runs are completed. 

3)	**check_parent_done()** checks if the initial non-evolutionary design for the parent variant has concluded. If fewer designs than N_PARENT_JOBS have been made, the controller skips the Boltzmann selection and instead runs **start_parent_design()**, starting a **run_RosettaDesign()** job on the parent structure.

4)	**boltzmann_selection()** initiates the Boltzmann selection to identify the next scaffold for design.

5)	**start_calculation()** finally starts the design based on the selected input structure. **start_calculation()** checks if there is a structure of the selected design that went through **ESMfold_RosettaRelax()**. If not, the controller will start the structure prediction and block the selected index. If there is a relaxed structure, the controller will generate a new index into which the design will be stored. This involves creating a folder for the new design and appending the **all_scores.csv** file with the selected index. To allow for a blend of design methods, the selected design will use **ProteinMPNN** instead of **RosettaDesign** with a user-defined probability **ProteinMPNN_PROB**.

### Boltzmann selection

In contrast to all other scores, the **catalytic_score** has a clear minimum, with a score of zero reflecting a perfect agreement of the design with the target catalytic geometry. Thus, **catalytic_score** was not included during Boltzmann selection but used as a binary cutoff to exclude variants before selection. To that end, the mean plus one standard deviation of the **catalytic_score** of all designs is calculated, and all variants with a **catalytic_score** below that cutoff are removed. Additionally, structures that are currently undergoing structure prediction are excluded from Boltzmann selection to prevent redundant structure prediction on the same structure.

Boltzmann selection is based on the design potentials and not on their scores. Boltzmann selection is performed on the combined_potential, which is calculated from the average of the z-score standardized **total_potential**, **interface_potential**, and **efield_potential**. Note that during normalization, potentials for which lower values are better are inverted (**total_potential** and **interface_potential**). Thus, higher combined_potentials correspond to better variants. Boltzmann selection is performed at a user-defined temperature kbt_boltzmann. To increase selection stringency during design, kbt_boltzmann decreases with each new design from an initial **KBT_BOLTZMANN** value with a **KBT_BOLTZMANN_DECAY** rate in a single exponential decay.
Boltzmann selection

### Database

The all_scores.csv database is the primary file containing all information from an AI.zymes run. Among other details, the database includes data on the specific settings for making each design and its resulting scores. Additionally, **all_scores.csv** keeps track of which structures are currently undergoing structure prediction and are therefore excluded from Boltzmann selection. This exclusion prevents redundant structure prediction from being performed multiple times on the same structure.

## Available Modules in AI.zymes

Currently, AI.zymes has established **run_RosettaDesign()** and **run_ProteinMPNN()** for design. For structure prediction and scoring, **run_ESMfold_RosettaRelax()** and **calc_efields_score()** have been established. Importantly, AI.zymes is built highly modularly, facilitating the future addition of other protein engineering packages to augment the computational evolution algorithm.

### Design with RosettaDesign

**run_RosettaDesign()** setups and submits a RosettaDesign run for the selected index based on RosettaScripts. AI.zymes dynamically generates the input .xml files that control RosettaScripts. An example .xml file can be found in 6.4 Example .xml file for RosettaDesign. Briefly, a geometry bias from the RosettaMatch constraint file is introduced with the AddOrRemoveMatchCsts (6.2 Rosetta Match constraint file), and the protein is repacked and minimized using the EnzRepackMinimize mover. Subsequently, the protein is designed using FastDesign for 3 repeats while applying a bias to the input sequence using the FavorSequenceProfile mover with a weight of **CST_WEIGHT**. Designable active-site residues are defined with DESIGN, and all mutations but cysteine are permitted to avoid the introduction of disulfide bonds, which can affect reproducibility if redox states are not tightly controlled. For the catalytic residue, only glutamate and aspartate are permitted to maintain the catalytic mechanisms relying on a carboxylate base for protein abstraction. Catalytic residues were defined via a Rosetta Match Constraint File. After design, the protein is relaxed for 1 repeat with FastRelax mover without the geometry bias from the RosettaMatch constraint file. The final scores, including the **interface_score** calculated with InterfaceScoreCalculator, are given for the relaxed structure.

### Design using ProteinMPNN

**run_ProteinMPNN()** setups and submits a ProteinMPNN run for the selected index using its sequence as input. A **bias_by_res.json** file is generated that applies a bias to the input sequence. The input structure was parsed using the ProteinMPNN helper scripts (**parse_multiple_chains.py**, **assign_fixed_chains.py**, **make_fixed_positions_dict.py**) to define the target chain and specify the fixed positions. Residues in DESIGN are excluded from design with ProteinMPNN and are only designed with Rosetta. ProteinMPNN is executed with the specified sampling temperature ProteinMPNN_T and the generated bias file, producing a set of 100 candidate sequences. The highest-scoring sequence in terms of **global_score** is used for the subsequent modeling steps. Because ProteinMPNN does not provide a structure but only a sequence, **run_ProteinMPNN()** always spawns a **run_ESMfold_RosettaRelax()** to generate a structure and scores for Boltzmann selection.

### Structure prediction with ESMfold, RosettaRelax and MDMin

**run_ESMfold_RosettaRelax()** using both the sequence and structure as input to predict the protein structure with ESMFold. Because ESMFold cannot predict the structure of the substrate-enzyme complex, the coordinates of the ligand are transferred from the parent structure file into the predicted structure after the alignment of the two structures. Afterward, sidechains are stripped from the ESMfold model, and the resulting backbone-ligand complex is repacked and relaxed using Rosetta with the FastRelax mover. For designs that do not provide a structure (e.g., ProteinMPNN), **run_ESMfold_RosettaRelax()** uses the structure of the parent variant as input.

### Electric field calculations

**calc_efields_score()** is used to determine active-site electric fields of the input structures. Electric field calculation is performed with FieldTools (https://github.com/bunzela/FieldTools). Fields are calculated using point charges from the ff19SB AMBER forcefield. All stated fields correspond to the effective field along the scissile C-H bond. FieldTools relies on Coulomb's law using the point charges from the system's topology file and the coordinates from the input structure or trajectory to calculate the electric field along a target bond. FieldTools calculates the electric field vectors E using the Coulomb constant and the vector from the center of the target bond to the charges in the system. Subsequently, the effective field E_eff projected along the target bond is calculated from the scalar product of the directional unity vector along that bond and the total field vector E.