## Coding philosophy
AI.zymes is a modular program that seamlessly combines computational methods for design, structure prediction, and machine learning in a coherent enzyme design workflow (Fig. 1). At the core, AI.zymes employes the controller that decides what action to take and assures that the maximum number of design jobs are running in parallel. The controller collects information from the designs and stores them in a shared database, selects which variants to submit for design and decides what type of design or structure prediction to perform with the selected variant. 
 
![General Flow Chart of AI.zymes](path/to/flow_chart_image.png)
<!-- Figure: fig_1 -->
<b>Fig. 1 | General Flow Chart of AI.zymes.</b> Based on a set of input variables (grey), AI.zymes will be set up and started (yellow). The main program of AI.zymes is the controller (blue), which controls the overall workflow, decides what action to take (salmon, red), and writes information into the databases (green).

## Available Packages
Currently established design packages include run_RosettaMatch and run_RosettaDesign. For structure prediction, run_ESMfold_RosettaRelax has been established. The next step will be to establish run_ProteinMPNN for protein design and run_ElectricFields to generate an additional scoring metric based on electrostatic stabilization of the TS. In the future, additional AI and MD packages should be implemented to guide the controller and augment scoring.
  
![Available packages](path/to/packages_image.png)
<!-- Figure: fig_2 -->
<b>Fig. 2 | Available packages.</b> Implemented packages are depicted in blue, packages for future implementation are depicted in grey and white based on urgency.

### RosettaMatch

RosettaMatch is an optional design step in AI.zymes that can be used to create de novo active sites. AI.zymes starts from all files in FOLDER_PARENT. If RosettaMatch should not be run, these can also be manually supplied by the user. RosettaMatch screens an input structure for potentially binding sites using an enzdes-type CST file. To that end, the Matcher tries to find pockets that can accommodate the ligand as well as all catalytic residues defined in a constraint file. Importantly, the Matcher ignores all sidechains present in the structure and only recognizes the input structure backbone. The Matcher can thus introduce new pockets and does not rely on structures that already contain a pocket.

run_RosettaMatch requires an input WT structure with a ligand molecule positioned roughly where the new active site is to be designed, run_RosettaMatch will relax the structure using run_ESMfold_RosettaRelax, so no initial realax is required. Furthermore, an enzdes-type CST file {LIGAND}_enzdes.cst and a {LIGAND}.params file  must be supplied. Finally, the matcher requires the definition of the central ligand atom {LIGAND}.central and various parameters. 

run_RosettaMatch produces several Match PDB structures in {FOLDER_HOME}/{FOLDER_MATCH} /matches that contain new catalytic residues and the bound reaction transition state. These structures can be accessed by the main AI.zymes algorithm through FOLDER_PARENT.

### RosettaDesign

### ESMfold_RosettaRelax

## Basic concepts

### System startup based on input settings

The system startup involve n optional setup / reset of AI.zymes (setup_aizymes) to start AI.zymes blank. In addition, a startup script is run every time the controller is started to load all necessary information for the controller to run smoothly (startup_controller).
To set up the system, various global variables need to be defined (see Globally stored variables). Among others, these include the name of the protein and ligand as well as which residues can be designed. Furthermore, general settings can be set such as how many jobs may run in parallel and how many designs are to be done in total. 

### Scoring

Three different scores have thus far proven valuable to identify promising enzyme designs: The total_score corresponding to the total energy of the system, the interface_score corresponding to the binding energy of the ligand to the protein, as well as the catalytic_score corresponding to the score of the catalytic interaction. AI.zymes uses the concept of potential to select which variants to take forward for design. Potential is aimed to provide some predictive information on the variants. Thus, each potential value corresponds to the arithmetic average of a structure’s score, as well as of the corresponding score from all its directed descendants. To select a variant for design, the total_potential, catalytic_potential, and interface_potential are normalized from 0 to 1 with 1 being the best, and the geometric mean is calculated from these potentials for each variant to give the combined_potential (Eq. 1). Boltzmann selection is performed on the combined_potential to finally identify the variant to be taken forward for design.
	combined_potential=∛(total_potential ×interface_potential×catalytic_potential  )	Eq. 1

### Databases

The ALL_SCORES.csv database is the main file that holds all information of the AI.zymes run. Amongst others, ALL_SCORES contains information on the parent scaffold variant, the precise design algorithm used, as well as key scoring metrics obtained from Rosetta. In addition, the BLOCKED.csv database contains a list of all structures that are currently undergoing structure prediction. These structures are excluded from Boltzmann selection, to prevent that structure prediction is needlessly performed multiple times based on the same structure.

### Controller

The controller is the central program of AI.zymes. It constantly cycles between three different scripts. The first scrip (update_scores) checks all designs in ALL_SCORES.csv that do not yet contain any scores. If it finds a finished design, it will update the scores of that design. update_scores also unblocks all indices for which the structure prediction runs are completed. Subsequently, the controller will find the next scaffold for design by Boltzman selection. Only unblocked indices will go into the selection algorithm and selection will be based on the combined_potential. Once an index is selected, the control starts the calculation. To that end, it checks if there is a structure of the selected design that went through ESMfold_RosettaRelax. If not, the controller will start the structure prediction and block the selected index. If there is a relaxed structure, the controller will generate a new index into which the design will be stored. This involved creating a folder for the new design and appending the ALL_SCORES.csv file with the selected index. Finally, the controller will check how many jobs are currently running. It will wait until the number of running jobs is lower than the maximum number of jobs to restart the controller cycle. 

### Globally stored variables

Various key variables controlling the behavior of AI.zymes are stored in variables.json. Variables that keep track of the system include the name of the parent structure (PARENT), the name of the bound ligand (LIGAND), a list of residue numbers to repack, design, and restrict (REPACK, DESIGN, RESTRICT), and the remark line that should be added on top of the PDB to define catalytic interactions (REMARK). The overall design flow is controlled by the maximum number of jobs that can run in parallel (MAX_JOBS), the number of jobs that should be run with the parent structure before the selection of the designed structure kicks in (N_PARENT_JOBS) and the maximum number of designs to be performed (MAX_DESIGNS). In addition, specific variables controlling the behavior of specific programs are set, including the Boltzmann temperature used during selection (KBT_BOLTZMANN), the constraint weight biasing design towards the parent sequence (CST_WEIGHT) and the probability to run ProteinMPNN instead of RosettaDesign (ProteinMPNN_PROB) as well as the temperauter used for ProteinMPNN ('ProteinMPNN_T'). Several other variables control the overall file architecture, including the current design folder (DESIGN_FOLDER), the path to Rosetta (ROSETTA_PATH), whether or not to run design in a quick testing mode (EXPLORE), the prefix used for job submission to identify the AI.zymes jobs (SUBMIT_PREFIX), and the identity of the cluster currently used (BLUEPEBBLE or GRID).
