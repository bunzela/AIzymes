## Class: AIzymes_MAIN

Main class for managing AIzymes workflow, including setup, initialization, control, and plotting functions.


### Function: __init__

Initializes an instance of the AIzymes_MAIN class.


### Function: setup

Sets up the AIzymes project environment with specified parameters.

**Parameters:**

**FOLDER_HOME (str)**:	Path to the main folder.

**FOLDER_PARENT (str)**:	Path to the parent folder.

**CST_NAME (str)**:	Constraint name.

**WT (str)**:	Wild type information.

**LIGAND (str)**:	Ligand data.

**DESIGN (str)**:	Design specifications.

**MAX_JOBS (int)**:	Maximum number of jobs to run concurrently.

**N_PARENT_JOBS (int)**:	Number of parent jobs.

**MAX_DESIGNS (int)**:	Maximum number of designs.

**KBT_BOLTZMANN (list)**:	Boltzmann constant values.

**CST_WEIGHT (float)**:	Constraint weight.

**ProteinMPNN_PROB (float)**:	Probability parameter for ProteinMPNN.

**ProteinMPNN_BIAS (float)**:	Bias parameter for ProteinMPNN.

**LMPNN_PROB (float)**:	Probability parameter for LMPNN.

**FOLDER_MATCH (str)**:	Path to match folder.

**ProteinMPNN_T (str)**:	Temperature for ProteinMPNN.

**LMPNN_T (str)**:	Temperature for LMPNN.

**LMPNN_BIAS (float)**:	Bias parameter for LMPNN.

**SUBMIT_PREFIX (str)**:	Submission prefix.

**SYSTEM (str)**:	System information.

**MATCH (str)**:	Match specifications.

**ROSETTA_PATH (str)**:	Path to Rosetta source.

**EXPLORE (bool)**:	Whether to explore parameter space.

**FIELD_TOOLS (str)**:	Path to FieldTools script.

**LOG (str)**:	Logging level.

**PARENT_DES_MED (str)**:	Parent design method.



### Function: initialize

Initializes AIzymes with given parameters.

**Parameters:**

**FOLDER_HOME (str)**:	Path to the main folder.

**UNBLOCK_ALL (bool)**:	Flag to unblock all processes.

**PRINT_VAR (bool)**:	Flag to print variables.

**PLOT_DATA (bool)**:	Flag to plot data.

**LOG (str)**:	Logging level.



### Function: controller

Controls the AIzymes project based on scoring and normalization parameters.

**Parameters:**

**HIGHSCORE (float)**:	High score threshold for evaluation.

**NORM (dict)**:	Normalization values for different scores.



### Function: plot

Generates plots based on AIzymes data, including main, tree, and landscape plots.

**Parameters:**

**main_plots (bool)**:	Flag to generate main plots.

**tree_plot (bool)**:	Flag to generate tree plot.

**landscape_plot (bool)**:	Flag to generate landscape plot.

**print_vals (bool)**:	Flag to print values on plots.

**NORM (dict)**:	Normalization values for different scores.

**HIGHSCORE_NEGBEST (dict)**:	High score and negative best score for different metrics.


