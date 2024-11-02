# design_ESMfold


### Function: prepare_ESMfold

Predicts structure of sequence in {index} using ESMfold.

**Parameters:**

**- index (str)**:	The index of the protein variant to be predicted.

**- cmd (str)**:	Growing list of commands to be exected by run_design using submit_job.


**Returns:**

**- cmd (str)**:	Command to be exected by run_design using submit_job.



# design_LigandMPNN


### Function: prepare_LigandMPNN

Executes the LigandMPNN pipeline for a given protein-ligand structure and generates
new protein sequences with potentially higher functional scores considering the ligand context.

**Parameters:**

**- parent_index (str)**:	The index of the parent protein variant.

**- new_index (str)**:	The index assigned to the new protein variant.

**- all_scores_df (DataFrame)**:	A DataFrame containing information for protein variants.



# design_match


# design_ProteinMPNN


### Function: prepare_ProteinMPNN

Executes the ProteinMPNN pipeline for a given protein structure and generates
new protein sequences with potentially higher functional scores.

**Parameters:**

**- new_index (str)**:	The index of the designed variant.

**- cmd (str)**:	Growing list of commands to be exected by run_design using submit_job.


**Returns:**

**- cmd (str)**:	Command to be exected by run_design using submit_job


Note:
This function assumes the ProteinMPNN toolkit is available and properly set up in the specified location.
It involves multiple subprocess calls to Python scripts for processing protein structures and generating new sequences.


# design_RosettaDesign


### Function: prepare_RosettaDesign

Designs protein structure in {new_index} based on {parent_index} using RosettaDesign.

**Parameters:**

**- parent_index (str)**:	Index of the parent protein variant to be designed.

**- new_index (str)**:	Index assigned to the resulting design.

**- input_suffix (str)**:	Suffix of the input structure to be used for design.


**Returns:**

**- cmd (str)**:	Command to be exected by run_design using submit_job.



# design_RosettaRelax


### Function: prepare_RosettaRelax

Relaxes protein structure in {index} using RosettaRelax.

**Parameters:**

**- index (str)**:	The index of the protein variant to be relaxed.

**- cmd (str)**:	collection of commands to be run, this script wil append its commands to cmd


Optional parameters:
- PreMatchRelax (bool): True if ESMfold to be run without ligand (prior to RosettaMatch).


# helper


### Function: run_command

Wrapper to execute .py files in runtime with arguments, and print error messages if they occur.

**Parameters:**

**- command**:	The command to run as a list of strings.

**- cwd**:	Optional; The directory to execute the command in.

**- capture_output**:	Optional; If True, capture stdout and stderr. Defaults to False (This is to conserve memory).



### Function: get_PDB_in

Based on index, find the input PDB files for the AIzymes modules

Paramters:
- index: The indix of the current design

Output:
- PDBfile_Design_in: Input file for RosettaDesign
- PDBfile_Relax_in: Input file for RosettaRelax
- PDBfile_Relax_ligand_in: Input file for Ligand to be used in RosettaRelax


### Function: save_cat_res_into_all_scores_df

Finds the indices and names of the catalytic residue from <PDB_file_path>
Saves indices and residues into <all_scores_df> in row <index> as lists.
To make sure these are saved and loaded as list, ";".join() and .split(";") should be used
If information is read from an input structure for design do not save cat_resn


### Function: reset_to_after_index

This function resets the run back to a chosen index. It removes all later entries from the all_scores.csv and the home dir.
index: The last index to keep, after which everything will be deleted.


### Function: wait_for_file

Wait for a file to exist and have a non-zero size.


# main_design


# main_running


### Function: update_potential

Creates a <score_type>_potential.dat file in FOLDER_HOME/<index>

If latest score comes from Rosetta Relax - then the score for variant <index>
will be added to the <score_type>_potential.dat of the parent index
and the <score_type>_potential value of the dataframe for the parent
will be updated with the average of the parent and child scores.

**Parameters:**

**- score_type(str)**:	Type of score to update, one of these options: total, interface, catalytic, efield

**- index (int)**:	variant index to update



# main_scripts


# main_startup


# plotting


### Function: plot_interface_v_total_score_selection

Plots a scatter plot of total_scores vs interface_scores and highlights the points
corresponding to the selected indices.

**Parameters:**

**- ax (matplotlib.axes.Axes)**:	The Axes object to plot on.

**- total_scores (list or np.array)**:	The total scores of the structures.

**- interface_scores (list or np.array)**:	The interface scores of the structures.

**- selected_indices (list of int)**:	Indices of the points to highlight.



### Function: plot_interface_v_total_score_generation

Plots a scatter plot of total_scores vs interface_scores and colors the points
according to the generation for all data points, using categorical coloring.
Adds a legend to represent each unique generation with its corresponding color.

**Parameters:**

**- ax (matplotlib.axes.Axes)**:	The Axes object to plot on.

**- total_scores (list or np.array)**:	The total scores of the structures.

**- interface_scores (list or np.array)**:	The interface scores of the structures.

**- generation (pd.Series or np.array)**:	Generation numbers for all data points.



### Function: plot_stacked_histogram_by_cat_resi

Plots a stacked bar plot of interface scores colored by cat_resi on the given Axes object,
where each bar's segments represent counts of different cat_resi values in that bin.

**Parameters:**

**- ax (matplotlib.axes.Axes)**:	The Axes object to plot on.

**- all_scores_df (pd.DataFrame)**:	DataFrame containing 'cat_resi' and 'interface_score' columns.

**- color_map (dict)**:	Optional; A dictionary mapping catalytic residue indices to colors.

**- show_legend (bool)**:	Optional; Whether to show the legend. Defaults to False.



### Function: plot_stacked_histogram_by_cat_resn

Plots a stacked bar plot of interface scores colored by cat_resn on the given Axes object,
where each bar's segments represent counts of different cat_resn values in that bin.

**Parameters:**

**- ax (matplotlib.axes.Axes)**:	The Axes object to plot on.

**- all_scores_df (pd.DataFrame)**:	DataFrame containing 'cat_resn' and 'interface_score' columns.



### Function: plot_stacked_histogram_by_generation

Plots a stacked bar plot of interface scores colored by generation on the given Axes object,
where each bar's segments represent counts of different generation values in that bin.

**Parameters:**

**- ax (matplotlib.axes.Axes)**:	The Axes object to plot on.

**- all_scores_df (pd.DataFrame)**:	DataFrame containing 'generation' and 'interface_score' columns.



# scoring_efields


### Function: generate_AMBER_files

Uses tleap to create a .parm7 and .rst7 file from a pdb. Requires ambertools.
Also requires 5TS.prepi and 5TS.frcmod in the INPUT folder
TODO: Add script to generate these if not present.

**Parameters:**

**- filename (str)**:	The path to the pdb file to analyse without the file extension.



### Function: calc_efields_score

Executes the FieldTools.py script to calculate the electric field across the C-H bond of 5TS.
Requires a field_target.dat in the Input folder. Currently hard-coded based on 5TS
TODO: Make this function agnostic to contents of field_target

**Parameters:**

**- pdb_path (str)**:	The path to the pdb structure to analyse - either from design or relax.


**Returns:**

**- bond_field (float)**:	The total electric field across the 5TS@C9_:5TS@H04 bond in MV/cm. (Currently hard coded to these atoms)

**- all_fields (dict)**:	The components of the electric field across the 5TS@C9_:5TS@H04 bond per residue.



### Function: update_efieldsdf

Adds a new row to "{FOLDER_HOME}/electric_fields.csv" containing the electric fields
generated by FieldTools.py for all residues in the protein


# setup_system

