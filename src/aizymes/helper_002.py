"""
Contains utility functions and supporting routines used across multiple modules
within the AIzymes project.

Functions:
    normalize_scores():                     Normalizes scores for each selected score type in a DataFrame.
    one_to_three_letter_aa():               Converts a one-letter amino acid code to its corresponding three-letter code.
    run_command():                          Executes a shell command with optional output capturing and error handling.
    load_main_variables():                  Loads main configuration variables from a JSON file into the project.
    save_main_variables():                  Saves current project variables to a JSON file.
    submit_job():                           Submits a computational job using system-specific commands.
    sequence_from_pdb():                    Extracts the amino acid sequence from a PDB file.
    generate_remark_from_all_scores_df():   Generates REMARK annotations from catalytic residue data.
    save_cat_res_into_all_scores_df():      Extracts catalytic residue indices and names from a PDB file and saves them into the scores DataFrame.
    reset_to_after_parent_design():         Resets folders and updates the scores DataFrame after completing parent designs.
    reset_to_after_index():                 Removes all design entries and corresponding folders beyond a specified index.
    save_all_scores_df():                   Atomically saves the scores DataFrame to a CSV file.
    get_best_structures():                  Identifies the best structure designs, generates plots, and archives selected structures.
    remove_intersection_best_structures():  Removes overlapping structure files between designated best-structure folders.
    trace_mutation_tree():                  Traces the mutation pathway through design generations and plots score progression.
    print_average_scores():                 Prints the average scores of designs (if applicable).
    wait_for_file():                        Waits for a specified file to exist and reach a non-zero size.
    hamming_distance():                     Computes the Hamming distance between two sequences.
    exponential_func():                     Evaluates an exponential decay function.
    
Modules Required:
    setup_system_001
"""

import os
import sys
import time
import json
import shutil
import logging
import numpy as np
import pandas as pd
import subprocess
import tempfile
import tarfile
from Bio import SeqIO
import matplotlib.pyplot as plt
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)
from Bio import BiopythonParserWarning
warnings.simplefilter('ignore', BiopythonParserWarning)
import re

from setup_system_001              import *

def check_running_jobs(self):
    """
    The check_running_job function returns the number of parallel jobs that are running by counting how many lines in the qstat output are present which correspond to the
    job prefix. This is true when working on the GRID, for the other system the same concept is being used but the terminology differs.

    Returns:
        int: Number of running jobs for the specific system.
    """

    num_running_jobs={}
    
    # Clean list of processes
    if hasattr(self, 'processes'):
        for p, out_file, err_file in self.processes:
            if p.poll() is not None: # Close finished files
                out_file.close()
                err_file.close()
        self.processes = [(p, out_file, err_file) for p, out_file, err_file in self.processes if p.poll() is None]

    # Count number of running GPU jobs
    if self.MAX_GPUS > 0:
        for gpu, proc in self.gpus.items():
            if proc is not None and proc.poll() is not None:
                self.gpus[gpu] = None
        num_running_jobs['running_gpu'] = sum(1 for value in self.gpus.values() if value is not None)
        
    # Count total number of jobs (rows)
    num_running_jobs['all'] = len(self.all_scores_df)
    
    # Count number of failed jobs
    num_running_jobs['failed'] = int((self.all_scores_df['blocked'] == 'failed').sum())
    
    # Count number of completed jobs (where next_steps is NaN)
    mask = (self.all_scores_df['next_steps'].isna() | self.all_scores_df['next_steps'].eq("")) & self.all_scores_df[f'{self.SELECTED_SCORES[0]}_score'].notna()
    num_running_jobs['completed'] = int(mask.sum())
    
    # Count number of scheduled jobs
    num_running_jobs['scheduled'] = int(num_running_jobs['all'] - num_running_jobs['completed'] - num_running_jobs['failed'])
    
    # Count number of running jobs (not unblocked and not failed)
    num_running_jobs['running'] = int((self.all_scores_df['blocked'] != 'unblocked').sum()) - int(num_running_jobs['failed'])

    logging.debug(f"num_running_jobs: {num_running_jobs}")

    return num_running_jobs
    
def normalize_scores(self, 
                     unblocked_all_scores_df, 
                     norm_all=False, # True is min max normalization, False is Z-score normalization
                     extension="score"):
    """
    Normalizes scores for each selected score type from the provided DataFrame.

    Parameters:
        unblocked_all_scores_df (df):   DataFrame containing the unnormalized score columns.
        norm_all (bool):                If True, use min-max normalization based on predefined ranges; if False, use Z-score normalization.
        extension (str):                Suffix appended to the score keys (defaults to "score").
    Returns:
        dict:                           A dictionary containing the normalized score arrays for each score type.
    """

    def neg_norm_array(array, score_type):

        if len(array) <= 1: return [1] # check that it's not only one value
                        
        array = -array

        # Normalize using predefined min max range in self.NORM
        if norm_all:
            array = (array-self.NORM[score_type][0])/(self.NORM[score_type][1]-self.NORM[score_type][0])
            return array

        normalization_array = array.copy()
        
        #import matplotlib.pyplot as plt
        #print('XXXXXXXXXXXXXXXXXXX raw data', score_type, array.mean(), array.std())
        #plt.show()
        #plt.scatter(range(len(array)),array)
        #plt.show()

        # For identical score, remove parents because they all have the same score
        if score_type == 'identical':
            if len(normalization_array) > self.N_PARENTS:
                normalization_array = normalization_array[self.N_PARENTS:] 
            else:
                return array # just return the values in the identical array

        # Take only NORM_BY_LAST items
        if isinstance(self.NORM_BY_LAST, int):    
            if len(normalization_array) > self.NORM_BY_LAST:
                normalization_array = normalization_array[-self.NORM_BY_LAST:]   
                
        p_low, p_high = np.percentile(normalization_array, [5, 95])
        normalization_array = np.clip(normalization_array, p_low, p_high)
        mean = normalization_array.mean()
        std  = normalization_array.std()

        if std == 0:
            array = array - mean
        else:            
            array = (array - mean) / std
            
        #import matplotlib.pyplot as plt
        #print('XXXXXXXXXXXXXXXXXXX normalized data', score_type, array.mean(), array.std())
        #plt.show()
        #plt.scatter(range(len(array)),array)
        #plt.ylim(-1,1)
        #plt.show()
        
        return array
        
    # Normalize and stack normalized scores in combined_scores
    scores = {}
    for score_type in self.SELECTED_SCORES:  
        scores[score_type] = unblocked_all_scores_df[f"{score_type}_{extension}"].to_numpy(dtype=np.float64)
        if score_type in ["efield", "identical"]: 
            scores[score_type] = -scores[score_type] # Adjust scale so that more negative is better for all score types

        normalized_scores = neg_norm_array(scores[score_type], f"{score_type}_{extension}")

        scores[f'{score_type}_{extension}'] = normalized_scores 

    # Weight scores
    weight_sum = 0
    if "efield" in self.SELECTED_SCORES:     
        scores[f'efield_{extension}']     = [x * self.WEIGHT_EFIELD for x in scores[f'efield_{extension}']]
        weight_sum += self.WEIGHT_EFIELD
    if "total" in self.SELECTED_SCORES:      
        scores[f'total_{extension}']     = [x * self.WEIGHT_TOTAL for x in scores[f'total_{extension}']]
        weight_sum += self.WEIGHT_TOTAL
    if "redox" in self.SELECTED_SCORES:      
        scores[f'redox_{extension}']     = [x * self.WEIGHT_REDOX for x in scores[f'redox_{extension}']]
        weight_sum += self.WEIGHT_REDOX
    if "interface" in self.SELECTED_SCORES: 
        scores[f'interface_{extension}']     = [x * self.WEIGHT_INTERFACE for x in scores[f'interface_{extension}']]
        weight_sum += self.WEIGHT_INTERFACE

    # Calculate Final and Combined Score
    score_arrays = []
    for score_type in self.SELECTED_SCORES:
        if score_type not in ["catalytic","identical"]:  
            score_arrays.append(scores[f'{score_type}_{extension}'])
    final_scores = np.stack(score_arrays, axis=0)    
    final_scores = np.sum(final_scores, axis=0)
    scores[f'final_{extension}'] = final_scores / weight_sum

    if "identical" in self.SELECTED_SCORES:
        scores[f'identical_{extension}']     = [x * self.WEIGHT_IDENTICAL for x in scores[f'identical_{extension}']]
        weight_sum += self.WEIGHT_IDENTICAL
        score_arrays.append(scores[f'identical_{extension}'])

    combined_scores = np.stack(score_arrays, axis=0)
    combined_scores = np.sum(combined_scores, axis=0)
    scores[f'combined_{extension}'] = combined_scores / weight_sum
    
    return scores

def one_to_three_letter_aa(one_letter_aa):
    """
    Converts a one-letter amino acid code to its corresponding three-letter code in all uppercase.

    Parameters:
        one_letter_aa (str): A one-letter amino acid code.
    Returns:
        str: The corresponding three-letter amino acid code.
    """

    # Dictionary mapping one-letter amino acid codes to three-letter codes in all caps
    aa_dict = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }
    
    # Convert it to the three-letter code in all caps
    return aa_dict[one_letter_aa]

def run_command(command, cwd=None, capture_output=False):
    """
    Executes a command (typically for running Python scripts) with optional output capturing and error handling.

    Parameters:
        command (list of str):              The command to run, provided as a list of strings.
        cwd (str, optional):                The working directory in which to run the command.
        capture_output (bool, optional):    If True, captures standard output and error; otherwise, output is suppressed. Defaults to False.
    Returns:
        str:                                The standard output of the command, if captured.
    """

    try:
        # If capture_output is True, capture stdout and stderr
        if capture_output:
            result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=cwd)
        else:
            # If capture_output is False, suppress all output by redirecting to os.devnull
            with open(os.devnull, 'w') as fnull:
                result = subprocess.run(command, stdout=fnull, stderr=fnull, text=True, check=True, cwd=cwd)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{e.cmd}' failed with return code {e.returncode}")
        logging.error(e.stderr)
        #maybe rerun command here in case of efields
        raise
    except Exception as e:
        logging.error(f"An error occurred while running command: {command}")
        raise

def load_main_variables(self, FOLDER_HOME):
    """
    Loads main configuration variables from a JSON file and sets them as attributes of the provided object.

    Parameters:
        FOLDER_HOME (str): The path to the main folder where the variables JSON file is located.
    """

    self.VARIABLES_JSON  = f'{FOLDER_HOME}/variables.json'
    with open(self.VARIABLES_JSON, 'r') as f:
        variables = json.load(f)
    for key, value in variables.items():
        setattr(self, key, value)

def save_main_variables(self):
    """
    Saves main configuration variables from the object's attributes to a JSON file.
    """
    
    variables = self.__dict__.copy()
    for key in ['resource_log_df','all_scores_df','UNBLOCK_ALL','PRINT_VAR','PLOT_DATA','LOG','HIGHSCORE','NORM']:
        if key in variables:
            del variables[key]    
    with open(self.VARIABLES_JSON, 'w') as f: json.dump(variables, f, indent=4)

def sequence_from_pdb(pdb_in):
    """
    Extracts the amino acid sequence from a PDB file.

    Parameters:
        pdb_in (str):   The base path (without extension) to the PDB file.
    Returns:
        str:            The amino acid sequence extracted from the PDB file.
    """

    with open(f"{pdb_in}.pdb", "r") as f:
        for record in SeqIO.parse(f, "pdb-atom"):
            seq = str(record.seq)
    
    return seq

def generate_remark_from_all_scores_df(self, index):
    """
    Generates a remark string from catalytic residue data contained in the all_scores_df DataFrame for a given index.

    Parameters:
        index:  The index (row key) in the DataFrame for which to generate the remark.
    Returns:
        str:    A multi-line string containing REMARK lines for each catalytic residue.
    """

    remark = ''    
    cat_resns = self.all_scores_df.loc[self.all_scores_df['index']==index, 'cat_resn'].item()
    cat_resns = str(cat_resns).split(';')
    cat_resis = self.all_scores_df.loc[self.all_scores_df['index']==index, 'cat_resi'].item()
    cat_resis = [int(float(x)) for x in str(cat_resis).split(';')]
    
    remarks = []
    for idx, (cat_resi, cat_resn) in enumerate(zip(cat_resis, cat_resns), start=1):
        # Check if cat_resn is valid before generating remark
        if cat_resn and cat_resn.strip().lower() != 'nan':
            # Format the REMARK line only if resn is valid
            remarks.append(f'REMARK 666 MATCH TEMPLATE X {self.LIGAND}    0 MATCH MOTIF A {cat_resn}{str(cat_resi).rjust(5)}  {idx}  1')
        else:
            # Log a warning if a catalytic residue entry is skipped due to invalid name
            logging.warning(f"Index {index}: Skipping REMARK 666 generation for catalytic residue entry {idx} due to invalid residue name: '{cat_resn}'")
            
    return "\n".join(remarks)

def save_cat_res_into_all_scores_df(self, index, PDB_path, save_resn=True):
    
    """
    Finds catalytic residue indices and names from a PDB file and saves them into the all_scores_df DataFrame at the specified index.

    Parameters:
        index:                      The DataFrame row index where catalytic residue information should be stored.
        PDB_path (str):             The path (without extension) to the PDB file.
    Optional Paramters:
        save_resn (bool, optional): If True, saves the residue names; otherwise, only saves indices.
    """

    with open(f'{PDB_path}.pdb', 'r') as f: 
        PDB = f.readlines()
    
    remarks = [i for i in PDB if i[:10] == 'REMARK 666']

    cat_resis = []
    cat_resns = []

    for remark in remarks:
        cat_resis.append(str(int(remark[55:59])))

    for cat_resi in cat_resis:
        for line in PDB[len(remarks)+2:]:
            atomtype = line[12:16]
            if atomtype != " CA ": continue
            resi = str(int(line[22:26]))
            resn = line[17:20]
            if resi == cat_resi:
                cat_resns.append(resn)
                break
    self.all_scores_df.at[index, 'cat_resi'] = ";".join(cat_resis)
    
    # Save resn only if enabled
    if save_resn:
        self.all_scores_df['cat_resn'] = self.all_scores_df['cat_resn'].astype(str)
        self.all_scores_df.at[index, 'cat_resn'] = ";".join(cat_resns)

def reset_to_after_parent_design():
    
    folders = []
    
    for folder_name in os.listdir(FOLDER_HOME):
        if os.path.isdir(os.path.join(FOLDER_HOME, folder_name)) and folder_name.isdigit():
            folders.append(int(folder_name))
    
    all_scores_df = make_empty_all_scores_df()
        
    PARENTS = [i for i in os.listdir(f'{FOLDER_HOME}/{FOLDER_PARENT}') if i[-4:] == ".pdb"]
    
    for folder in sorted(folders):
        
        folder_path = os.path.join(FOLDER_HOME, str(folder))
        
        if folder >= N_PARENT_JOBS * len(PARENTS):
            
            #Remove non-parent designs
            shutil.rmtree(folder_path)
            
        else:
            
            #Remove Potentials
            for item in os.listdir(folder_path):
                if 'potential.dat' not in item: continue
                item_path = os.path.join(folder_path, item)
                os.remove(item_path)
                print(item_path)
                    
            #Update Scorefile
            new_index, all_scores_df = create_new_index(parent_index="Parent", all_scores_df=all_scores_df)
            all_scores_df['design_method'] = all_scores_df['design_method'].astype('object') 
            all_scores_df.at[new_index, 'design_method'] = "RosettaDesign"
            all_scores_df['luca'] = all_scores_df['luca'].astype('object') 
            score_file_path = f"{FOLDER_DESIGN}/{int(index)}/score_rosetta_design.sc"
            with open(score_file_path, 'r') as f: score = f.readlines()[2]
            all_scores_df.at[new_index, 'luca'] = score.split()[-1][:-5]
    
            if new_index % 100 == 0: print(folder, new_index) 

    save_all_scores_df(all_scores_df)

def reset_to_after_index(index):
    '''This function resets the run back to a chosen index. It removes all later entries from the all_scores.csv and the home dir.
    index: The last index to keep, after which everything will be deleted.'''
    
    folders = []
    
    for folder_name in os.listdir(FOLDER_HOME):
        if os.path.isdir(os.path.join(FOLDER_HOME, folder_name)) and folder_name.isdigit():
            folders.append(int(folder_name))
    
    # Load the existing all_scores_df
    all_scores_df = pd.read_csv(ALL_SCORES_CSV)
    
    # Filter out rows with index greater than the specified index
    all_scores_df = all_scores_df[all_scores_df['index'] <= index]
    
    # Save the updated all_scores_df
    save_all_scores_df(all_scores_df)
    
    for folder in sorted(folders):
        if folder > index:
            folder_path = os.path.join(FOLDER_HOME, str(folder))
            shutil.rmtree(folder_path)
    
    print(f"Reset completed. All entries and folders after index {index} have been removed.")

def save_all_scores_df(self):
   
    temp_fd, temp_path = tempfile.mkstemp(dir=self.FOLDER_HOME) # Create a temporary file

    try:
        self.all_scores_df.to_csv(temp_path, index=False)  # Save DataFrame to the temporary file
        os.close(temp_fd)                                  # Close file descriptor
        os.rename(temp_path, self.ALL_SCORES_CSV)          # Rename temporary file to final filename
    except Exception as e:
        os.close(temp_fd)                                  # Ensure file descriptor is closed in case of error
        os.unlink(temp_path)                               # Remove the temporary file if an error occurs
        raise e
            
def save_resource_log_df(self):
   
    temp_fd, temp_path = tempfile.mkstemp(dir=self.FOLDER_HOME) # Create a temporary file

    try:
        self.resource_log_df.to_csv(temp_path, index=False)  # Save DataFrame to the temporary file
        os.close(temp_fd)                                    # Close file descriptor
        os.rename(temp_path, self.RESOURCE_LOG_CSV)          # Rename temporary file to final filename
    except Exception as e:
        os.close(temp_fd)                                  # Ensure file descriptor is closed in case of error
        os.unlink(temp_path)                               # Remove the temporary file if an error occurs
        raise e

def get_best_structures(self):

    print('SEQ_PER_ACTIVE_SITE: number of variants per unique active site sequence.')
    print('N_HITS:              number of top variants to select in total for each score type.')

          
    statistics = ""
    
    # Check for the CSV file; exit if not found.
    if not os.path.isfile(f'{self.FOLDER_HOME}/all_scores.csv'):
        print(f"ERROR: {self.FOLDER_HOME}/all_scores.csv does not exist!")
        sys.exit()
    
    # Read the CSV into unique_scores_df and drop rows with missing total_score.
    unique_scores_df = pd.read_csv(self.ALL_SCORES_CSV)
    unique_scores_df = unique_scores_df.dropna(subset=['total_score']).copy()
    statistics += f'All designs: {len(unique_scores_df)}'

    # Calculate normalized scores using your normalize_scores function.
    scores = normalize_scores(self, unblocked_all_scores_df=unique_scores_df, norm_all=False)
    for score in scores:
        unique_scores_df[score] = scores[score]
        
    # Remove bad structures below 3 standard deviations
    for score in ['final']+[i for i in self.SELECTED_SCORES if i not in ['identical','catalytic']]:
        score = f'{score}_score'
        mean = unique_scores_df[score].mean()
        std = unique_scores_df[score].std()
        unique_scores_df = unique_scores_df[(unique_scores_df[score] >= mean - 3 * std)]

    # Compute group statistics based on final_score.
    unique_scores_df['final_score_replicate_mean'] = unique_scores_df.groupby('sequence')['final_score'].transform('mean')
    unique_scores_df['final_score_replicate_std'] = unique_scores_df.groupby('sequence')['final_score'].transform('std')
    
    # Create best_scores_df by sorting on final_score and keeping only one variant per sequence.
    unique_scores_df = unique_scores_df.sort_values('final_score', ascending=False).drop_duplicates(subset=['sequence'], keep='first')
    statistics += f' - Unique designs: {len(unique_scores_df)}'

    best_scores_df = unique_scores_df.copy()
    
    # Get active site sequences
    if self.SEQ_PER_ACTIVE_SITE is not None:
        
        def get_active_site_sequence(sequence, active_site_position):
            return ''.join(sequence[pos - 1] for pos in active_site_position)
            
        if self.ACTIVE_SITE is None: self.ACTIVE_SITE = self.DESIGN # Default to self.DESIGN if self.ACTIVE_SITE is not given
        active_site_position = [int(pos) for pos in self.ACTIVE_SITE.split(',')]
        best_scores_df['active_site_sequence'] = best_scores_df['sequence'].apply(lambda seq: get_active_site_sequence(seq, active_site_position))

    # Filter dataframe for each SELECTED_SCORES
    filtered_best_scores_df = []
    for score in ['final']+[i for i in self.SELECTED_SCORES if i not in ['identical','catalytic']]:
    
        score = f'{score}_score'
        score_df = best_scores_df.sort_values(score, ascending=[False])

        # Filter by active site sequence
        if self.SEQ_PER_ACTIVE_SITE is not None:
    
            # Filter out best variants
            top_variants = []
            group_counts = {}
            for _, row in score_df.iterrows():
                group = row['active_site_sequence']
                group_counts[group] = group_counts.get(group, 0)
                if group_counts[group] < self.SEQ_PER_ACTIVE_SITE:
                    top_variants.append(row)
                    group_counts[group] += 1
                if len(top_variants) >= self.N_HITS:
                    break
                    
            filtered_best_scores_df.append(pd.DataFrame(top_variants))
            
        else:
            
            filtered_best_scores_df.append(score_df.head(self.N_HITS))

    best_scores_df = pd.concat(filtered_best_scores_df).drop_duplicates()
    statistics += f' - Filtered designs: {len(best_scores_df)}'
    print(statistics)
 
    # Plotting setup.
    score_types = [i for i in self.SELECTED_SCORES if i not in ['identical','catalytic']]
    nplots = len(score_types)
    fig, axes = plt.subplots(1, nplots, figsize=(nplots * self.PLOT_SIZE, self.PLOT_SIZE))
    axes = np.atleast_1d(axes)

    # For each score type, build plots.
    for idx, score in enumerate(score_types):
        score_col = f'{score}_score'   
        score_label = score_col.replace("_", " ")
        axes[idx].scatter(unique_scores_df['final_score'], unique_scores_df[score_col], s=5, c="grey", label="all")
        axes[idx].scatter(best_scores_df['final_score'], best_scores_df[score_col], s=5, c="b", label="best")
        final_label = "final score"  # replacing final_score with "final score"
        axes[idx].set_xlabel(final_label)
        axes[idx].set_ylabel(score_label)
        axes[idx].set_title(f"{score_label} vs. {final_label}".replace("_", " "))
    
    plt.tight_layout()
    save_path = os.path.join(self.FOLDER_PLOT, "selected_variants.png")
    fig.savefig(save_path, bbox_inches='tight')
    plt.show()
    
    print(statistics)
    
    # Save best structures.
    print("Saving best structures...")
    base_folder = os.path.join(os.getcwd(), os.path.basename(self.FOLDER_HOME))
    os.makedirs(base_folder, exist_ok=True)
    best_structures_folder = os.path.join(base_folder, 'best_structures')
    os.makedirs(best_structures_folder, exist_ok=True)
    
    # Clean the best structures directory.
    for file in os.listdir(best_structures_folder):
        file_path = os.path.join(best_structures_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # For each best variant, copy the appropriate file (relax or design).
    for index, row in best_scores_df.iterrows():
        geom_mean = "{:.3f}".format(row['final_score_replicate_mean'])
        relax_file = f"{self.FOLDER_DESIGN}/{int(index)}/{self.WT}_RosettaRelax_{int(index)}.pdb"
        design_file = f"{self.FOLDER_DESIGN}/{int(index)}/{self.WT}_RosettaDesign_{int(index)}.pdb"
        if os.path.isfile(relax_file):
            src_file = relax_file
        else:
            src_file = design_file
        dest_file = os.path.join(best_structures_folder, f"{geom_mean}_{self.WT}_{index}.pdb")
        shutil.copy(src_file, dest_file)
    
    # Save best scores CSV.
    csv_path = os.path.join(best_structures_folder, "best_scores.csv")
    best_scores_df.to_csv(csv_path, index=False)
    
    # Create tar archive of best structures.
    tar_path = os.path.join(base_folder, 'best_structures', f'{os.path.basename(self.FOLDER_HOME)}_best_structures.tar')
    folder_to_tar = os.path.join(base_folder, 'best_structures')
    with tarfile.open(tar_path, "w") as tar:
        for file_name in os.listdir(folder_to_tar):
            file_path = os.path.join(folder_to_tar, file_name)
            tar.add(file_path, arcname=file_name)
    
    print("Saved structures to:", best_structures_folder)
            
def remove_intersection_best_structures():
    # Define the paths to the folders
    best_structures_folder = os.path.join(FOLDER_HOME, 'best_structures')
    best_structures_nocat_folder = os.path.join(FOLDER_HOME, 'best_structures_nocat')

    # Get the list of files in both folders
    best_structures_files = [f for f in os.listdir(best_structures_folder) if os.path.isfile(os.path.join(best_structures_folder, f))]
    best_structures_nocat_files = [f for f in os.listdir(best_structures_nocat_folder) if os.path.isfile(os.path.join(best_structures_nocat_folder, f))]

    # Extract the structure names (numbers before .pdb) from both folders
    best_structures_names = {file.split('_')[-1] for file in best_structures_files}
    best_structures_nocat_names = {file.split('_')[-1] for file in best_structures_nocat_files}

    # Find the intersection of structure names
    intersection_names = best_structures_names.intersection(best_structures_nocat_names)

    # Remove the overlapping structures from best_structures_nocat
    intersect_count = 0
    for file in best_structures_nocat_files:
        structure_name = file.split('_')[-1]
        if structure_name in intersection_names:
            intersect_count += 1
            os.remove(os.path.join(best_structures_nocat_folder, file))
            print(f"Removed {file}.")

    print(f"Removed {intersect_count} structures.")

def trace_mutation_tree(all_scores_df, index):
    mutations = []
    offspring_counts = []
    combined_scores = []
    total_scores = []
    interface_scores = []
    efield_scores = []
    generations = []

    all_scores_df = all_scores_df.dropna(subset=['total_score'])
    
    # Calculate combined scores using normalized scores
    scores = normalize_scores(all_scores_df, norm_all=True)
    combined_scores_normalized = scores['combined_score']
    
    # Add combined scores to the DataFrame
    all_scores_df['combined_score'] = combined_scores_normalized

    # Cast index column to int
    all_scores_df['index'] = all_scores_df['index'].astype(int)
    all_scores_df['parent_index'] = all_scores_df['parent_index'].apply(lambda x: int(x) if x != "Parent" else x)

    def get_mutations(parent_seq, child_seq):
        return [f"{p}{i+1}{c}" for i, (p, c) in enumerate(zip(parent_seq, child_seq)) if p != c]

    def count_offspring(all_scores_df, parent_index):
        children = all_scores_df[all_scores_df['parent_index'] == parent_index]
        count = len(children)
        for child_index in children['index']:
            count += count_offspring(all_scores_df, child_index)
        return count

    total_variants = len(all_scores_df)
    total_mutations = int(all_scores_df.loc[all_scores_df['index'] == index, 'mutations'].values[0])
    current_index = index
    accumulated_mutations = 0

    while current_index in all_scores_df['index'].values:
        current_row = all_scores_df[all_scores_df['index'] == current_index].iloc[0]
        parent_index = current_row['parent_index']
        
        if parent_index in all_scores_df['index'].values:
            parent_row = all_scores_df[all_scores_df['index'] == parent_index].iloc[0]
            parent_seq = parent_row['sequence']
            child_seq = current_row['sequence']
            mutation = get_mutations(parent_seq, child_seq)
            offspring_count = count_offspring(all_scores_df, parent_index)
            
            mutations.append(mutation)
            offspring_counts.append(offspring_count)
            generations.append(current_row['generation'])
            
            # Store actual scores
            combined_scores.append(current_row['combined_score'])
            total_scores.append(current_row['total_score'])
            interface_scores.append(current_row['interface_score'])
            efield_scores.append(current_row['efield_score'])
        
        current_index = parent_index

    # Plot the actual scores
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    def plot_scores(ax, scores, title):
        ax.plot(generations[::-1], scores[::-1], marker='o', linestyle='-', color='b')
        ax.set_title(title)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Score')
        ax.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

    plot_scores(axs[0, 0], combined_scores, 'Combined Score vs Generations')
    plot_scores(axs[0, 1], total_scores, 'Total Score vs Generations')
    plot_scores(axs[1, 0], interface_scores, 'Interface Score vs Generations')
    plot_scores(axs[1, 1], efield_scores, 'Efield Score vs Generations')

    plt.tight_layout()
    plt.show()

    return mutations[::-1], offspring_counts[::-1], combined_scores[::-1], total_scores[::-1], interface_scores[::-1], efield_scores[::-1]

def wait_for_file(file_path, timeout=5):
    """Wait for a file to exist and have a non-zero size."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return True
        time.sleep(0.1)  # Wait for 0.1 seconds before checking again
    return False

#Define the hamming distance function and other required functions
def hamming_distance(seq1, seq2):
    #Ensures that seq2 is a string
    if not isinstance(seq2, str):
        return None
     #Ensures that the current and predecessor sequence length is equal
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")
    #Returns the number of differences between the current sequence and the parent sequence.
    return sum(ch1 != ch2 for ch1, ch2 in zip(seq1, seq2))

def exponential_func(x, A, k, c):
    return c-A*np.exp(-k * x)

def create_new_index(self, 
                     parent_index: str, 
                     luca: str, 
                     input_variant: str,
                     next_steps: str,
                     final_method: str,
                     design_method: str):
      
    """
    The create_new_index function is responsible for generating a new index.
    It adds a new row to all_scores_df and makes the folders for the index.

    Parameters:
        parent_index (str): The index of the parent variant for designs.
        luca (str): 
        design_method (str): 
        next_steps (list): 
    """

    # Get new index
    if self.all_scores_df.empty:
        new_index = 0  
    else:
        new_index = self.all_scores_df['index'].max() + 1 

    if parent_index == 'Parent':
        generation = 0
    else:
        generation = self.all_scores_df['generation'][int(parent_index)]+1

    # Defines the values of the kbt_boltzmann and cst_weight based on the generation of the new index
    if isinstance(self.KBT_BOLTZMANN, (float, int)):
        kbt_boltzmann = self.KBT_BOLTZMANN
    elif len(self.KBT_BOLTZMANN) == 2:
        kbt_boltzmann = self.KBT_BOLTZMANN[0] * np.exp(-self.KBT_BOLTZMANN[1]*generation)
    elif len(self.KBT_BOLTZMANN) == 3:
        kbt_boltzmann = (self.KBT_BOLTZMANN[0] - self.KBT_BOLTZMANN[2]) * np.exp(-self.KBT_BOLTZMANN[1]*generation)+self.KBT_BOLTZMANN[2]
        
    # Determines the cst_weight of the new index
    if isinstance(self.CST_WEIGHT, (float, int)):
        cst_weight = self.CST_WEIGHT
    elif len(self.CST_WEIGHT) == 2:
        cst_weight = self.CST_WEIGHT[0]*np.exp(-self.CST_WEIGHT[1]*generation)
    elif len(self.CST_WEIGHT) == 3:
        cst_weight = (self.CST_WEIGHT[0] - self.CST_WEIGHT[2])*np.exp(-self.CST_WEIGHT[1]*generation) + self.CST_WEIGHT[2]

    final_variant = f'{self.FOLDER_DESIGN}/{new_index}/{self.WT}_{final_method}_{new_index}',

    # Creates a new dataframe with all the necessary columns for the new index, concatenates it with the existing all_scores dataframe, and saves it
    new_index_df = pd.DataFrame({
        'parent_index': parent_index,
        'index': new_index,
        'kbt_boltzmann': kbt_boltzmann,
        'cst_weight': cst_weight,
        'generation': generation,
        'luca': luca,
        'blocked': 'unblocked',
        'design_method': design_method,
        'next_steps': next_steps,
        'input_variant': input_variant,
        'step_input_variant': input_variant,
        'step_output_variant': input_variant,
        'final_variant': final_variant,
    }, index = [0] , dtype=object)  
    self.all_scores_df = pd.concat([self.all_scores_df, new_index_df], ignore_index=True)
    
    # Add catalytic residues
    save_cat_res_into_all_scores_df(self, new_index, input_variant, save_resn=True)    
    save_all_scores_df(self)
    
    # Create the folder for the new index
    os.makedirs(f"{self.FOLDER_DESIGN}/{new_index}/scripts", exist_ok=True)
    logging.debug(f"Child index {new_index} created for parent index {parent_index}.")

    return new_index

def count_mutations(seq1, seq2):
    """
    Counts the number of differing positions (mutations) between two sequences.

    Parameters:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.
    Returns:
        int:        The number of mutations.
    """

    mutations = sum(1 for a, b in zip(seq1, seq2) if a != b)
    return int(mutations)

def print_statistics_df(self):
    
    self.all_scores_df = pd.read_csv(self.ALL_SCORES_CSV)

    num_running_jobs = check_running_jobs(self)
    
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.max_columns", None)   

    if self.PRINT_NUMBERS:
        print(f"All designs: {num_running_jobs['all']}", end = " ")
        print(f"- Completed designs: {num_running_jobs['completed']}", end = " ") 
        print(f"- Scheduled designs: {num_running_jobs['scheduled']}", end = " ") 
        print(f"- Running designs: {num_running_jobs['running']}", end = " ") 
        print(f"- Failed designs: {num_running_jobs['failed']}")
    
    if self.PRINT_COLUMN:
        pd.set_option("display.max_rows", 4)
        display(self.all_scores_df)

    if self.PRINT_RUNNING:
        pd.set_option("display.max_rows", None)
        display(self.all_scores_df[self.all_scores_df[f'{self.SELECTED_SCORES[0]}_score'].isna()][['design_method', 'blocked', 'next_steps']])
    
    if self.PRINT_ROW:
        for col, value in self.all_scores_df.iloc[self.INDEX].items():
            print(f"{col:33}: {(str(value)[:100] + '...') if len(str(value)) > 100 else str(value)}")