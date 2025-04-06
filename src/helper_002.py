
"""
Contains utility functions and supporting routines used across multiple modules
within the AIzymes project.

Functions:
    - normalize_scores
    - one_to_three_letter_aa
    - run_command
    - load_main_variables
    - save_main_variables
    - submit_job
    - sequence_from_pdb
    - generate_remark_from_all_scores_df
    - save_cat_res_into_all_scores_df
    - reset_to_after_parent_design
    - reset_to_after_index
    - save_all_scores_df
    - get_best_structures
    - remove_intersection_best_structures
    - trace_mutation_tree
    - print_average_scores
    - wait_for_file
    - hamming_distance
    - exponential_func

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
from Bio import SeqIO
import matplotlib.pyplot as plt
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)
from Bio import BiopythonParserWarning
warnings.simplefilter('ignore', BiopythonParserWarning)
import re

from setup_system_001         import *

def normalize_scores(self, 
                     unblocked_all_scores_df, 
                     norm_all=False, # True is min max normalization, False is Z-score normalization
                     extension="score"):

    def neg_norm_array(array, score_type):

        if len(array) > 1:  ##check that it's not only one value
            
            array = -array
            
            if norm_all:

                # Normalize using predefined min max range in self.NORM
                array = (array-self.NORM[score_type][0])/(self.NORM[score_type][1]-self.NORM[score_type][0])
    
            else:
                
                # Normalize using mean and standard deviation
                if np.std(array) == 0:
                    array = np.where(np.isnan(array), array, 0.0)  # Handle case where all values are the same
                else:
                    array = (array - np.mean(array)) / np.std(array)

            return array
        
        else:
            # do not normalize if array only contains 1 value
            return [1]

    # Normalize and stack normalized scores in combined_scores
    scores = {}
    for score_type in self.SELECTED_SCORES:  
        scores[score_type] = unblocked_all_scores_df[f"{score_type}_{extension}"].to_numpy(dtype=np.float64)
        if score_type in ["efield", "identical"]: 
            scores[score_type] = -scores[score_type] # Adjust scale so that more negative is better for all score types

        normalized_scores = neg_norm_array(scores[score_type], f"{score_type}_{extension}")

        scores[f'{score_type}_{extension}'] = normalized_scores 

    if len(scores[f'{score_type}_{extension}']) == 0: 
        combined_scores = []
        final_scores = []
    else:
        score_arrays = []
        for score_type in self.SELECTED_SCORES:
            if score_type != "catalytic":  
                score_arrays.append(scores[f'{score_type}_{extension}'])
        final_scores = np.stack(score_arrays, axis=0)
        final_scores = np.mean(final_scores, axis=0)
        scores[f'final_{extension}'] = final_scores

        if "catalytic" in self.SELECTED_SCORES:
            score_arrays.append(scores[f'{score_type}_{extension}'])
        combined_scores = np.stack(score_arrays, axis=0)
        combined_scores = np.mean(combined_scores, axis=0)
        scores[f'combined_{extension}'] = combined_scores
                    
    return scores

def one_to_three_letter_aa(one_letter_aa):
    
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
    """Wrapper to execute .py files in runtime with arguments, and print error messages if they occur.
    
    Parameters:
    command: The command to run as a list of strings.
    cwd: Optional; The directory to execute the command in.
    capture_output: Optional; If True, capture stdout and stderr. Defaults to False (This is to conserve memory).
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
    
    self.VARIABLES_JSON  = f'{FOLDER_HOME}/variables.json'
    with open(self.VARIABLES_JSON, 'r') as f:
        variables = json.load(f)
    for key, value in variables.items():
        setattr(self, key, value)
        
def save_main_variables(self):
    
    variables = self.__dict__.copy()
    for key in ['resource_log_df','all_scores_df','UNBLOCK_ALL','PRINT_VAR','PLOT_DATA','LOG','HIGHSCORE','NORM']:
        if key in variables:
            del variables[key]    
    with open(self.VARIABLES_JSON, 'w') as f: json.dump(variables, f, indent=4)
        
def submit_job(self, index, job, ram=16, bash=False):        
              
    submission_script = submit_head(self, index, job, ram)

    submission_script += f"""
# Output folder
cd {self.FOLDER_DESIGN}/{index}
pwd
bash {self.FOLDER_DESIGN}/{index}/scripts/{job}_{index}.sh
""" 

    # Create the submission_script
    with open(f'{self.FOLDER_DESIGN}/{index}/scripts/submit_{job}_{index}.sh', 'w') as file: file.write(submission_script)
    
    if bash:
        
        # Bash the submission_script for testing
        with open(f'{self.FOLDER_HOME}/n_running_jobs.dat') as f: cpu_id = int(f.read())-1
        subprocess.run(f'taskset -c {cpu_id} bash {self.FOLDER_DESIGN}/{index}/scripts/submit_{job}_{index}.sh', shell=True, text=True)

    elif self.RUN_PARALLEL:

        # Bash submission script parallel in background
        out_file = open(f"{self.FOLDER_DESIGN}/{index}/scripts/{job}_{index}.out", "w")
        err_file = open(f"{self.FOLDER_DESIGN}/{index}/scripts/{job}_{index}.err", "w")

        process = subprocess.Popen(f'bash -l -c "bash {self.FOLDER_DESIGN}/{index}/scripts/submit_{job}_{index}.sh"', 
                                   shell=True, 
                                   stdout=out_file, 
                                   stderr=err_file)
        self.processes.append((process, out_file, err_file)) 

        with open(f'{self.FOLDER_DESIGN}/{index}/scripts/{job}_{index}.sh', "r") as f: script = f.read()
        match = re.search(r'CUDA_VISIBLE_DEVICES\s*=\s*([0-9]+)', script)
        if match: 
            gpu = int(match.group(1))
            self.gpus[gpu] = process 
        
        logging.debug(f'Job started with {self.FOLDER_DESIGN}/{index}/scripts/submit_{job}_{index}.sh')  
      
    else:
        
        #Submit the submission_script
        if self.SYSTEM == 'GRID':
            if "ESM" in job:
                
                output = subprocess.check_output(
    (f'ssh $USER@bs-submit04.ethz.ch "qsub -l h=\'!bs-dsvr64&!bs-dsvr58&!bs-dsvr42&!bs-grid64&!bs-grid65&!bs-grid66&!bs-grid67&!bs-grid68&!bs-grid69&!bs-grid70&!bs-grid71&!bs-grid72&!bs-grid73&!bs-grid74&!bs-grid75&!bs-grid76&!bs-grid77&!bs-grid78&!bs-headnode04&!bs-stellcontrol05&!bs-stellsubmit05\' -q regular.q {self.FOLDER_DESIGN}/{index}/scripts/submit_{job}_{index}.sh"'),
    shell=True, text=True
)

            else:
                output = subprocess.check_output(f'ssh $USER@bs-submit04.ethz.ch qsub -q regular.q \
                                                {self.FOLDER_DESIGN}/{index}/scripts/submit_{job}_{index}.sh', \
                                                shell=True, text=True)
            logging.debug(output[:-1]) #remove newline at end of output
            
        elif self.SYSTEM == 'BLUEPEBBLE':
            output = subprocess.check_output(f'sbatch {self.FOLDER_DESIGN}/{index}/scripts/submit_{job}_{index}.sh', \
                                             shell=True, text=True)
            logging.debug(output[:-1]) #remove newline at end of output
            
        elif self.SYSTEM == 'BACKGROUND_JOB':

            stdout_log_file_path = f'{self.FOLDER_DESIGN}/{index}/scripts/submit_{job}_{index}_stdout.log'
            stderr_log_file_path = f'{self.FOLDER_DESIGN}/{index}/scripts/submit_{job}_{index}_stderr.log'

            with open(stdout_log_file_path, 'w') as stdout_log_file, open(stderr_log_file_path, 'w') as stderr_log_file:
                process = subprocess.Popen(f'bash {self.FOLDER_DESIGN}/{index}/scripts/submit_{job}_{index}.sh &', 
                                           shell=True, stdout=stdout_log_file, stderr=stderr_log_file)
        
        elif self.SYSTEM == 'ABBIE_LOCAL':

            stdout_log_file_path = f'{self.FOLDER_DESIGN}/{index}/scripts/submit_{job}_{index}_stdout.log'
            stderr_log_file_path = f'{self.FOLDER_DESIGN}/{index}/scripts/submit_{job}_{index}_stderr.log'

            with open(stdout_log_file_path, 'w') as stdout_log_file, open(stderr_log_file_path, 'w') as stderr_log_file:
                process = subprocess.Popen(f'bash {self.FOLDER_DESIGN}/{index}/scripts/submit_{job}_{index}.sh &', 
                                           shell=True, stdout=stdout_log_file, stderr=stderr_log_file)
            
        else:
            logging.error(f"ERROR! SYSTEM: {self.SYSTEM} not defined in submit_job() in helper.py.")
            sys.exit()

def sequence_from_pdb(pdb_in):
    
    with open(f"{pdb_in}.pdb", "r") as f:
        for record in SeqIO.parse(f, "pdb-atom"):
            seq = str(record.seq)
    
    return seq

def generate_remark_from_all_scores_df(self, index):

    remark = ''
    cat_resns = str(self.all_scores_df.at[index, 'cat_resn']).split(';')
    cat_resis = [int(float(x)) for x in str(self.all_scores_df.at[index, 'cat_resi']).split(';')]
    
    remarks = []
    for idx, (cat_resi, cat_resn) in enumerate(zip(cat_resis, cat_resns), start=1):
        remarks.append(f'REMARK 666 MATCH TEMPLATE X {self.LIGAND}    0 MATCH MOTIF A {cat_resn}{str(cat_resi).rjust(5)}  {idx}  1')
    return "\n".join(remarks)

def save_cat_res_into_all_scores_df(self, index, PDB_path, save_resn=True):
    
    '''Finds the indices and names of the catalytic residue from <PDB_file_path> 
       Saves indices and residues into <all_scores_df> in row <index> as lists.
       To make sure these are saved and loaded as list, ";".join() and .split(";") should be used
       If information is read from an input structure for design do not save cat_resn'''

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

    # Condition to check if the ALL_SCORES_CSV file exists, otherwise it returns the function.
    if not os.path.isfile(f'{self.FOLDER_HOME}/all_scores.csv'): 
        print(f"ERROR: {self.FOLDER_HOME}/all_scores.csv does not exist!")
        sys.exot()    
    
    all_scores_df = pd.read_csv(self.ALL_SCORES_CSV)
    best_scores_df = all_scores_df.dropna(subset=['total_score']).copy()
    
    # Calculate the final_scores
    scores = normalize_scores(self, unblocked_all_scores_df=best_scores_df, norm_all=False)
    for score in scores:
        best_scores_df[score] = scores[score]
    best_scores_df['final_score'] = best_scores_df[[f'{score}_score' for score in self.SELECTED_SCORES if score != 'identical']].mean(axis=1)
    best_scores_df['final_score_replicate_mean'] = best_scores_df.groupby('sequence')['final_score'].transform('mean')
    best_scores_df['final_score_replicate_std'] = best_scores_df.groupby('sequence')['final_score'].transform('std')

    # Remove replicates and keep only highest final
    best_scores_df.sort_values(by=['final_score'], ascending=[False], inplace=True)
    best_scores_df.drop_duplicates(subset=['sequence'], keep='first', inplace=True)

    # Filter by active site sequences
    if self.SEQ_PER_ACTIVE_SITE is not None:
        
        def get_active_site_sequence(sequence, active_site_position):
            return ''.join(sequence[pos - 1] for pos in active_site_position)
        
        # Get active_site_sequence
        if self.ACTIVE_SITE is None: self.ACTIVE_SITE = self.DESIGN # Default to self.DESIGN if self.ACTIVE_SITE is not given
        active_site_position = [int(pos) for pos in self.ACTIVE_SITE.split(',')]
        best_scores_df['active_site_sequence'] = best_scores_df['sequence'].apply(lambda seq: get_active_site_sequence(seq, active_site_position))

        # Filter out best variants
        top_variants = []
        group_counts = {}
        for _, row in best_scores_df.iterrows():
            group = row['active_site_sequence']
            group_counts[group] = group_counts.get(group, 0)
            if group_counts[group] < self.SEQ_PER_ACTIVE_SITE:
                top_variants.append(row)
                group_counts[group] += 1
            if len(top_variants) >= self.N_HITS:
                break
                
        best_scores_df = pd.DataFrame(top_variants)
        
    else:
        best_scores_df = best_scores_df.head(self.N_HITS)

    # Save best structures
    print("Saving...")
    best_structures_folder = os.path.join(self.FOLDER_HOME, 'best_structures')
    os.makedirs(best_structures_folder, exist_ok=True)

    # Clean directory
    for file in os.listdir(best_structures_folder):
        file_path = os.path.join(best_structures_folder, file)
        if os.path.isfile(file_path): os.remove(file_path)
    
    for index, row in best_scores_df.iterrows():

        # Define input filename 
        geom_mean = "{:.3f}".format(row['final_score_replicate_mean'])
        relax_file = f"{self.FOLDER_DESIGN}/{int(index)}/{self.WT}_RosettaRelax_{int(index)}.pdb"
        design_file = f"{self.FOLDER_DESIGN}/{int(index)}/{self.WT}_RosettaDesign_{int(index)}.pdb"
        if os.path.isfile(relax_file):
            src_file = relax_file
        else:
            src_file = design_file

        # Copy file
        dest_file = os.path.join(best_structures_folder, f"{geom_mean}_{self.WT}")
        shutil.copy(src_file, dest_file)
        
    csv_path = os.path.join(best_structures_folder, "best_scores.csv")
    best_scores_df.to_csv(csv_path, index=False)
    
    print("Saved structures and CSV to:", best_structures_folder)

            
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
        new_index = self.all_scores_df.index[-1] + 1 

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

    step_input_variant = input_variant
    
    step_output_variant = None
    for next_step in next_steps.split(","):
        if next_step in self.SYS_STRUCT_METHODS:
            step_output_variant = f'{self.FOLDER_DESIGN}/{new_index}/{self.WT}_{next_step}_{new_index}'
            break
    if step_output_variant == None:
        logging.error(f"ERROR! New design at index {new_index} initated, but next_steps: {next_steps} does not prodce any output structure!")
        sys.exit()   
        
    # Creates a new dataframe with all the necessary columns for the new index, concatenes it with the existing all_scores dataframe and saves it
    new_index_df = pd.DataFrame({
        'parent_index': parent_index,
        'kbt_boltzmann': kbt_boltzmann,
        'cst_weight': cst_weight,
        'generation': generation,
        'luca': luca,
        'blocked': 'unblocked',
        'design_method': design_method,
        'next_steps': next_steps,
        'input_variant': input_variant,
        'step_input_variant': step_input_variant,
        'step_output_variant': step_output_variant,
        'final_variant': final_variant,
    }, index = [0] , dtype=object)  
    self.all_scores_df = pd.concat([self.all_scores_df, new_index_df], ignore_index=True)
    
    # Add catalytic residues
    save_cat_res_into_all_scores_df(self, new_index, input_variant, save_resn=False)    
    save_all_scores_df(self)
    
    # Create the folder for the new index
    os.makedirs(f"{self.FOLDER_DESIGN}/{new_index}/scripts", exist_ok=True)
    logging.debug(f"Child index {new_index} created for parent index {parent_index}.")

    return new_index

def count_mutations(seq1, seq2):
    mutations = sum(1 for a, b in zip(seq1, seq2) if a != b)
    return int(mutations)