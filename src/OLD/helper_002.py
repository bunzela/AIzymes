
"""
Contains utility functions and supporting routines used across multiple modules
within the AIzymes project.

Functions:
    - normalize_scores
    - one_to_three_letter_aa
    - run_command
    - get_PDB_in
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

from setup_system_001         import *

def normalize_scores(self, 
                     unblocked_all_scores_df, 
                     print_norm=False, 
                     norm_all=False, # True is min max normalization, False is Z-score normalization
                     extension="score"):
    
    def neg_norm_array(array, score_type):

        if len(array) > 1:  ##check that it's not only one value
            
            array    = -array
            
            if norm_all:
                if print_norm:
                    print(self.NORM[score_type])
                    #print(score_type,self.NORM[score_type],end=" ")
                array = (array-self.NORM[score_type][0])/(self.NORM[score_type][1]-self.NORM[score_type][0])
    
                if np.any(array > 1.0):
                    print(f"\nNORMALIZATION ERROR! {score_type} has a value >1! Max value is {max(array)}") 
                if np.any(array < 0.0):
                    print(f"\nNORMALIZATION ERROR! {score_type} has a value <0! Min value is {min(array)}")
            else:
                if print_norm:
                    print(score_type,[np.mean(array),np.std(array)],end=" ")
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
        scores[score_type] = unblocked_all_scores_df[f"{score_type}_{extension}"]
        if score_type in ["efield", "identical"]: 
            scores[score_type] = -scores[score_type] # Adjust scale so that more negative is better for all score types
        normalized_scores = neg_norm_array(scores[score_type], f"{score_type}_{extension}")
        globals()[f"{score_type}_scores"] = normalized_scores # Save normalized scores in arrays called scoretype_scores

    if len(total_scores) == 0: 
        combined_scores = []
    else:
        score_arrays = []
        for score_type in self.SELECTED_SCORES:
            if score_type != "catalytic":  
                score_arrays.append(globals()[f"{score_type}_scores"])
        combined_scores = np.stack(score_arrays, axis=0)
        combined_scores = np.mean(combined_scores, axis=0)
        
    if print_norm:
        if combined_scores.size > 0:
            print("HIGHSCORE:","{:.2f}".format(np.amax(combined_scores)),end=" ")
            print("Designs:",len(combined_scores),end=" ")
            parents = [i for i in os.listdir(self.FOLDER_PARENT) if i[-4:] == ".pdb"]
            print("Parents:",len(parents))
            
    return catalytic_scores, total_scores, interface_scores, efield_scores, identical_scores, combined_scores

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
        
def get_PDB_in(self, index):
    
    """Based on index, find the input PDB files for the AIzymes modules
    
    Parameters:
    - index: The index of the current design
    
    Output:
    - input_pdb_paths (dict): Contains relevant paths for input PDBs. Keys: ligand_in, Relax_in, Design_in and if MDMin = True, MDMin_in
    """
    input_pdb_paths = {}
    
    parent_index  = self.all_scores_df.loc[index, 'parent_index']
    design_method = self.all_scores_df.loc[index, 'design_method']
    
    # PDBfile_Relax_ligand_in
    if design_method == "ProteinMPNN":        
        if parent_index == "Parent":
            # Take ligand for relax run from Parent strucutre
            PDBfile_Relax_ligand_in = f"{self.FOLDER_PARENT}/{self.WT}"
        else:
            # Take ligand for relax run from parent_index RosettaRelaxed structure
            PDBfile_Relax_ligand_in  = f'{self.FOLDER_HOME}/{parent_index}/{self.WT}_RosettaRelax_{parent_index}'  

    else:
        # Take ligand for relax run from designed structure
        PDBfile_Relax_ligand_in  = f'{self.FOLDER_HOME}/{index}/{self.WT}_{design_method}_{index}'    
        
    # PDBfile_Relax_in 
    if self.MDMin:
        input_pdb_paths['MDMin_in'] = f'{self.FOLDER_HOME}/{index}/{self.WT}_ESMfold_{index}'  
        PDBfile_Relax_in = f'{self.FOLDER_HOME}/{index}/{self.WT}_MDMin_{index}' 
    
    else:
        PDBfile_Relax_in = f'{self.FOLDER_HOME}/{index}/{self.WT}_ESMfold_{index}'
    
    input_pdb_paths['ligand_in'] = PDBfile_Relax_ligand_in

    # PDBfile_Design_in
    if parent_index == "Parent":
        PDBfile_Design_in = f'{self.FOLDER_PARENT}/{self.WT}'
    else:
        PDBfile_Design_in = f'{self.FOLDER_HOME}/{parent_index}/{self.WT}_RosettaRelax_{parent_index}'

    if design_method == "ProteinMPNN": 
        if parent_index == "Parent":
            PDBfile_Design_seq_in = f'{self.FOLDER_PARENT}/{self.WT}'
        else:
            PDBfile_Design_seq_in = f'{self.FOLDER_HOME}/{parent_index}/{self.WT}_{parent_index}'
        input_pdb_paths['Design_seq_in'] = PDBfile_Design_seq_in
    
    input_pdb_paths['ligand_in'] = PDBfile_Relax_ligand_in
    input_pdb_paths['Relax_in'] = PDBfile_Relax_in
    input_pdb_paths['Design_in'] = PDBfile_Design_in
    
    return input_pdb_paths

def load_main_variables(self, FOLDER_HOME):
    
    self.VARIABLES_JSON  = f'{FOLDER_HOME}/variables.json'
    with open(self.VARIABLES_JSON, 'r') as f:
        variables = json.load(f)
    for key, value in variables.items():
        setattr(self, key, value)
        
def save_main_variables(self):
    
    variables = self.__dict__.copy()
    for key in ['all_scores_df','UNBLOCK_ALL','PRINT_VAR','PLOT_DATA','LOG','HIGHSCORE','NORM']:
        if key in variables:
            del variables[key]    
    with open(self.VARIABLES_JSON, 'w') as f: json.dump(variables, f, indent=4)
        
def submit_job(self, index, job, ram=16, bash=False):        
              
    submission_script = submit_head(self, index, job, ram)

    if self.RUN_PARALLEL: 
        submission_script = f"""
jobs=$(cat {self.FOLDER_HOME}/n_running_jobs.dat)
jobs=$((jobs + 1))
echo "$jobs" > {self.FOLDER_HOME}/n_running_jobs.dat
"""

    submission_script += f"""
# Output folder
cd {self.FOLDER_HOME}/{index}
pwd
bash {self.FOLDER_HOME}/{index}/scripts/{job}_{index}.sh
""" 
    
    if self.RUN_PARALLEL: 
        submission_script += f"""
jobs=$(cat {self.FOLDER_HOME}/n_running_jobs.dat)
jobs=$((jobs - 1))
echo "$jobs" > {self.FOLDER_HOME}/n_running_jobs.dat
"""
        
    # Create the submission_script
    with open(f'{self.FOLDER_HOME}/{index}/scripts/submit_{job}_{index}.sh', 'w') as file: file.write(submission_script)
    
    if bash:
        
        #Bash the submission_script for testing
        with open(f'{self.FOLDER_HOME}/n_running_jobs.dat') as f: cpu_id = int(f.read())-1
        subprocess.run(f'taskset -c {cpu_id} bash {self.FOLDER_HOME}/{index}/scripts/submit_{job}_{index}.sh', shell=True, text=True)

    elif self.RUN_PARALLEL:
        
        #Bash the submission_script in background using Popen to run jobs in parallel
        stdout_path = f"{self.FOLDER_HOME}/{index}/scripts/{job}_{index}.out"
        stderr_path = f"{self.FOLDER_HOME}/{index}/scripts/{job}_{index}.err"
        
        with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
            subprocess.Popen(
                f'bash {self.FOLDER_HOME}/{index}/scripts/submit_{job}_{index}.sh',
                shell=True,
                text=True,
                stdout=stdout_file,
                stderr=stderr_file
            )        
    else:
        
        #Submit the submission_script
        if self.SYSTEM == 'GRID':
            if "ESM" in job:
                
                output = subprocess.check_output(
    (f'ssh $USER@bs-submit04.ethz.ch "qsub -l h=\'!bs-dsvr64&!bs-dsvr58&!bs-dsvr42&!bs-grid64&!bs-grid65&!bs-grid66&!bs-grid67&!bs-grid68&!bs-grid69&!bs-grid70&!bs-grid71&!bs-grid72&!bs-grid73&!bs-grid74&!bs-grid75&!bs-grid76&!bs-grid77&!bs-grid78&!bs-headnode04&!bs-stellcontrol05&!bs-stellsubmit05\' -q regular.q {self.FOLDER_HOME}/{index}/scripts/submit_{job}_{index}.sh"'),
    shell=True, text=True
)

            else:
                output = subprocess.check_output(f'ssh $USER@bs-submit04.ethz.ch qsub -q regular.q \
                                                {self.FOLDER_HOME}/{index}/scripts/submit_{job}_{index}.sh', \
                                                shell=True, text=True)
            logging.debug(output[:-1]) #remove newline at end of output
            
        elif self.SYSTEM == 'BLUEPEBBLE':
            output = subprocess.check_output(f'sbatch {self.FOLDER_HOME}/{index}/scripts/submit_{job}_{index}.sh', \
                                             shell=True, text=True)
            logging.debug(output[:-1]) #remove newline at end of output
            
        elif self.SYSTEM == 'BACKGROUND_JOB':

            stdout_log_file_path = f'{self.FOLDER_HOME}/{index}/scripts/submit_{job}_{index}_stdout.log'
            stderr_log_file_path = f'{self.FOLDER_HOME}/{index}/scripts/submit_{job}_{index}_stderr.log'

            with open(stdout_log_file_path, 'w') as stdout_log_file, open(stderr_log_file_path, 'w') as stderr_log_file:
                process = subprocess.Popen(f'bash {self.FOLDER_HOME}/{index}/scripts/submit_{job}_{index}.sh &', 
                                           shell=True, stdout=stdout_log_file, stderr=stderr_log_file)
        
        elif self.SYSTEM == 'ABBIE_LOCAL':

            stdout_log_file_path = f'{self.FOLDER_HOME}/{index}/scripts/submit_{job}_{index}_stdout.log'
            stderr_log_file_path = f'{self.FOLDER_HOME}/{index}/scripts/submit_{job}_{index}_stderr.log'

            with open(stdout_log_file_path, 'w') as stdout_log_file, open(stderr_log_file_path, 'w') as stderr_log_file:
                process = subprocess.Popen(f'bash {self.FOLDER_HOME}/{index}/scripts/submit_{job}_{index}.sh &', 
                                           shell=True, stdout=stdout_log_file, stderr=stderr_log_file)
            
        else:
            logging.error(f"ERROR! SYSTEM: {self.SYSTEM} not defined in submit_job() in helper.py.")
            sys.exit()

def start_controller_parallel(self):

    with open(f"{self.FOLDER_HOME}/n_running_jobs.dat", "w") as f: f.write("0") ### set number of running jobs to 0

    cmd = f'''import sys, os
sys.path.append(os.path.join(os.getcwd(), '../../src'))
from AIzymes_014 import *
AIzymes = AIzymes_MAIN()
AIzymes.initialize(FOLDER_HOME    = '{os.path.basename(self.FOLDER_HOME)}', 
                   LOG            = '{self.LOG}',
                   CHECK_PARALLEL = False,
                   PRINT_VAR      = False)

with open("test_py.txt", "w") as f: f.write("AIzymes.initialized! \\n") ### CHECK TO SEE IF PYTHON IS RUNNING

AIzymes.controller()
'''
    with open(f"{self.FOLDER_HOME}/start_controller_parallel.py", "w") as f:
        f.write(cmd)

    ### Prepare submission script
    if self.SYSTEM == 'GRID': 

        cmd = f"""#!/bin/bash
#$ -V
#$ -cwd
#$ -N {self.SUBMIT_PREFIX}_controller
#$ -l mf=1G
#$ -pe openmpi144 {self.MAX_JOBS} 
#$ -o {self.FOLDER_HOME}/controller.out
#$ -e {self.FOLDER_HOME}/controller.err

set -e  # Exit script on any error

cd {self.FOLDER_HOME}/..

pwd > test.txt
date >> test.txt

python {self.FOLDER_HOME}/start_controller_parallel.py

echo test2 >> test.txt

"""      

    else: 
        logging.error(f"ERROR! SYSTEM: {self.SYSTEM} not defined in start_controller_parallel() in helper.py.")
        sys.exit()    
        
    with open(f"{self.FOLDER_HOME}/start_controller_parallel.sh", "w") as f:
        f.write(cmd)
        
    logging.info(f"Starting parallel controller.")

    ### Start job
    if self.SYSTEM == 'GRID': 
        output = subprocess.check_output(
    (f'qsub {self.FOLDER_HOME}/start_controller_parallel.sh'),
    shell=True, text=True
    )
    else: 
        logging.error(f"ERROR! SYSTEM: {self.SYSTEM} not defined in start_controller_parallel() in helper.py.")
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

def save_cat_res_into_all_scores_df(self, index, PDB_file_path, save_resn=True):
    
    '''Finds the indices and names of the catalytic residue from <PDB_file_path> 
       Saves indices and residues into <all_scores_df> in row <index> as lists.
       To make sure these are saved and loaded as list, ";".join() and .split(";") should be used
       If information is read from an input structure for design do not save cat_resn'''

    with open(PDB_file_path, 'r') as f: 
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
            score_file_path = f"{FOLDER_HOME}/{int(index)}/score_rosetta_design.sc"
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

def get_best_structures(self, save_structures = False, include_catalytic_score = False, seq_per_active_site = 100, DESIGN = None, WT = None):
    if save_structures:
        print("Saving structures...")
    else:
        print("Not saving structures...")

    # Condition to check if the ALL_SCORES_CSV file exists, otherwise it returns the function.
    if not os.path.isfile(f'{self.FOLDER_HOME}/all_scores.csv'): 
        print(f"ERROR: {self.FOLDER_HOME}/all_scores.csv does not exist!")
        return    
    
    all_scores_df = pd.read_csv(self.ALL_SCORES_CSV)

    # Calculate the combined scores using the normalize_scores function
    catalytic_scores, total_scores, interface_scores, efield_scores, identical_scores, combined_scores = normalize_scores(self, unblocked_all_scores_df=all_scores_df, print_norm=False, norm_all=False)
    all_scores_df['combined_score'] = combined_scores
    all_scores_df['norm_total_score'] = total_scores
    all_scores_df['norm_interface_score'] = interface_scores
    all_scores_df['norm_efield_score'] = efield_scores
    
    # Remove rows where 'sequence' is NaN
    all_scores_df = all_scores_df.dropna(subset=['sequence'])  
    
    # Calculate the final score which excludes the catalytic and identical score
    all_scores_df['final_score'] = all_scores_df[['norm_total_score', 'norm_interface_score', 'norm_efield_score']].mean(axis=1) 
    
    all_scores_df['replicate_sequences'] = 0  # Initialize to count duplicates
    all_scores_df['replicate_sequences_final_score'] = 0.0  # To store the average score
    all_scores_df['replicate_sequences_final_score_std'] = 0.0  # To store the standard deviation

    # Loop to find duplicates, calculate average score, and standard deviation
    for i, row in all_scores_df.iterrows():
        duplicates = all_scores_df[all_scores_df['sequence'] == row['sequence']]
        avg_score = duplicates['final_score'].mean()
        std_dev = duplicates['final_score'].std()

        all_scores_df.at[i, 'replicate_sequences'] = len(duplicates)
        all_scores_df.at[i, 'replicate_sequences_final_score'] = avg_score
        all_scores_df.at[i, 'replicate_sequences_final_score_std'] = std_dev

    # Remove replicates and keep only highest final
    all_scores_df.sort_values(by=['final_score'], ascending=[False], inplace=True)
    all_scores_df.drop_duplicates(subset=['sequence'], keep='first', inplace=True)

    # Define Design group
    def get_design_sequence(sequence, design_positions):
        return ''.join(sequence[pos - 1] for pos in design_positions)
    design_positions = [int(pos) for pos in DESIGN.split(',')]
    
    all_scores_df['design_group'] = all_scores_df['sequence'].apply(lambda seq: get_design_sequence(seq, design_positions))

    # Use the standard deviation selection for catalytic score
    if not include_catalytic_score:
        # Use the standard deviation selection for catalytic score
        mean_catalytic_score = all_scores_df['catalytic_score'].mean()
        std_catalytic_score = all_scores_df['catalytic_score'].std()
        all_scores_df = all_scores_df[all_scores_df['catalytic_score'] < mean_catalytic_score + std_catalytic_score]

    # Get the best variants while respecting the seq_per_active_site limit
    top_variants = []
    group_counts = {}

    for _, row in all_scores_df.iterrows():
        group = row['design_group']
        if group not in group_counts:
            group_counts[group] = 0
        if group_counts[group] < seq_per_active_site:
            top_variants.append(row)
            group_counts[group] += 1
        if len(top_variants) >= 100:
            break

    top100 = pd.DataFrame(top_variants)

    selected_indices = np.array(top100['index'].tolist(), dtype=int)
    print(selected_indices)
    print(top100)

    # Print average scores for top 100 and all data points
    score_types = ['final_score', 'combined_score', 'total_score', 'norm_total_score', 'interface_score', 'norm_interface_score', 'efield_score', 'norm_efield_score', 'catalytic_score', 'identical_score', 'mutations']
    print_average_scores(all_scores_df, top100, score_types)

    # Create the destination folder if it doesn't exist
    if include_catalytic_score:
        best_structures_folder = os.path.join(self.FOLDER_HOME, 'best_structures')
    else:
        best_structures_folder = os.path.join(self.FOLDER_HOME, 'best_structures_nocat')
    os.makedirs(best_structures_folder, exist_ok=True)

    # Create the plots folder
    plots_folder = os.path.join(best_structures_folder, 'plots')
    os.makedirs(plots_folder, exist_ok=True)

    # Copy files based on the top100 'index'
    print(best_structures_folder)
    if save_structures:
        print("Saving...")
        for index, row in top100.iterrows():
            geom_mean = "{:.3f}".format(row['final_score'])
            relax_file = f"{self.FOLDER_HOME}/{int(index)}/{WT}_RosettaRelax_{int(index)}.pdb"
            design_file = f"{self.FOLDER_HOME}/{int(index)}/{WT}_RosettaDesign_{int(index)}.pdb"
            if os.path.isfile(relax_file):
                src_file = relax_file
            else:
                src_file = design_file
            dest_file = os.path.join(best_structures_folder, f"{geom_mean}_{WT}_Rosetta_{os.path.basename(src_file)}")
            shutil.copy(src_file, dest_file)
        print("Saved structures to: ", best_structures_folder)
            
    # Plot sorted total score, interface score, and efield score distributions
    def plot_elbow_curve(scores_dict, title, top_indices, filename):
        sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        indices, scores = zip(*sorted_scores)
        colors = ['orange' if idx in top_indices else 'blue' for idx in indices]
        alphas = [1.0 if idx in top_indices else 0.05 for idx in indices]
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(scores)), scores, color=colors, alpha=alphas, s=1)
        plt.title(title)
        plt.xlabel('Variants')
        plt.ylabel('Score')
        plt.savefig(filename)
        plt.show()
        plt.close()

    top100_indices = set(top100['index'])

    plot_elbow_curve(all_scores_df.set_index('index')['total_score'].to_dict(), 'Total Score Elbow Curve', top100_indices, os.path.join(plots_folder, 'total_score_elbow_curve.png'))
    plot_elbow_curve(all_scores_df.set_index('index')['interface_score'].to_dict(), 'Interface Score Elbow Curve', top100_indices, os.path.join(plots_folder, 'interface_score_elbow_curve.png'))
    plot_elbow_curve(all_scores_df.set_index('index')['efield_score'].to_dict(), 'Efield Score Elbow Curve', top100_indices, os.path.join(plots_folder, 'efield_score_elbow_curve.png'))
    plot_elbow_curve(all_scores_df.set_index('index')['catalytic_score'].to_dict(), 'Catalytic Score Elbow Curve', top100_indices, os.path.join(plots_folder, 'catalytic_score_elbow_curve.png'))

    return selected_indices

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
    _, _, _, _, combined_scores_normalized = normalize_scores(all_scores_df, print_norm=True, norm_all=True)
    
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

def print_average_scores(all_scores_df, top100, score_types):
    print("\nSummary of Average Scores:")
    print(f"{'Score Type':<20} {'Average of All':<20} {'Average of Top 100':<20}")
    print("="*60)
    for score_type in score_types:
        avg_all = all_scores_df[score_type].mean()
        avg_top100 = top100[score_type].mean()
        print(f"{score_type.replace('_', ' ').title():<20} {avg_all:<20.4f} {avg_top100:<20.4f}")
    print("\n")

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