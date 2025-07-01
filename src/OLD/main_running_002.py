"""
main_running_002.py

This module contains the main control functions for managing the AIzymes workflow, including job submission,
score updating, and Boltzmann selection. The functions in this file are responsible for the high-level management
of the design process, interacting with the AIzymes_MAIN class to initiate, control, and evaluate design variants.

Classes:
    None

Functions:
    start_controller(self)
    check_running_jobs(self)
    update_potential(self, score_type, index)
    update_scores(self)
    boltzmann_selection(self)
    check_parent_done(self)
    start_parent_design(self)
    start_calculation(self, parent_index)
    create_new_index(self, parent_index)
"""

import os
import time
import subprocess
import pandas as pd
import numpy as np
import logging
import random
import json

from helper_001               import *
from main_design_001          import *
from scoring_efields_001      import *

# -------------------------------------------------------------------------------------------------------------------------
# Keep the controller as clean as possible!!! All complex operations are performed on the next level! ---------------------
# -------------------------------------------------------------------------------------------------------------------------

def start_controller(self):
    '''
    The start_controller function is called in the AIzyme_0X script's controller function.
    The controller function decides what action to take, 
    assures that the maximum number of design jobs are run in parallel, 
    collects information from the designs and stores them in a shared database, 
    selects the variants to submit for design,
    and decides the type of structure prediction to perform with the selected variant.
    
    Each ofthese tasks is performed by the functions introduced before, and thereforer the start_controller function controls the flow of actions.
    '''

    # Run this part of the function until the maximum number of designs has been reached.
    while len(self.all_scores_df['index']) < int(self.MAX_DESIGNS): 
            
        # Check how many jobs are currently running
        # Wait if the number is equal or bigger than the maximum number of parallel jobs allowed.
        num_running_jobs = check_running_jobs(self)
        
        if num_running_jobs >= self.MAX_JOBS: 
            time.sleep(20)
        
        #If the number of maximum parallel jobs is not reached, the AIzyme design protocol is executed.
        else:
            update_scores(self)
                        
            # Checks if all the parent designs needed, defined by the PARENT_JOB variable, are generated.
            parent_done = check_parent_done(self)
            
            # If the parent designs are not generated, it continues design until the number PARENT_JOB is reached. 
            if not parent_done:
                start_parent_design(self)

            else:
                # Performs Boltzmann selection to select the parent index for the next design
                selected_index = boltzmann_selection(self)
                
                # Checks whether a valid index is returned by the Boltzmann selection (error handling)
                # Starts the design of the next variant using the selected index as parent
                if selected_index is not None:
                    start_calculation(self, selected_index)
    
        # Sleep a bit for safety
        time.sleep(0.1)
        
    # When the maximum number of designs has been generated, the corresponding scores are calculated and added to the all_scores.csv file.
    update_scores(self)
    
    print(f"Stopped because {len(self.all_scores_df['index'])}/{self.MAX_DESIGNS} designs have been made.")
        
def check_running_jobs(self):
    """
    The check_running_job function returns the number of parallel jobs that are running by counting how many lines in the qstat output are present which correspond to the
    job prefix. This is true when working on the GRID, for the other system the same concept is being used but the terminology differs.

    Returns:
        int: Number of running jobs for the specific system.
    """
   
    if self.RUN_PARALLEL:

        for p, out_file, err_file in self.processes:
            if p.poll() is not None: # Close finished files
                out_file.close()
                err_file.close()
        self.processes = [(p, out_file, err_file) for p, out_file, err_file in self.processes if p.poll() is None]
        logging.debug(f"{len(self.processes)} parallel jobs.")  
                
        return len(self.processes)
   
    elif self.SYSTEM == 'GRID': 
        command = ["ssh", "mdewaal@bs-submit04.ethz.ch", "qstat", "-u", "mdewaal"]
        result = subprocess.run(command, capture_output=True, text=True)
        jobs = result.stdout.split("\n")
        jobs = [job for job in jobs if self.SUBMIT_PREFIX in job]
        return len(jobs)
        
    elif self.SYSTEM == 'BLUEPEBBLE':
        jobs = subprocess.check_output(["squeue","--me"]).decode("utf-8").split("\n")
        jobs = [job for job in jobs if self.SUBMIT_PREFIX in job]
        jobs = len(jobs)

    elif self.SYSTEM == 'ABBIE_LOCAL':
        jobs = 0

    else:
        logging.error(f"SYSTEM: {self.SYSTEM} not defined in check_running_jobs() which is part of main_running.py.")
        sys.exit()

    if jobs == None : jobs = 0
    return jobs
    
def update_potential(self, score_type, index): 
    """
    Updates the potential file for a given score type at the specified variant index.

    Creates or appends to a `<score_type>_potential.dat` file in `FOLDER_HOME/<index>`, calculating and
    updating potentials for the parent variant if necessary.

    Parameters:
        score_type (str): Type of score to update (e.g., total, interface, catalytic, efield).
        index (int): Variant index to update potential data.
    """
    score = self.all_scores_df.at[index, f'{score_type}_score']
    score_taken_from = self.all_scores_df.at[index, 'score_taken_from']    
    parent_index = self.all_scores_df.at[index, "parent_index"] 
    parent_filename = f"{self.FOLDER_HOME}/{parent_index}/{score_type}_potential.dat"  

    # Update current potential
    with open(f"{self.FOLDER_HOME}/{index}/{score_type}_potential.dat", "w") as f: 
        f.write(str(score))
    self.all_scores_df.at[index, f'{score_type}_potential'] = score

    #Update parent potential
    if score_taken_from != "RosettaRelax": return                     # Only update the parent potential for RosettaRelax
    if parent_index == "Parent":           return                     # Do not update the parent potential of a variant from parent
    if score_type == "Identical":          return                     # Do not update the potential of the identical score
    with open(parent_filename, "a") as f:  f.write(f"\n{str(score)}") # Appends to parent_filename
    with open(parent_filename, "r") as f:  potentials = f.readlines() # Reads in potential values 
    self.all_scores_df.at[int(parent_index), f'{score_type}_potential'] = np.average([float(i) for i in potentials])

 
def update_scores(self):
    """
    Updates various scores, including total, interface, catalytic, and efield scores for each design variant.

    This function iterates over design variants, updating scores based on files generated by different processes.
    It also updates sequence information, tracks mutations, and saves the updated DataFrame.

    Parameters:
        self: An instance of the AIzymes_MAIN class with setup attributes and properties.
    """

    logging.debug("Updating scores")
        
    for _, row in self.all_scores_df.iterrows():

        if pd.isna(row['index']): continue # Prevents weird things from happening
        index = int(row['index'])
        parent_index = row['parent_index']         
        
        #unblock index for calculations that should only be executed once!
        for unblock in ["RosettaRelax","ESMfold"]:
            if self.all_scores_df.at[int(index), f"blocked_{unblock}"] == True:
                if os.path.isfile(f"{self.FOLDER_HOME}/{index}/{self.WT}_{unblock}_{index}.pdb"):
                    self.all_scores_df.at[index, f"blocked_{unblock}"] = False
                    logging.debug(f"Unblocked {unblock} index {int(index)}.")      
             
        seq_path = f"{self.FOLDER_HOME}/{index}/{self.WT}_{index}.seq"
        
        # Update sequence, central residues and mutations if row does not yet contain a sequence
        if pd.isna(self.all_scores_df.at[index, 'sequence']):
            if os.path.exists(seq_path):
                with open(f"{self.FOLDER_PARENT}/{self.WT}.seq", "r") as f:
                    reference_sequence = f.read()
                with open(seq_path, "r") as f:
                    current_sequence = f.read()
                mutations = sum(1 for a, b in zip(current_sequence, reference_sequence) if a != b)
                self.all_scores_df['sequence'] = self.all_scores_df['sequence'].astype('object')
                self.all_scores_df.at[index, 'sequence']  = current_sequence
                self.all_scores_df.at[index, 'central_res']  = ''.join(current_sequence[int(pos)] for pos in self.DESIGN.split(','))
                self.all_scores_df.at[index, 'mutations'] = int(mutations)

        # Calculate identical score
        identical_score = 0.0
        if "identical" in self.SELECTED_SCORES:
            
            central_res = self.all_scores_df.at[index, 'central_res']
            parent_index = self.all_scores_df.at[index, 'parent_index']
            
            if pd.notna(central_res):
                if parent_index == 'Parent': # If it's a parent, set identical score to 1
                    identical_score = 1.0 
                else:
                    # Calculate the number of occurrences of the identical central residues
                    identical_count = (self.all_scores_df['central_res'] == central_res).sum()
                    identical_score = 1 / identical_count if identical_count > 0 else 0.0
        
        # Update identical score and potential
        self.all_scores_df.at[index, 'identical_score'] = identical_score
        update_potential(self, score_type='identical', index=index)
        
        # Check what structure to score on
        if os.path.exists(f"{self.FOLDER_HOME}/{int(index)}/score_RosettaRelax.sc"): # Score based on RosettaRelax            
            if row['score_taken_from'] == 'RosettaRelax': continue # Do NOT update score to prevent repeated scoring!
            score_type = 'RosettaRelax'
                        
        elif os.path.exists(f"{self.FOLDER_HOME}/{int(index)}/score_RosettaDesign.sc"): # Score based on RosettaDesign
            if row['score_taken_from'] == 'RosettaDesign': continue # Do NOT update score to prevent repeated scoring! 
            score_type = 'RosettaDesign'
            
        elif os.path.exists(seq_path): # Update just cat_resn (needed for ProteinMPNN and LigandMPNN)
            
            with open(seq_path, "r") as f:
                seq = f.read()
            cat_resns = []
            for cat_resi in str(self.all_scores_df.at[index, 'cat_resi']).split(";"): 
                cat_resns += [one_to_three_letter_aa(seq[int(float(cat_resi))-1])]
            self.all_scores_df['cat_resn'] = self.all_scores_df['cat_resn'].astype(str)
            self.all_scores_df.at[index, 'cat_resn'] = ";".join(cat_resns)
            
            continue # Do NOT update anything else
        
        else:
            
            continue # Do NOT update score, job is not done.
        
        # Set paths
        score_file_path = f"{self.FOLDER_HOME}/{int(index)}/score_{score_type}.sc"
        pdb_path = f"{self.FOLDER_HOME}/{int(index)}/{self.WT}_{score_type}_{int(index)}.pdb"
            
        # Do not update score if files do not exist!
        if not os.path.isfile(pdb_path): continue
        if not os.path.isfile(seq_path): continue

        # Check if ElectricFields are done   
        if not os.path.exists(f"{self.FOLDER_HOME}/{int(index)}/ElectricFields/{self.WT}_{score_type}_{index}_fields.pkl"):
            continue 
        
        # Load scores
        with open(score_file_path, "r") as f:
            scores = f.readlines()
        if len(scores) < 3: continue # If the timing is bad, the score file is not fully written. Check if len(scores) > 2!
        headers = scores[1].split()
        scores  = scores[2].split()
        
        # Update score_taken_from
        self.all_scores_df['score_taken_from'] = self.all_scores_df['score_taken_from'].astype(str)
        if "RosettaRelax" in score_file_path:
            self.all_scores_df.at[index, 'score_taken_from'] = 'RosettaRelax'
        if "RosettaDesign" in score_file_path:
            self.all_scores_df.at[index, 'score_taken_from'] = 'RosettaDesign'
                
        # Update catalytic residues
        save_cat_res_into_all_scores_df(self, index, pdb_path, save_resn=True) 
        #cat_res = self.all_scores_df.at[index, 'cat_resi']
        
        # Calculate scores
        catalytic_score = 0.0
        interface_score = 0.0
        efield_score = 0.0
        total_score = 0.0
        
        for idx_headers, header in enumerate(headers):
            if "total" in self.SELECTED_SCORES:
                if header == 'total_score':                total_score      = float(scores[idx_headers])
            # Subtract constraints from interface score
            if "interface" in self.SELECTED_SCORES:
                if header == 'interface_delta_X':          interface_score += float(scores[idx_headers])
                if header in ['if_X_angle_constraint', 
                          'if_X_atom_pair_constraint', 
                          'if_X_dihedral_constraint']: interface_score -= float(scores[idx_headers])   
            # Calculate catalytic score by adding constraints
            if "catalytic" in self.SELECTED_SCORES:
                if header in ['atom_pair_constraint']:     catalytic_score += float(scores[idx_headers])       
                if header in ['angle_constraint']:         catalytic_score += float(scores[idx_headers])       
                if header in ['dihedral_constraint']:      catalytic_score += float(scores[idx_headers]) 
        
        # Calculate efield score
        if "efield" in self.SELECTED_SCORES:
            efield_score, index_efields_dict = get_efields_score(self, index, score_type)  
            update_efieldsdf(self, index, index_efields_dict)   

        # Update scores
        self.all_scores_df.at[index, 'total_score']     = total_score
        self.all_scores_df.at[index, 'interface_score'] = interface_score                
        self.all_scores_df.at[index, 'catalytic_score'] = catalytic_score
        self.all_scores_df.at[index, 'efield_score']    = efield_score

        # This is just for book keeping. AIzymes will always use the most up_to_date scores saved above
        if "RosettaRelax" in score_file_path:
            self.all_scores_df.at[index, 'relax_total_score']     = total_score
            self.all_scores_df.at[index, 'relax_interface_score'] = interface_score                
            self.all_scores_df.at[index, 'relax_catalytic_score'] = catalytic_score
            self.all_scores_df.at[index, 'relax_efield_score'] = efield_score
            
        if "RosettaDesign" in score_file_path:
            self.all_scores_df.at[index, 'design_total_score']     = total_score
            self.all_scores_df.at[index, 'design_interface_score'] = interface_score                
            self.all_scores_df.at[index, 'design_catalytic_score'] = catalytic_score
            self.all_scores_df.at[index, 'design_efield_score'] = efield_score

        for score_type in self.SELECTED_SCORES: 
            if score_type != "identical":
                update_potential(self, score_type=score_type, index= index)   

        logging.info(f"Updated scores and potentials of index {index}.")
        if self.all_scores_df.at[index, 'score_taken_from'] == 'Relax' and self.all_scores_df.at[index, 'parent_index'] != "Parent":
            logging.info(f"Adjusted potentials of {self.all_scores_df.at[index, 'parent_index']}, parent of {int(index)}).")

    save_all_scores_df(self)
        

def boltzmann_selection(self):
    """
    Selects a design variant based on a Boltzmann-weighted probability distribution.

    Filters variants based on certain conditions (e.g., scores, block status), then computes probabilities
    using Boltzmann factors with a temperature factor (`KBT_BOLTZMANN`) to select a variant for further design steps.

    Returns:
        int: Index of the selected design variant.
    """
        
    parent_indices = set(self.all_scores_df['parent_index'].astype(str).values)
    unblocked_all_scores_df = self.all_scores_df
    
    # Remove blocked indices
    for unblock in ["RosettaRelax","ESMfold"]:                    
        unblocked_all_scores_df = unblocked_all_scores_df[unblocked_all_scores_df[f"blocked_{unblock}"] == False]
             
    # Complet ESMfold and RosettaRelax
    filtered_indices = unblocked_all_scores_df[unblocked_all_scores_df['score_taken_from'] != 'RosettaRelax'] # Remove Relaxed Indeces
    
    # Before running the Boltzmann selection, check if files should be run immediatly
    #  |-- if a design was made (seq exist)
    #    |-- select index is there is no ESMfold
    #      |-- select index is there is no RosettaRelax
    
    for index, row in filtered_indices.iterrows():
        
        # Check if sequence file exists
        if not os.path.isfile(f'{self.FOLDER_HOME}/{index}/{self.WT}_{index}.seq'):
            continue
            
        # If there are designed structures that were not run through ESMFold, run them
        if not os.path.isfile(f'{self.FOLDER_HOME}/{index}/{self.WT}_ESMfold_{index}.pdb'):
            selected_index = index
            return selected_index
            
        # If there are structures that ran through ESMFold but have not been Relax, run them
        elif not os.path.isfile(f'{self.FOLDER_HOME}/{index}/{self.WT}_RosettaRelax_{index}.pdb'):
            selected_index = index
            return selected_index     

    # Drop catalytic scroes > mean + 1 std
    if "catalytic" in self.SELECTED_SCORES:
        mean_catalytic_score = unblocked_all_scores_df['catalytic_score'].mean()
        std_catalytic_score = unblocked_all_scores_df['catalytic_score'].std()
        mean_std_catalytic_score = mean_catalytic_score + std_catalytic_score
        if len(unblocked_all_scores_df) > 10:
            unblocked_all_scores_df = unblocked_all_scores_df[unblocked_all_scores_df['catalytic_score'] < mean_std_catalytic_score]
        
    # Remove indices without score (design running)
    unblocked_all_scores_df = unblocked_all_scores_df.dropna(subset=['total_score'])     
  
    # If there are structures that ran through RosettaRelax but have never been used for design,
    # run design (exclude ProteinMPNN as it is always relaxed)
    relaxed_indices = unblocked_all_scores_df[unblocked_all_scores_df['score_taken_from'] == 'RosettaRelax']
    relaxed_indices = relaxed_indices[relaxed_indices['design_method'] != 'ProteinMPNN']
    relaxed_indices = relaxed_indices[relaxed_indices['design_method'] != 'LigandMPNN']
    relaxed_indices = [str(i) for i in relaxed_indices.index]
    filtered_indices = [index for index in relaxed_indices if index not in parent_indices]

    #### NEEDS TO BE ADJUSTED!!!!
    if len(filtered_indices) >= 1:
        selected_index = filtered_indices[0]
        logging.info(f"{selected_index} selected because its relaxed but nothing was designed from it.")
        return int(selected_index)
    
    # Do Boltzmann Selection if some scores exist
    scores = normalize_scores(self, 
                            unblocked_all_scores_df, 
                            norm_all=False, 
                            extension="potential", 
                            print_norm=False) 
        
    combined_potentials = scores["combined_potential"]
    
    if len(combined_potentials) > 0:
        generation=self.all_scores_df['generation'].max()
                
        if isinstance(self.KBT_BOLTZMANN, (float, int)):
            kbt_boltzmann = self.KBT_BOLTZMANN

        elif len(self.KBT_BOLTZMANN) == 2:
            kbt_boltzmann = self.KBT_BOLTZMANN[0] * np.exp(-self.KBT_BOLTZMANN[1]*generation)

        elif len(self.KBT_BOLTZMANN) == 3:
            kbt_boltzmann = (self.KBT_BOLTZMANN[0] - self.KBT_BOLTZMANN[2]) * np.exp(-self.KBT_BOLTZMANN[1]*generation)+self.KBT_BOLTZMANN[2]
        
        # Some issue with numpy exp when calculating boltzman factors.
        combined_potentials_list = [float(x) for x in combined_potentials]
        combined_potentials = np.array(combined_potentials_list)
        
        boltzmann_factors = np.exp(combined_potentials / kbt_boltzmann)
        probabilities = boltzmann_factors / sum(boltzmann_factors)
        
        if len(unblocked_all_scores_df) > 0:
            selected_index = int(np.random.choice(unblocked_all_scores_df["index"].to_numpy(), p=probabilities))
        else:
            logging.debug(f'Boltzmann selection tries to select a variant for design, but all are blocked. Waiting 20 seconds')
            time.sleep(20)
            return None
        
    else:
        selected_index = 0

    return selected_index


def check_parent_done(self):
    """
    The check_parent_done function is used to determine whether all the parent designs are being generated, namely the design that constitute the first selection pool and that
    belong to generation 0.

    Returns:
        bool: True if parent designs are complete, otherwise False.
    """
    number_of_indices = len(self.all_scores_df)
    
    #Number of ancestor structures used for the experiment
    parent_structures = [i for i in os.listdir(self.FOLDER_PARENT) if i[-4:] == ".pdb"]
    
    if number_of_indices < self.N_PARENT_JOBS * len(parent_structures):
        parent_done = False
        logging.debug(f'Parent design not yet done. {number_of_indices+1}/{self.N_PARENT_JOBS * len(parent_structures)} jobs submitted.')
    else:
        parent_done = True
        
    return parent_done


def start_parent_design(self):
    """
    The start_parent_design function is used to generate the parent designs, namely the variants that belong to generation 0 and that constitute the initial selection pool. The
    number of designs needed for the generation 0 pool is defined by the PARENT_JOB variable set at the beginning of the experiment (standard = 50). All these designs are
    characterized by the parent_index "Parent".
    The function also sets up the required files and configuration for designing a parent structure, then calls the design method
    specified in `PARENT_DES_MED`.

    Parameters:
        self: An instance of the AIzymes_MAIN class with setup attributes and properties.
    """
    number_of_indices = len(self.all_scores_df)
    
    #Number of ancestor structures used for the experiment
    parent_structures = [i for i in os.listdir(self.FOLDER_PARENT) if i[-4:] == ".pdb"]
    
    #Determines the index of the structure in the parent_structures list of structures that is being used
    selected_index = int(number_of_indices / self.N_PARENT_JOBS)
    parent_structure= parent_structures[selected_index][:-4]
        
    new_index = create_new_index(self, parent_index="Parent")
    
    #Adds the parent structure (luca) and design method information to the all_scores dataframe at the new index
    self.all_scores_df['design_method'] = self.all_scores_df['design_method'].astype('object') 
    self.all_scores_df.at[new_index, 'design_method'] = self.PARENT_DES_MED
    self.all_scores_df['luca'] = self.all_scores_df['luca'].astype('object') 
    self.all_scores_df.at[new_index, 'luca'] = parent_structure

    # Add cat res to new entry
    save_cat_res_into_all_scores_df(self, new_index, 
                                   f'{self.FOLDER_PARENT}/{parent_structures[selected_index]}',
                                   save_resn=True)
    
    # Difficult to set kbt_boltzmann of first design. Here we just assign it the number of the second design?????????????
    if new_index == 1: 
        self.all_scores_df.at[new_index-1, 'kbt_boltzmann'] = self.all_scores_df.at[new_index, 'kbt_boltzmann']

    #Runs the design of the parent structure
    if self.PARENT_DES_MED in ["ProteinMPNN","LigandMPNN"]:
        if self.MDMin:
            run_design(self, new_index, [self.PARENT_DES_MED, "ESMfold", "MDMin", "RosettaRelax", "ElectricFields"])
        else:
            run_design(self, new_index, [self.PARENT_DES_MED, "ESMfold", "RosettaRelax", "ElectricFields"])
        self.all_scores_df.at[new_index, "blocked_ESMfold"] = True 
        self.all_scores_df.at[new_index, "blocked_RosettaRelax"] = True 
    elif self.PARENT_DES_MED in ["RosettaDesign"]:  
        run_design(self, new_index, [self.PARENT_DES_MED, "ElectricFields"])
    else:
        logging.error(f"ERROR! PARENT_DES_MED: {self.PARENT_DES_MED} not defined.")
        sys.exit()
      
    save_all_scores_df(self)
    
# Decides what to do with selected index

def start_calculation(self, parent_index: int):
    """
    The start_calculation function checks whether the index that was selected by the Boltzmann selection to be parent_index has gone through ESMfold and RosettaRelax, and if
    necessary execute them. The function then creates the new child index and decides based on the defined probabilities whether to generate the new child variant with
    RosettaDesign, ProteinMPNN, or LigandMPNN and calls the design function.

    Parameters:
        parent_index (int): Index of the variant to evaluate for further calculations.
    """
    logging.debug(f"Starting new calculation for index {parent_index}.")
     
    # if blocked
    #  └──> Error
    # elif no esmfold
    #  └──> run esmfold
    # elif no relax
    #  └──> run relax
    # else
    #  └──> run design

    # Check if index is still blocked, if yes --> STOP. This shouldn't happen!
    if any(self.all_scores_df.at[parent_index, col] == True for col in [f"blocked_RosettaRelax", f"blocked_ESMfold"]):
        logging.error(f"Index {parent_index} is being worked on. Skipping index.")
        logging.error(f"Note: This should not happen! Check blocking and Boltzman selection.")        
        return
    
    # Check if ESMfold is done
    elif not f"{self.WT}_ESMfold_{parent_index}.pdb" in os.listdir(os.path.join(self.FOLDER_HOME, str(parent_index))):
        logging.info(f"Index {parent_index} has no predicted structure, starting ESMfold.")
        self.all_scores_df.at[parent_index, "blocked_ESMfold"] = True   
        self.all_scores_df.at[parent_index, "blocked_RosettaRelax"] = True 
        if self.MDMin:
            run_design(self, parent_index, ["ESMfold", "MDMin", "RosettaRelax", "ElectricFields"])  
        else:
            run_design(self, parent_index, ["ESMfold", "RosettaRelax", "ElectricFields"])  
    
    # Check if RosettaRelax is done --> should never happen, probabbly not needed anymore but keep for consistency
    elif not f"{self.WT}_RosettaRelax_{parent_index}.pdb" in os.listdir(os.path.join(self.FOLDER_HOME, str(parent_index))):
        logging.info(f"Index {parent_index} has no relaxed structure, starting RosettaRelax.")
        self.all_scores_df.at[parent_index, "blocked_RosettaRelax"] = True 
        if self.MDMin:
            run_design(self, parent_index, ["MDMin", "RosettaRelax", "ElectricFields"])    
        else:
            run_design(self, parent_index, ["RosettaRelax", "ElectricFields"])   

    # If all OK, start Design
    else:

        # RosettaRelax is done, create a new index
        new_index = create_new_index(self, parent_index)

        # Add cat res to new entry
        save_cat_res_into_all_scores_df(self, new_index, 
                                       f"{self.FOLDER_HOME}/{parent_index}/{self.WT}_RosettaRelax_{parent_index}.pdb",
                                       save_resn=True)
        
        #####
        # Here, we can add an AI to decide on the next steps
        #####

        # Run Design with new_index --> need to add code to CHECK IF self.ProteinMPNN_PROB + self.LigandMPNN_PROB is below 1!!!!
        if random.random() < self.ProteinMPNN_PROB:  
            self.all_scores_df.at[new_index, 'design_method'] = "ProteinMPNN"
            if self.MDMin:
                run_design(self, new_index, ["ProteinMPNN", "ESMfold", "MDMin", "RosettaRelax", "ElectricFields"]) 
            else:
                run_design(self, new_index, ["ProteinMPNN", "ESMfold", "RosettaRelax", "ElectricFields"])
            #Possibly could put this into indivual prepare_design steps?
            self.all_scores_df.at[new_index, "blocked_ESMfold"] = True 
            self.all_scores_df.at[new_index, "blocked_RosettaRelax"] = True 
        elif random.random()+self.ProteinMPNN_PROB < self.LigandMPNN_PROB+self.ProteinMPNN_PROB:
            self.all_scores_df.at[new_index, 'design_method'] = "LigandMPNN"
            if self.MDMin:
                run_design(self, new_index, ["LigandMPNN", "ESMfold", "MDMin", "RosettaRelax", "ElectricFields"]) 
            else:
                run_design(self, new_index, ["LigandMPNN", "ESMfold", "RosettaRelax", "ElectricFields"])
            self.all_scores_df.at[new_index, "blocked_ESMfold"] = True 
            self.all_scores_df.at[new_index, "blocked_RosettaRelax"] = True 
        else:                    
            self.all_scores_df.at[new_index, 'design_method'] = "RosettaDesign"
            run_design(self, new_index, ["RosettaDesign", "ElectricFields"])
        
    save_all_scores_df(self)
        

def create_new_index(self, parent_index: str):
    """
    The create_new_index function is responsible for generating a new index for both the initial generation (called by start_parent_design) and subsequent generations (called
    by start_calculation). This function assigns a new index, sets various attributes (such as generation, parent index, cst_weight, and other parameters), and updates the
    all_scores DataFrame with the new index and its associated values. Additionally, it creates a dedicated folder for the new index within the experiment directory.
    The parameters kbt_weight and cst_weight can take one of three formats: a single value, a two-value list, or a three-value list, depending on the context. When multiple
    values are provided, the weights are gradually reduced with each successive generation. This progressive reduction allows for more mutations and a more flexible selection
    process in later generations, facilitating a smoother exploration of the sequence landscape by the AIzyme program.

    Parameters:
        parent_index (str): The index of the parent variant or "Parent" for initial designs.

    Returns:
        int: The newly created index for the variant.
    """
    #Creates the new index value
    new_index = len(self.all_scores_df)
    
    #Defines the parent_index, generation and luca values
    
    #When the create_new_index function is called from the start_parent_design function, parent_index is "Parent" and therefore the generation is set to 0.
    if parent_index == 'Parent':
        generation = 0
        luca = "x"
        
    #If the new index doesn't belong to generation 0, the generation and the luca structure are assessed from its parent, which is defined by parent_index
    else:
        generation = self.all_scores_df['generation'][int(parent_index)]+1
        luca = self.all_scores_df['luca'][int(parent_index)]
        
    #Defines the values of the kbt_boltzmann and cst_weight based on the generation of the new index
    
    #Determines the kbt_weight of the new index
    if isinstance(self.KBT_BOLTZMANN, (float, int)):
        kbt_boltzmann = self.KBT_BOLTZMANN
        
    elif len(self.KBT_BOLTZMANN) == 2:
        kbt_boltzmann = self.KBT_BOLTZMANN[0] * np.exp(-self.KBT_BOLTZMANN[1]*generation)
        
    elif len(self.KBT_BOLTZMANN) == 3:
        kbt_boltzmann = (self.KBT_BOLTZMANN[0] - self.KBT_BOLTZMANN[2]) * np.exp(-self.KBT_BOLTZMANN[1]*generation)+self.KBT_BOLTZMANN[2]
        
    #Determines the cst_weight of the new index
    if isinstance(self.CST_WEIGHT, (float, int)):
        cst_weight = self.CST_WEIGHT
        
    elif len(self.CST_WEIGHT) == 2:
        cst_weight = self.CST_WEIGHT[0]*np.exp(-self.CST_WEIGHT[1]*generation)

    elif len(self.CST_WEIGHT) == 3:
        cst_weight = (self.CST_WEIGHT[0] - self.CST_WEIGHT[2])*np.exp(-self.CST_WEIGHT[1]*generation) + self.CST_WEIGHT[2]

    
    #Creates a new dataframe with all the necessary columns for the new index, concatenes it with the existing all_scores dataframe and saves it
    new_index_df = pd.DataFrame({'index': int(new_index), 
                                'parent_index': parent_index,
                                'kbt_boltzmann': kbt_boltzmann,
                                'cst_weight': cst_weight,
                                'generation': generation,
                                'luca': luca,
                                'blocked_ESMfold': False,
                                'blocked_RosettaRelax': False,
                                }, index = [0] , dtype=object)  
        
    self.all_scores_df = pd.concat([self.all_scores_df, new_index_df], ignore_index=True)

    save_all_scores_df(self)

    # Create the folders for the new index
    os.makedirs(f"{self.FOLDER_HOME}/{new_index}/scripts", exist_ok=True)
    os.makedirs(f"{self.FOLDER_HOME}/{new_index}/ElectricFields", exist_ok=True)
           
    logging.debug(f"Child index {new_index} created for {parent_index}.")
    
    return new_index