"""
main_running_001.py

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

from helper_002               import *
from main_design_001          import *
from scoring_efields_001      import *

# -------------------------------------------------------------------------------------------------------------------------
# Keep the controller as clean as possible!!! All complex operations are performed on the next level! ---------------------
# -------------------------------------------------------------------------------------------------------------------------


def start_controller(self):
    """
    Runs the main loop to manage the design process until the maximum number of designs is reached.

    Continues submitting jobs, monitoring running processes, and updating scores based on Boltzmann selection
    until `MAX_DESIGNS` is achieved. The function pauses or starts new designs based on system resources.

    Parameters:
        self: An instance of the AIzymes_MAIN class with setup attributes and properties.
    """
    
    while len(self.all_scores_df['index']) < int(self.MAX_DESIGNS): #Run until MAX_DESIGNS are made
            
        # Check how many jobs are currently running
        num_running_jobs = check_running_jobs(self)
        
        if num_running_jobs >= self.MAX_JOBS: 

            # Pause and continue after some time
            time.sleep(20)
            
        else:
                   
            # Update scores
            update_scores(self)
                        
            # Check if parent designs are done, if not, start design
            parent_done = check_parent_done(self)
            
            if not parent_done:
                
                start_parent_design(self)

            else:
                
                # Boltzmann Selection
                selected_index = boltzmann_selection(self)
                
                # Decide Fate of selected index
                if selected_index is not None:
                    start_calculation(self, selected_index)
    
        # Sleep a bit for safety
        time.sleep(0.1)

    update_scores(self)
    
    print(f"Stopped because {len(self.all_scores_df['index'])}/{self.MAX_DESIGNS} designs have been made.")


def check_running_jobs(self):
    """
    Checks the current number of running jobs based on the system type.

    Depending on the value of `SYSTEM`, this function counts the active jobs in GRID, BLUEPEBBLE, BACKGROUND_JOB,
    or ABBIE_LOCAL systems.

    Returns:
        int: Number of running jobs for the specific system.
    """
   
    if self.SYSTEM == 'GRID':
        jobs = subprocess.check_output(["qstat", "-u", self.USERNAME]).decode("utf-8").split("\n")
        jobs = [job for job in jobs if self.SUBMIT_PREFIX in job]
        return len(jobs)
        
    if self.SYSTEM == 'BLUEPEBBLE':
        jobs = subprocess.check_output(["squeue","--me"]).decode("utf-8").split("\n")
        jobs = [job for job in jobs if self.SUBMIT_PREFIX in job]
        return len(jobs)
        
    if self.SYSTEM == 'BACKGROUND_JOB':
        with open(f'{self.FOLDER_HOME}/n_running_jobs.dat', 'r') as f: jobs = int(f.read())
        return jobs
    
    if self.SYSTEM == 'ABBIE_LOCAL':
        return 0


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

    #Update parrent potential
    if score_taken_from != "RosettaRelax": return                     # Only update the parent potential for RosettaRelax
    if parent_index == "Parent":           return                     # Do not update the parent potential of a variant from parent
    with open(parent_filename, "a") as f:  f.write(f"\n{str(score)}") # Appends to parent_filename
    with open(parent_filename, "r") as f:  potentials = f.readlines() # Reads in potential values 
    self.all_scores_df.at[parent_index, f'{score_type}_potential'] = np.average([float(i) for i in potentials])
        

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
        cat_res = self.all_scores_df.at[index, 'cat_resi']
        
        # Calculate catalytic and interface score
        catalytic_score = 0.0
        interface_score = 0.0
        for idx_headers, header in enumerate(headers):
            if header == 'total_score':                total_score      = float(scores[idx_headers])
            # Subtract constraints from interface score
            if header == 'interface_delta_X':          interface_score += float(scores[idx_headers])
            if header in ['if_X_angle_constraint', 
                          'if_X_atom_pair_constraint', 
                          'if_X_dihedral_constraint']: interface_score -= float(scores[idx_headers])   
            # Calculate catalytic score by adding constraints
            if header in ['atom_pair_constraint']:     catalytic_score += float(scores[idx_headers])       
            if header in ['angle_constraint']:         catalytic_score += float(scores[idx_headers])       
            if header in ['dihedral_constraint']:      catalytic_score += float(scores[idx_headers]) 

        # Calculate efield score
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

        for score_type in ['total', 'interface', 'catalytic', 'efield']:     
            update_potential(self, score_type=score_type, index= index)   

        logging.info(f"Updated scores and potentials of index {index}.")
        if self.all_scores_df.at[index, 'score_taken_from'] == 'Relax' and self.all_scores_df.at[index, 'parent_index'] != "Parent":
            logging.info(f"Adjusted potentials of {self.all_scores_df.at[index, 'parent_index']}, parent of {int(index)}).")
                        
        # Update sequence and mutations
        with open(f"{self.FOLDER_PARENT}/{self.WT}.seq", "r") as f:
            reference_sequence = f.read()
        with open(seq_path, "r") as f:
            current_sequence = f.read()
        mutations = sum(1 for a, b in zip(current_sequence, reference_sequence) if a != b)
        self.all_scores_df['sequence'] = self.all_scores_df['sequence'].astype('object')
        self.all_scores_df.at[index, 'sequence']  = current_sequence
        self.all_scores_df.at[index, 'mutations'] = int(mutations)

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
    _, _, _, _, combined_potentials = normalize_scores(self, 
                                                       unblocked_all_scores_df, 
                                                       norm_all=False, 
                                                       extension="potential", 
                                                       print_norm=False) 
        
    if len(combined_potentials) > 0:
        
        if isinstance(self.KBT_BOLTZMANN, (float, int)):
            kbt_boltzmann = self.KBT_BOLTZMANN
        elif len(self.KBT_BOLTZMANN) > 2:
            logging.error(f"KBT_BOLTZMANN must either be a single value or list of two values.")
            logging.error(f"KBT_BOLTZMANN is {self.KBT_BOLTZMANN}")
        elif len(self.KBT_BOLTZMANN) == 2:
            # Ramp down kbT_boltzmann over time (i.e., with increaseing indices)
            # datapoints = legth of all_scores_df - number of parents generated
            num_pdb_files = len([file for file in os.listdir(self.FOLDER_PARENT) if file.endswith('.pdb')])
            datapoints = max(self.all_scores_df['index'].max() + 1 - num_pdb_files*self.N_PARENT_JOBS, 0)
            kbt_boltzmann = float(max(self.KBT_BOLTZMANN[0] * np.exp(-self.KBT_BOLTZMANN[1]*datapoints), 0.05))
        
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
    Determines if parent designs are complete based on the number of generated designs and parent jobs.

    Returns:
        bool: True if parent designs are complete, otherwise False.
    """
       
    number_of_indices = len(self.all_scores_df)
    parents = [i for i in os.listdir(self.FOLDER_PARENT) if i[-4:] == ".pdb"]
    if number_of_indices < self.N_PARENT_JOBS * len(parents):
        parent_done = False
        logging.debug(f'Parent design not yet done. {number_of_indices+1}/{self.N_PARENT_JOBS * len(parents)} jobs submitted.')
    else:
        parent_done = True
        
    return parent_done
    

def start_parent_design(self):
    """
    Initiates a new design process for a parent structure by creating a new variant entry.

    Sets up the required files and configuration for designing a parent structure, then calls the design method
    specified in `PARENT_DES_MED`.

    Parameters:
        self: An instance of the AIzymes_MAIN class with setup attributes and properties.
    """
    
    number_of_indices = len(self.all_scores_df)
    PARENTS = [i for i in os.listdir(self.FOLDER_PARENT) if i[-4:] == ".pdb"]
    
    selected_index = int(number_of_indices / self.N_PARENT_JOBS)
    parent_index = PARENTS[selected_index][:-4]
        
    new_index = create_new_index(self, parent_index="Parent")
    self.all_scores_df['design_method'] = self.all_scores_df['design_method'].astype('object') 
    self.all_scores_df.at[new_index, 'design_method'] = self.PARENT_DES_MED
    self.all_scores_df['luca'] = self.all_scores_df['luca'].astype('object') 
    self.all_scores_df.at[new_index, 'luca'] = parent_index

    # Add cat res to new entry
    save_cat_res_into_all_scores_df(self, new_index, 
                                   f'{self.FOLDER_PARENT}/{PARENTS[selected_index]}',
                                   save_resn=False)
    
    # Difficult to set kbt_boltzmann of first design. Here we just assign it the number of the second design
    if new_index == 1: 
        self.all_scores_df.at[new_index-1, 'kbt_boltzmann'] = self.all_scores_df.at[new_index, 'kbt_boltzmann']

    # Start design
    if self.PARENT_DES_MED in ["ProteinMPNN","LigandMPNN"]:
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

def start_calculation(self, parent_index):
    """
    Decides the next calculation step for the specified design variant index.

    Based on the current design state, this function decides to run ESMfold, RosettaRelax, or a design method
    for the given index.

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
        run_design(self, parent_index, ["ESMfold", "MDMin", "RosettaRelax", "ElectricFields"])  
    
    # Check if RosettaRelax is done    
    elif not f"{self.WT}_RosettaRelax_{parent_index}.pdb" in os.listdir(os.path.join(self.FOLDER_HOME, str(parent_index))):
        logging.info(f"Index {parent_index} has no relaxed structure, starting RosettaRelax.")
        self.all_scores_df.at[parent_index, "blocked_RosettaRelax"] = True 
        run_design(self, parent_index, ["RosettaRelax", "ElectricFields"])    

    # If all OK, start Design
    else:

        # RosettaRelax is done, create a new index
        new_index = create_new_index(self, parent_index)

        # Add cat res to new entry
        save_cat_res_into_all_scores_df(self, new_index, 
                                       f"{self.FOLDER_HOME}/{parent_index}/{self.WT}_RosettaRelax_{parent_index}.pdb",
                                       save_resn=False)
        
        #####
        # Here, we can add an AI to decide on the next steps
        #####

        # Run Design with new_index --> need to add code to CHECK IF self.ProteinMPNN_PROB + self.LMPNN_PROB is below 1!!!!
        if random.random() < self.ProteinMPNN_PROB:  
            self.all_scores_df.at[new_index, 'design_method'] = "ProteinMPNN"
            run_design(self, new_index, ["ProteinMPNN", "ESMfold", "RosettaRelax", "ElectricFields"])
        elif random.random()+self.ProteinMPNN_PROB < self.LMPNN_PROB+self.ProteinMPNN_PROB:
            self.all_scores_df.at[new_index, 'design_method'] = "LigandMPNN"
            run_design(self, new_index, ["LigandMPNN", "ESMfold", "RosettaRelax", "ElectricFields"])
        else:                    
            self.all_scores_df.at[new_index, 'design_method'] = "RosettaDesign"
            run_design(self, new_index, ["RosettaDesign", "ElectricFields"])
        
    save_all_scores_df(self)
        

def create_new_index(self, parent_index):
    """
    Creates a new design entry in `all_scores_df` with a unique index, inheriting attributes from a parent variant.

    Updates the DataFrame with new index information, saves the updated file, and sets up the directory structure
    for the new design variant.

    Parameters:
        parent_index (str): The index of the parent variant or "Parent" for initial designs.

    Returns:
        int: The newly created index for the variant.
    """
    
    # Create a new line with the next index and parent_index
    new_index = len(self.all_scores_df)
    
    # Append the new line to the DataFrame and save to  all_scores_df.csv
    if isinstance(self.KBT_BOLTZMANN, (float, int)):
        kbt_boltzmann = self.KBT_BOLTZMANN
    elif len(self.KBT_BOLTZMANN) == 2:
        num_pdb_files = len([file for file in os.listdir(self.FOLDER_PARENT) if file.endswith('.pdb')])
        datapoints = max(self.all_scores_df['index'].max() +1 - num_pdb_files*self.N_PARENT_JOBS, 0)
        kbt_boltzmann = max(self.KBT_BOLTZMANN[0] * np.exp(-self.KBT_BOLTZMANN[1]*datapoints), 0.05)
    if parent_index == 'Parent':
        generation = 0
        luca = "x"
    else:
        generation = self.all_scores_df['generation'][int(parent_index)]+1
        luca       = self.all_scores_df['luca'][int(parent_index)]
        
    new_index_df = pd.DataFrame({'index': int(new_index), 
                                'parent_index': parent_index,
                                'kbt_boltzmann': kbt_boltzmann,
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