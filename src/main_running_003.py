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
import getpass
import itertools

from helper_002               import *
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
    while len(self.all_scores_df) < int(self.MAX_DESIGNS): 
            
        # Check how many jobs are currently running
        num_running_jobs = check_running_jobs(self)
        
        # Wait if the number is equal or bigger than the maximum number of jobs.
        if num_running_jobs >= self.MAX_JOBS: 
            
            time.sleep(20)
        
        else:

            # Update the all_scores_df dataframe
            update_scores(self)
            
            # Selects variant based on calculations scheduled in all_scores_df
            selected_index = select_scheduled_variant(self)

            # Selects variant for redesign based on boltzmann selection
            if selected_index is None:
                
                parent_index = boltzmann_selection(self)

                if parent_index != None:
                    
                     selected_index = assign_design_method(self, parent_index)
            
            # Starts the design of the next variant using the selected index
            if selected_index != None: start_calculation(self, selected_index)
                
        # Sleep a bit for safety
        time.sleep(0.1)
        
    # When the maximum number of designs has been generated, the corresponding scores are calculated and added to the all_scores.csv file.
    update_scores(self)
    
    print(f"Stopped because {len(self.all_scores_df['index'])}/{self.MAX_DESIGNS} designs have been made.")

def select_scheduled_variant(self):

    # Iterate through dataframe to see if any variant is scheduled for calculation with GPU
    if self.MAX_GPUS > 0:
        if all(value is not None for value in self.gpus.values()): return None # Check if there is a free GPU
        for index, row in self.all_scores_df.iloc[::-1].iterrows():
            if pd.isna(row['next_steps']) or row['next_steps'] == "": continue
            if row['blocked'] != 'unblocked': continue   
            print(f"check if row['next_steps'] contain a job amenable to GPU")
            
    # Iterate through dataframe to see if any variant is scheduled for calculation
    for index, row in self.all_scores_df.iloc[::-1].iterrows():
        print("row['next_steps']", index, row['next_steps'])
        if pd.isna(row['next_steps']) or row['next_steps'] == "": continue
        if row['blocked'] != 'unblocked': continue    
        return index
    return None
    
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
        command = ["ssh", f"{getpass.getuser()}@bs-submit04.ethz.ch", "qstat", "-u", getpass.getuser()]
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
    with open(parent_filename, "a") as f:  f.write(f"\n{str(score)}") # Appends to parent_filename
    with open(parent_filename, "r") as f:  potentials = f.readlines() # Reads in potential values 
    self.all_scores_df.at[int(parent_index), f'{score_type}_potential'] = np.average([float(i) for i in potentials])

 
def update_scores(self):
    """
    Updates the all_scores dataframe.

    This function iterates over design variants, updating scores based on files generated by different processes.
    It also updates sequence information, tracks mutations, and saves the updated DataFrame.
    """

    logging.debug("Updating scores")
    
    display(self.all_scores_df)
       
    for index, row in self.all_scores_df.iterrows():

        parent_index = row['parent_index']         
        
        # Unblock indeces
        if self.all_scores_df.at[int(index), f"blocked"] != 'unblocked': 
            design_step = self.all_scores_df.at[int(index), f"blocked"]
            # Unblock indeces for runs that produce structures 
            if os.path.isfile(f"{self.FOLDER_HOME}/{index}/{self.WT}_{design_step}_{index}.pdb"):
                self.all_scores_df.at[int(index), f"blocked"] = 'unblocked'
                logging.debug(f"Unblocked {design_step} index {int(index)}.")
            
        seq_path        = f"{self.FOLDER_HOME}/{index}/{self.WT}_{index}.seq"
        ref_seq_path    = f"{self.FOLDER_PARENT}/{self.WT}.seq"
        parent_seq_path = f"{self.FOLDER_HOME}/{parent_index}/{self.WT}_{parent_index}.seq"     
        if parent_index == "Parent": parent_seq_path = ref_seq_path
        
        # Update sequence and mutations if row does not yet contain a sequence
        if pd.isna(self.all_scores_df.at[index, 'sequence']):
            if os.path.exists(seq_path):
                
                with open(ref_seq_path, "r") as f:    
                    reference_sequence = f.read()
                with open(seq_path, "r") as f:        
                    current_sequence = f.read()
                with open(parent_seq_path, "r") as f:
                    parent_sequence = f.read()

                self.all_scores_df['sequence'] = self.all_scores_df['sequence'].astype('object')
                self.all_scores_df.at[index, 'sequence']  = current_sequence
                self.all_scores_df.at[index, 'total_mutations'] = count_mutations(reference_sequence, current_sequence)
                self.all_scores_df.at[index, 'parent_mutations'] = count_mutations(parent_sequence, current_sequence)

        # Calculate identical score
        identical_score = 0.0
        if "identical" in self.SELECTED_SCORES:        
            sequence = self.all_scores_df.at[index, 'sequence']
            parent_index = self.all_scores_df.at[index, 'parent_index'] 
            if pd.notna(sequence):
                if parent_index == 'Parent':
                    identical_score = 1.0 
                else:
                    identical_count = (self.all_scores_df['sequence'] == sequence).sum()
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
        self.all_scores_df.at[int(index), f"blocked"] = 'unblocked'
        logging.debug(f"Unblocked {design_step} index {int(index)}.")
        
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

    # Get unblocked structures
    unblocked_all_scores_df = self.all_scores_df
    unblocked_all_scores_df = unblocked_all_scores_df[unblocked_all_scores_df["blocked"] == 'unblocked']
             
    # Drop catalytic scroes > mean + 1 std
    if "catalytic" in self.SELECTED_SCORES:
        mean_catalytic_score = unblocked_all_scores_df['catalytic_score'].mean()
        std_catalytic_score = unblocked_all_scores_df['catalytic_score'].std()
        mean_std_catalytic_score = mean_catalytic_score + std_catalytic_score
        if len(unblocked_all_scores_df) > 10:
            unblocked_all_scores_df = unblocked_all_scores_df[unblocked_all_scores_df['catalytic_score'] < mean_std_catalytic_score]
        
    # Remove indices without score (design running)
    unblocked_all_scores_df = unblocked_all_scores_df.dropna(subset=['total_score'])     
  
    # If there are structures that ran through RosettaRelax but have never been used for design, run design
    unrelaxed_indices = unblocked_all_scores_df[unblocked_all_scores_df['relax_total_score'].isna()]
    filtered_indices = [index for index in unrelaxed_indices.index if index not in parent_indices]
    if len(filtered_indices) >= 1:
        selected_index = filtered_indices[-1]
        logging.info(f"{selected_index} selected because it's relaxed but nothing was designed from it.")
        return selected_index
    
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
            selected_index = np.random.choice(unblocked_all_scores_df.index.to_numpy(), p=probabilities)
        else:
            logging.debug(f'Boltzmann selection tries to select a variant for design, but all are blocked. Waiting 20 seconds')
            time.sleep(20)
            return None
        
    else:
        selected_index = 0
        
    return selected_index

def assign_design_method(self, parent_index):

    # CHECK if self.all_scores_df has nan in relaxed_total_score
    if pd.isna(self.all_scores_df.iloc[parent_index]["relax_total_score"]):

        # Assing scoring relax methods
        final_structure_method = [i for i in self.SCORING_METHODS if i in self.SYS_STRUCT_METHODS][-1]
        print(final_structure_method,"Xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print(self.SCORING_METHODS,"Xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print(self.SYS_STRUCT_METHODS,"Xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        final_variant = f'{self.FOLDER_HOME}/{parent_index}/{self.WT}_{final_structure_method}_{parent_index}.pdb',
        self.all_scores_df.at[parent_index, "final_variant"] = final_variant
        self.all_scores_df.at[parent_index, "next_steps"] = ",".join(self.SCORING_METHODS)
        
        return parent_index

    else:
        
        # Select design method from self.DESIGN_METHOD
        probabilities = [i[0] for i in self.DESIGN_METHODS]  # Extract probabilities
        cumulative_probs = list(itertools.accumulate(probabilities))  # Cumulative sum
        rnd = random.random()
        for idx, cp in enumerate(cumulative_probs):
            if rnd < cp:
                design_methods = self.DESIGN_METHODS[idx][1:]  # Select the corresponding design method
                break
                
        # Define intial and final structures
        final_structure_method = [i for i in design_methods if i in self.SYS_STRUCT_METHODS][-1]
        design_method = [i for i in design_methods if i in self.SYS_DESIGN_METHODS][0]

        # Make new index
        new_index = create_new_index(self, 
                                     parent_index  = parent_index, 
                                     luca          = self.all_scores_df[parent_index, 'luca'],
                                     input_variant = self.all_scores_df[parent_index, 'final_variant'],
                                     final_method  = final_structure_method,
                                     next_steps    = ",".join(design_methods), 
                                     design_method = design_method)    
        return new_index
    
# Decides what to do with selected index

def start_calculation(self, selected_index: int):
        
    logging.debug(f"Starting new calculation for index {selected_index}.")
     
    # Check if index is still blocked, if yes --> STOP. This shouldn't happen!
    if self.all_scores_df.at[selected_index, "blocked"] != 'unblocked':
        logging.error(f"Index {selected_index} is being worked on. Skipping index.")
        logging.error(f"Note: This should not happen! Check blocking and Boltzman selection.")        
        return

    next_steps = self.all_scores_df.at[selected_index, "next_steps"]
    next_steps = next_steps.split(",")
    
    self.all_scores_df.at[selected_index, "next_steps"] = ",".join(next_steps[1:])
    self.all_scores_df.at[selected_index, "blocked"] = next_steps[0]
    if next_steps[0] in self.SYS_STRUCT_METHODS:
        latest_variant = f'{self.FOLDER_HOME}/{selected_index}/{self.WT}_{next_steps[0]}_{selected_index}.pdb'
        self.all_scores_df["latest_variant"] = self.all_scores_df["latest_variant"].astype(str)
        self.all_scores_df.at[selected_index, "latest_variant"] = latest_variant
    
    save_all_scores_df(self)
    
    run_design(self, selected_index, [next_steps[0]])