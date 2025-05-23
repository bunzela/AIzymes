"""
Contains the main control functions for running AIzymes, including job submission,
score updating, and Boltzmann selection. The functions in this file are responsible for the high-level management
of the design process, interacting with the AIzymes_MAIN class to initiate, control, and evaluate design variants.

Functions:
    start_controller():         Runs the design loop until the maximum number of designs is reached.
    check_running_jobs():       Returns the count of current and scheduled jobs.
    update_potential():         Appends new scores to potential files and recalculates averages.
    update_scores():            Updates variant scores from outputs and unblocks completed jobs.
    boltzmann_selection():      Chooses a variant based on Boltzmann-weighted potentials.
    check_parent_done():        Checks if a parent variant's design jobs are finished.
    start_parent_design():      Initiates design processing for parent variants.
    start_calculation():        Launches the next design step for a selected variant.
    create_new_index():         Creates a new design index, updates scores, and logs the new entry.

Modules Required:
    helper_002
    main_design_001
    scoring_efields_001
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
import glob
import datetime
from dateutil.parser import parse
import tarfile

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
    while self.all_scores_df[f'{self.SELECTED_SCORES[0]}_score'].notna().sum() < int(self.MAX_DESIGNS): 

        # Update the all_scores_df dataframe
        update_scores(self)
        
        # Check how many jobs are currently running
        num_running_jobs = check_running_jobs(self)
        
        # Wait if the number of running or scheduled jobs is equal or bigger than the maximum number of jobs.
        if num_running_jobs['running'] >= self.MAX_JOBS: 
            
            time.sleep(1)
            
        else:           
            
            # Selects variant based on calculations scheduled in all_scores_df
            selected_index = select_scheduled_variant(self)
            
            if selected_index != None:

                # Starts the design of the next variant using the selected index
                start_calculation(self, selected_index)

            else:
                
                # Schedules calculations based on boltzmann selection                
                parent_index = boltzmann_selection(self)
                
                # Ensure design does not "run ahead" without completing design
                if parent_index != None and num_running_jobs['scheduled'] < self.MAX_JOBS*2: 
                                    
                    schedule_design_method(self, parent_index)

        # Wait a bit for safety
        time.sleep(0.1)

    # Final score update
    update_scores(self)
    print(f"Stopped because {len(self.all_scores_df)}/{self.MAX_DESIGNS} designs have been made.")

def select_scheduled_variant(self):
    
    # Iterate through dataframe to see if any variant is scheduled for calculation with GPU
    if self.MAX_GPUS > 0:

        if any(value is None for value in self.gpus.values()): # Check if there is a free GPU

            # First, try to find an index where next_steps is a SYS_GPU_METHODS but not 'AlphaFold3INF'
            for index, row in self.all_scores_df.iterrows():
                if pd.isna(row['next_steps']) or row['next_steps'] == "": continue
                if row['blocked'] != 'unblocked': continue
                if row['next_steps'].split(',')[0] == 'AlphaFold3INF': continue
                if row['next_steps'].split(',')[0] not in self.SYS_GPU_METHODS: continue
                return index
    
            # If none found, then try to find an index where next_steps is 'AlphaFold3INF'
            for index, row in self.all_scores_df.iterrows():
                if pd.isna(row['next_steps']) or row['next_steps'] == "": continue
                if row['blocked'] != 'unblocked': continue
                if row['next_steps'].split(',')[0] not in self.SYS_GPU_METHODS: continue
                return index
    
    # Iterate through dataframe to see if any variant is scheduled for calculation
    for index, row in self.all_scores_df.iterrows():
        if pd.isna(row['next_steps']) or row['next_steps'] == "": continue        
        if row['blocked'] != 'unblocked': continue        
        if row['next_steps'].split(',')[0] in self.SYS_GPU_METHODS: continue    
        return index
    
    return None
       
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
    parent_filename = f"{self.FOLDER_DESIGN}/{parent_index}/{score_type}_potential.dat"  

    # Update current potential
    with open(f"{self.FOLDER_DESIGN}/{index}/{score_type}_potential.dat", "w") as f: 
        f.write(str(score))
    self.all_scores_df.at[index, f'{score_type}_potential'] = score

    #Update parent potential
    if score_taken_from != "RosettaRelax":  return                     # Only update the parent potential for RosettaRelax
    if parent_index == "Parent":            return                     # Do not update the parent potential of a variant from parent
    if not os.path.isfile(parent_filename): return                     # Do not update if this is a potential not in the top1000 designs anymore
    with open(parent_filename, "a") as f:  f.write(f"\n{str(score)}")  # Appends to parent_filename
    with open(parent_filename, "r") as f:  potentials = f.readlines()  # Reads in potential values 
    self.all_scores_df.at[int(parent_index), f'{score_type}_potential'] = np.average([float(i) for i in potentials])

def update_scores(self):
    """
    Updates the all_scores dataframe.

    This function iterates over design variants, updating scores based on files generated by different processes.
    It also updates sequence information, tracks mutations, and saves the updated DataFrame.
    """

    logging.debug("Updating scores")
           
    for index, row in self.all_scores_df.iterrows():
        
        parent_index = row['parent_index']         

        # Find failed designs
        if self.all_scores_df.at[index, "blocked"] == 'failed': continue
        if self.all_scores_df.at[index, "blocked"] != 'unblocked': 
            
            if os.path.isfile(f"{self.FOLDER_DESIGN}/{index}/ROSETTA_CRASH.log"):   
                self.all_scores_df.at[int(index), "blocked"] = 'failed'

            error_files = glob.glob(os.path.join(self.FOLDER_HOME, str(index), "scripts", "*.err"))
            for err_file in error_files:
                with open(err_file, "r", errors="ignore") as f:
                    if any("error" in line.lower() for line in f):
                        self.all_scores_df.at[int(index), "blocked"] = 'failed'
                
        # Unblock indices
        if self.all_scores_df.at[index, "blocked"] != 'unblocked': 
            
            score_type = self.all_scores_df.at[index, "blocked"]
            pdb_path = self.all_scores_df.at[index, "final_variant"]
            final_method = os.path.basename(pdb_path)
            final_method = final_method.split("_")[-2]   
            
            # Unblock indices for runs that produce structures
            if self.all_scores_df.at[int(index), "blocked"] in self.SYS_STRUCT_METHODS:
                if os.path.isfile(f"{self.FOLDER_DESIGN}/{index}/{self.WT}_{score_type}_{index}.pdb"):
                    self.all_scores_df.at[int(index), "blocked"] = 'unblocked'
                    logging.debug(f"Unblocked {score_type} index {int(index)}.")
                    
            # Unblock indices for ElectricFields
            if self.all_scores_df.at[int(index), "blocked"] == 'ElectricFields':
                if os.path.isfile(f"{self.FOLDER_DESIGN}/{index}/ElectricFields/{self.WT}_{final_method}_{index}_fields.pkl"):
                    self.all_scores_df.at[int(index), "blocked"] = 'unblocked'
                    logging.debug(f"Unblocked {score_type} index {int(index)}.")                    
                    
            # Unblock indices for BioDC
            if self.all_scores_df.at[int(index), "blocked"] == 'BioDC':
                if os.path.isfile(f"{self.FOLDER_DESIGN}/{index}/BioDC/{self.WT}_{final_method}_{index}/EE/DG.txt"):
                    self.all_scores_df.at[int(index), "blocked"] = 'unblocked'
                    logging.debug(f"Unblocked {score_type} index {int(index)}.")
                    
            # Unblock indices for Alphafold3MSA
            if self.all_scores_df.at[int(index), "blocked"] == 'AlphaFold3MSA':
                if os.path.isfile(f"{self.FOLDER_DESIGN}/{index}/AlphaFold3/MSA/{self.WT}/{self.WT}_data.json"):
                    self.all_scores_df.at[int(index), "blocked"] = 'unblocked'
                    logging.debug(f"Unblocked {score_type} index {int(index)}.")

            # Unblock indices for MPNN:
            if self.all_scores_df.at[int(index), "blocked"] in ['ProteinMPNN','LigandMPNN','SolubleMPNN']:
                if os.path.isfile(f"{self.FOLDER_DESIGN}/{index}/{self.WT}_{index}.seq"):
                    self.all_scores_df.at[int(index), "blocked"] = 'unblocked'
                    logging.debug(f"Unblocked {score_type} index {int(index)}.")
                          
        # Paths for sequence-based information
        seq_path        = f"{self.FOLDER_DESIGN}/{index}/{self.WT}_{index}.seq"
        ref_seq_path    = f"{self.FOLDER_PARENT}/{self.WT}.seq"
        
        # Update sequence and mutations if not yet contained in dataframe
        if pd.isna(self.all_scores_df.at[index, 'sequence']) and os.path.exists(seq_path):
   
            with open(ref_seq_path, "r") as f:    
                reference_sequence = f.read()
            with open(seq_path, "r") as f:        
                current_sequence = f.read()

            if parent_index == "Parent":
                parent_sequence = reference_sequence
            else:
                parent_sequence = self.all_scores_df.at[parent_index, 'sequence']

            self.all_scores_df['sequence'] = self.all_scores_df['sequence'].astype('object')
            self.all_scores_df.at[index, 'sequence']  = current_sequence
            self.all_scores_df.at[index, 'active_site_res']  = ''.join(current_sequence[int(pos)-1] for pos in self.DESIGN.split(','))
            self.all_scores_df.at[index, 'total_mutations'] = count_mutations(reference_sequence, current_sequence)
            self.all_scores_df.at[index, 'parent_mutations'] = count_mutations(parent_sequence, current_sequence)
        
            # Update catalytic residue identity
            if self.CST_NAME != None:
                cat_resns = []
                for cat_resi in str(self.all_scores_df.at[index, 'cat_resi']).split(";"): 
                    cat_resns += [one_to_three_letter_aa(current_sequence[int(float(cat_resi))-1])]
                self.all_scores_df['cat_resn'] = self.all_scores_df['cat_resn'].astype(str)
                self.all_scores_df.at[index, 'cat_resn'] = ";".join(cat_resns)
                    
        # Update identical score and potential
        if pd.notna(self.all_scores_df.at[index, 'sequence']) and "identical" in self.SELECTED_SCORES:

            if self.IDENTICAL_DESIGN:
                sequence_column = 'active_site_res'
            else:
                sequence_column = 'sequence'
                
            seq = self.all_scores_df.at[index, sequence_column]
            parent_index = self.all_scores_df.at[index, 'parent_index']

            if parent_index == 'Parent': # If it's a parent, identical score will be set to mean of scores
                if self.all_scores_df['identical_score'][self.N_PARENT_JOBS:].dropna().empty:
                    identical_score = 1
                else:
                    identical_score = self.all_scores_df['identical_score'][self.N_PARENT_JOBS:].dropna().mean()
                    
            else:
                identical_count = (self.all_scores_df[sequence_column] == seq).sum()
                identical_score = 1 / identical_count if identical_count > 0 else 0

            self.all_scores_df.at[index, 'identical_score'] = identical_score
            self.all_scores_df.at[index, 'identical_potential'] = identical_score

        # Check what structure to score on
        if os.path.exists(f"{self.FOLDER_DESIGN}/{int(index)}/score_RosettaRelax.sc"): # Score based on RosettaRelax            

            if row['score_taken_from'] == 'RosettaRelax': continue # Do NOT update score to prevent repeated scoring!
            score_type = 'RosettaRelax'
        
        elif os.path.exists(f"{self.FOLDER_DESIGN}/{int(index)}/score_RosettaDesign.sc"): # Score based on RosettaDesign

            if row['score_taken_from'] == 'RosettaDesign': continue # Do NOT update score to prevent repeated scoring! 
            score_type = 'RosettaDesign'
       
        elif os.path.exists(f"{self.FOLDER_DESIGN}/{int(index)}/score_RosettaDesign.sc"): # Score based on RosettaDesign

            if row['score_taken_from'] == 'RosettaDesign': continue # Do NOT update score to prevent repeated scoring! 
            score_type = 'RosettaDesign'
                    
        else:
            
            continue # Do NOT update scores, job is not done.

        # Update scores
        
        # Set paths
        score_file_path = f"{self.FOLDER_DESIGN}/{int(index)}/score_{score_type}.sc"
        pdb_path = self.all_scores_df.at[int(index), "final_variant"]
        final_method = os.path.basename(pdb_path)
        final_method = final_method.split("_")[-2]        
        field_path = f"{self.FOLDER_DESIGN}/{int(index)}/ElectricFields/{self.WT}_{final_method}_{index}_fields.pkl"
        redox_path = f"{self.FOLDER_DESIGN}/{int(index)}/BioDC/{self.WT}_{final_method}_{index}/EE/DG.txt"
        
        # Check if everything is done   
        if not os.path.isfile(f'{pdb_path}.pdb'): continue
        if not os.path.isfile(seq_path): continue
        if not os.path.exists(field_path) and "efield" in self.SELECTED_SCORES: continue 
        if not os.path.exists(redox_path) and "redox" in self.SELECTED_SCORES: continue 
        
        # Unblock structures
        if self.all_scores_df.at[int(index), f"blocked"] != 'unblocked':
            logging.debug(f"Unblocked {self.all_scores_df.at[int(index), f'blocked']} index {int(index)}.")
            self.all_scores_df.at[int(index), f"blocked"] = 'unblocked'

        # Save cat resn
        if self.CST_NAME is not None:
            save_cat_res_into_all_scores_df(self, index, pdb_path, save_resn=True)
        
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
        
        # Calculate scores
        catalytic_score = 0.0
        interface_score = 0.0
        efield_score = 0.0
        total_score = 0.0
        redox_score = 0.0
        
        for idx_headers, header in enumerate(headers):

            # Calculate total score
            if "total" in self.SELECTED_SCORES:
                if header == 'total_score':                total_score           = float(scores[idx_headers])
                    
            # Subtract constraints from interface score
            if "interface" in self.SELECTED_SCORES:
                if header == 'interface_delta_X':          interface_score      += float(scores[idx_headers])
                if header in ['if_X_angle_constraint', 
                          'if_X_atom_pair_constraint', 
                          'if_X_dihedral_constraint']:     interface_score      -= float(scores[idx_headers])   
                    
            # Calculate catalytic score by adding constraints
            if "catalytic" in self.SELECTED_SCORES:
                
                if header in ['atom_pair_constraint']:     catalytic_score      += float(scores[idx_headers])       
                if header in ['angle_constraint']:         catalytic_score      += float(scores[idx_headers])       
                if header in ['dihedral_constraint']:      catalytic_score      += float(scores[idx_headers]) 
                    
                if header in ['atom_pair_constraint']:     self.all_scores_df.at[index, 'atom_pair_constraint'] = float(scores[idx_headers]) 
                if header in ['angle_constraint']:         self.all_scores_df.at[index, 'angle_constraint']     = float(scores[idx_headers]) 
                if header in ['dihedral_constraint']:      self.all_scores_df.at[index, 'dihedral_constraint']  = float(scores[idx_headers]) 
        
        # Calculate efield score
        if "efield" in self.SELECTED_SCORES:
            efield_score, index_efields_dict = get_efields_score(self, index, final_method)  
            update_efieldsdf(self, index, index_efields_dict)   

        if "redox" in self.SELECTED_SCORES:
            redox_score, redoxpotential = get_redox_score(self, index, final_method) 
            self.all_scores_df.at[index, 'BioDC_redox'] = redoxpotential
        
        # Update scores
        if "total"     in self.SELECTED_SCORES: self.all_scores_df.at[index, 'total_score']          = total_score
        if "interface" in self.SELECTED_SCORES: self.all_scores_df.at[index, 'interface_score']      = interface_score              
        if "catalytic" in self.SELECTED_SCORES: self.all_scores_df.at[index, 'catalytic_score']      = catalytic_score         
        if "efield"    in self.SELECTED_SCORES: self.all_scores_df.at[index, 'efield_score']         = efield_score
        if "redox"     in  self.SELECTED_SCORES: self.all_scores_df.at[index, 'redox_score']         = redox_score

        # This is just for book keeping. AIzymes will always use the most up_to_date scores saved above
        if "RosettaRelax" in score_file_path:
            if "total" in self.SELECTED_SCORES:     self.all_scores_df.at[index, 'relax_total_score']     = total_score
            if "interface" in self.SELECTED_SCORES: self.all_scores_df.at[index, 'relax_interface_score'] = interface_score         
            if "catalytic" in self.SELECTED_SCORES: self.all_scores_df.at[index, 'relax_catalytic_score'] = catalytic_score
            if "efield" in self.SELECTED_SCORES:    self.all_scores_df.at[index, 'relax_efield_score']    = efield_score
            if "redox" in self.SELECTED_SCORES:     self.all_scores_df.at[index, 'relax_redox_score']     = redox_score
            
        if "RosettaDesign" in score_file_path:
            if "total" in self.SELECTED_SCORES:     self.all_scores_df.at[index, 'design_total_score']     = total_score
            if "interface" in self.SELECTED_SCORES: self.all_scores_df.at[index, 'design_interface_score'] = interface_score        
            if "catalytic" in self.SELECTED_SCORES: self.all_scores_df.at[index, 'design_catalytic_score'] = catalytic_score
            if "efield" in self.SELECTED_SCORES:    self.all_scores_df.at[index, 'design_efield_score']    = efield_score
            if "redox" in self.SELECTED_SCORES:     self.all_scores_df.at[index, 'design_redox_score']     = redox_score

        for score_type in self.SELECTED_SCORES: 
            if score_type != "identical":
                update_potential(self, score_type=score_type, index= index)   

        logging.info(f"Updated scores and potentials of index {index}.")
        if self.all_scores_df.at[index, 'score_taken_from'] == 'Relax' and self.all_scores_df.at[index, 'parent_index'] != "Parent":
            logging.info(f"Adjusted potentials of {self.all_scores_df.at[index, 'parent_index']}, parent of {int(index)}).")

    save_all_scores_df(self)

    # Packup folders   
    for index, row in self.all_scores_df.iterrows():
        
        folder = f'{self.FOLDER_DESIGN}/{index}'
        tar_path = f'{folder}/{index}.tar'
        
        if pd.isna(self.all_scores_df.at[index, 'next_steps']): continue
        if self.all_scores_df.at[index, 'blocked'] != 'unblocked': continue    
        if pd.isna(self.all_scores_df.at[index, f'{self.SELECTED_SCORES[0]}_score']): continue
        if os.path.isfile(tar_path): continue
        if not os.path.isfile(f"{self.all_scores_df.at[index, 'final_variant']}.pdb"): continue
            
        subdirs  = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        subfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        with tarfile.open(tar_path, "w") as tar:
            
            # Tar subdirectories
            for subdir in subdirs:
                tar.add(f'{folder}/{subdir}', arcname=subdir)
                shutil.rmtree(f'{folder}/{subdir}')

            # Tar files
            for filename in subfiles:
                if '.tar' in filename: continue
                if '.seq' in filename: continue
                if 'potential' in filename: continue
                if os.path.basename(self.all_scores_df.at[index, 'final_variant']) in filename: continue
                tar.add(f'{folder}/{filename}', arcname="files")
                os.remove(f'{folder}/{filename}')

    update_resource_log(self)

def update_resource_log(self):

    last_time = int(self.resource_log_df['time'].iloc[-1])
    current_time = datetime.datetime.now().timestamp()
    
    if current_time - last_time < 60: return

    total_designs = len(self.all_scores_df)
    finished_designs = self.all_scores_df[f'{self.SELECTED_SCORES[0]}_score'].notna().sum()
    num_running_jobs = check_running_jobs(self)
    
    new_entry = pd.DataFrame({
        'time': int(current_time),
        'cpus_used': num_running_jobs['running'],
        'gpus_used': num_running_jobs['running_gpu'],
        'total_designs': total_designs,
        'finished_designs': finished_designs,
        'unfinished_designs': num_running_jobs['scheduled'],
        'failed_designs': num_running_jobs['failed'],
        'kbt_boltzmann': self.all_scores_df["kbt_boltzmann"].iloc[-1],
    }, index = [0] , dtype=object)  
                                      
    self.resource_log_df = pd.concat([self.resource_log_df, new_entry], ignore_index=True)
    save_resource_log_df(self)
                                          
def boltzmann_selection(self):
    """
    Selects a design variant based on a Boltzmann-weighted probability distribution.

    Filters variants based on certain conditions (e.g., scores, block status), then computes probabilities
    using Boltzmann factors with a temperature factor (`KBT_BOLTZMANN`) to select a variant for further design steps.

    Returns:
        int: Index of the selected design variant.
    """

    # Remove indices for which calculations have been scheduled
    filtered_all_scores_df = self.all_scores_df[self.all_scores_df['next_steps'].isna() | (self.all_scores_df['next_steps'] == "")]
    
    # Get list of all indices (blocked and unblocked_
    parent_indices = set(filtered_all_scores_df['parent_index'].astype(str).values)

    # Get unblocked structures
    unblocked_all_scores_df = filtered_all_scores_df[filtered_all_scores_df["blocked"] == 'unblocked']
             
    # Drop catalytic scroes > mean + 1 std
    if "catalytic" in self.SELECTED_SCORES:
        mean_catalytic_score = unblocked_all_scores_df['catalytic_score'].mean()
        std_catalytic_score = unblocked_all_scores_df['catalytic_score'].std()
        mean_std_catalytic_score = mean_catalytic_score + std_catalytic_score
        if len(unblocked_all_scores_df) > 10:
            unblocked_all_scores_df = unblocked_all_scores_df[unblocked_all_scores_df['catalytic_score'] < mean_std_catalytic_score]
             
    # Drop Variants with distances greater than self.DISTANCE_CUTOFF (in RosettaScores, NOT Angstrom!)
    if "catalytic" in self.SELECTED_SCORES and self.CST_DIST_CUTOFF:
        if len(unblocked_all_scores_df) > 10:
            unblocked_all_scores_df = unblocked_all_scores_df[unblocked_all_scores_df['atom_pair_constraint'] < self.CST_DIST_CUTOFF]
            
    # Remove indices without score (design running)
    unblocked_all_scores_df = unblocked_all_scores_df.dropna(subset=[f'{self.SELECTED_SCORES[0]}_score'])     
  
    # If there are structures that ran through RosettaRelax but have never been used for design, run design
    # CANNOT HAPPEN ANYMORE! LIKELY NOT NEEDED> COMMENTED OUT BY AB
    #unrelaxed_indices = unblocked_all_scores_df[unblocked_all_scores_df[f'{self.SELECTED_SCORES[0]}_score'].isna()]
    #filtered_indices = [index for index in unrelaxed_indices.index if index not in parent_indices]
    #if len(filtered_indices) >= 1:
    #    selected_index = filtered_indices[-1]
    #    logging.info(f"{selected_index} selected because it's relaxed but nothing was designed from it.")
    #    return selected_index
    
    # Do Boltzmann Selection if some scores exist
    scores = normalize_scores(self, 
                            unblocked_all_scores_df, 
                            norm_all=False, 
                            extension="potential") 
        
    combined_potentials = scores["combined_potential"]
    
    if len(combined_potentials) > 0:
        
        generation=self.all_scores_df['generation'].max()
                
        if isinstance(self.KBT_BOLTZMANN, (float, int)):
            kbt_boltzmann = self.KBT_BOLTZMANN

        elif len(self.KBT_BOLTZMANN) == 2:
            kbt_boltzmann = self.KBT_BOLTZMANN[0] * np.exp(-self.KBT_BOLTZMANN[1]*generation)

        elif len(self.KBT_BOLTZMANN) == 3:
            kbt_boltzmann = (self.KBT_BOLTZMANN[0]-self.KBT_BOLTZMANN[2])*np.exp(-self.KBT_BOLTZMANN[1]*generation)+self.KBT_BOLTZMANN[2]
        
        # Some issue with numpy exp when calculating boltzman factors.
        combined_potentials_list = [float(x) for x in combined_potentials]
        combined_potentials = np.array(combined_potentials_list)
        
        boltzmann_factors = np.exp(combined_potentials / kbt_boltzmann)
        probabilities = boltzmann_factors / sum(boltzmann_factors)
        
        if len(unblocked_all_scores_df) > 0:
            
            selected_index = np.random.choice(unblocked_all_scores_df.index.to_numpy(), p=probabilities)

            # If an index was selected that was tarred before, do not do the Boltzmann selection!
            if os.path.isfile(f"{self.FOLDER_DESIGN}/{selected_index}.tar"):
                return None
                
        else:
            logging.debug(f'Boltzmann selection tries to select a variant for design, but all are blocked. Waiting 1 second')
            time.sleep(1)
            return None
        
    else:
        selected_index = 0
    
    return selected_index

def schedule_design_method(self, parent_index):

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
                                 parent_index   = parent_index, 
                                 luca           = self.all_scores_df.at[parent_index, 'luca'],
                                 input_variant  = self.all_scores_df.at[parent_index, 'final_variant'],
                                 final_method   = final_structure_method,
                                 next_steps     = ",".join(design_methods), 
                                 design_method  = design_method)    

    return 
    
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

    logging.debug(f"Starting calculation based on {self.all_scores_df.at[selected_index, 'step_input_variant']}")
    if next_steps[0] in self.SYS_STRUCT_METHODS:
        logging.debug(f"Resulting in variant {self.all_scores_df.at[selected_index, 'step_output_variant']}")
    
    if next_steps[0] in self.SYS_STRUCT_METHODS:

        logging.debug(f"before update {selected_index}")
        logging.debug(f'previous_input_variant_for_reset {self.all_scores_df.at[selected_index, "previous_input_variant_for_reset"]}')
        logging.debug(f'step_input_variant {self.all_scores_df.at[selected_index, "step_input_variant"]}')
        logging.debug(f'step_output_variant {self.all_scores_df.at[selected_index, "step_output_variant"]}')
        
        self.all_scores_df.at[selected_index, "previous_input_variant_for_reset"] = self.all_scores_df.at[selected_index, "step_input_variant"]
            # Keep track to reset job!
        self.all_scores_df.at[selected_index, "step_input_variant"] = self.all_scores_df.at[selected_index, "step_output_variant"]
        step_output_variant = f'{self.FOLDER_DESIGN}/{selected_index}/{self.WT}_{next_steps[0]}_{selected_index}'
        self.all_scores_df.at[selected_index, "step_output_variant"] = step_output_variant

        logging.debug(f"after update {selected_index}")
        logging.debug(f'previous_input_variant_for_reset {self.all_scores_df.at[selected_index, "previous_input_variant_for_reset"]}')
        logging.debug(f'step_input_variant {self.all_scores_df.at[selected_index, "step_input_variant"]}')
        logging.debug(f'step_output_variant {self.all_scores_df.at[selected_index, "step_output_variant"]}')

    run_design(self, selected_index, next_steps[0]) ### MOVED DOWN  BY HAB - BUG OR NOT????

    save_all_scores_df(self)
