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
    
<<<<<<< HEAD
    while len(self.all_scores_df['index']) < int(self.MAX_DESIGNS): #Run until MAX_DESIGNS are made
=======
    while not os.path.exists(os.path.join(self.FOLDER_HOME, str(self.MAX_DESIGNS))): #Run until MAX_DESIGNS are made
       
        # Update scores
        update_scores(self)
>>>>>>> 5c737f120964d77d7e3de7727e8dc03e1a5380a9
            
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
    
    if self.SYSTEM == 'GRID':
        jobs = subprocess.check_output(["qstat", "-u", self.USERNAME]).decode("utf-8").split("\n")
        jobs = [job for job in jobs if self.SUBMIT_PREFIX in job]
        return len(jobs)
        
    if self.SYSTEM == 'BLUEPEBBLE':
        jobs = subprocess.check_output(["squeue","--me"]).decode("utf-8").split("\n")
        jobs = [job for job in jobs if self.SUBMIT_PREFIX in job]
        return len(jobs)
        
    if self.SYSTEM == 'BACKGROUND_JOB':
        with open(f'{self.FOLDER_HOME}/n_running_jobs.dat', 'r'): jobs = int(f.read())
        return jobs
    
    if self.SYSTEM == 'ABBIE_LOCAL':
        return 0

def update_potential(self, score_type, index):
    
    '''Creates a <score_type>_potential.dat file in FOLDER_HOME/<index> 
    
    If latest score comes from Rosetta Relax - then the score for variant <index>
    will be added to the <score_type>_potential.dat of the parent index 
    and the <score_type>_potential value of the dataframe for the parent 
    will be updated with the average of the parent and child scores. 
    
    Parameters:
    - score_type(str): Type of score to update, one of these options: total, interface, catalytic, efield
    - index (int): variant index to update
    
    '''
    
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
            
            # Do NOT update score to prevent repeated scoring!
            if row['score_taken_from'] == 'RosettaRelax': continue 
                
            # Set paths
            score_file_path = f"{self.FOLDER_HOME}/{int(index)}/score_RosettaRelax.sc"
            pdb_path = f"{self.FOLDER_HOME}/{int(index)}/{self.WT}_RosettaRelax_{int(index)}.pdb"
                        
        elif os.path.exists(f"{self.FOLDER_HOME}/{int(index)}/score_RosettaDesign.sc"): # Score based on RosettaDesign
            
            # Do NOT update score to prevent repeated scoring!
            if row['score_taken_from'] == 'RosettaDesign': continue 
                
            # Set paths
            score_file_path = f"{self.FOLDER_HOME}/{int(index)}/score_RosettaDesign.sc" 
            pdb_path = f"{self.FOLDER_HOME}/{int(index)}/{self.WT}_RosettaDesign_{int(index)}.pdb"
                        
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
    
        # Do not update score if files do not exist!
        if not os.path.isfile(pdb_path): continue
        if not os.path.isfile(seq_path): continue
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
        save_cat_res_into_all_scores_df(self, index, pdb_path) 
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
        efield_score, index_efields_dict = calc_efields_score(self, pdb_path, cat_res)  
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

def normalize_scores(self, unblocked_all_scores_df, include_catalytic_score=False, print_norm=False, norm_all=False, extension="score"):
    
    def neg_norm_array(array, score_type):

        if len(array) > 1:  ##check that it's not only one value
            
            array    = -array
            
            if norm_all:
                if print_norm:
                    print(score_type,NORM[score_type],end=" ")
                array = (array-NORM[score_type][0])/(NORM[score_type][1]-NORM[score_type][0])
                array[array < 0] = 0.0
                if np.any(array > 1.0): print("\nNORMALIZATION ERROR!",score_type,"has a value >1!") 
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
         
    catalytic_scores    = unblocked_all_scores_df[f"catalytic_{extension}"]
    catalytic_scores    = neg_norm_array(catalytic_scores, f"catalytic_{extension}")   
    
    total_scores        = unblocked_all_scores_df[f"total_{extension}"]
    total_scores        = neg_norm_array(total_scores, f"total_{extension}")   
    
    interface_scores    = unblocked_all_scores_df[f"interface_{extension}"]
    interface_scores    = neg_norm_array(interface_scores, f"interface_{extension}")  
    
    efield_scores    = unblocked_all_scores_df[f"efield_{extension}"]   ### to be worked on
    efield_scores    = neg_norm_array(-1*efield_scores, f"efield_{extension}")   ### to be worked on, with MINUS here
    
    if len(total_scores) == 0:
        combined_scores = []
    else:
        if include_catalytic_score:
            combined_scores     = np.stack((total_scores, interface_scores, efield_scores, catalytic_scores))
        else:
            combined_scores     = np.stack((total_scores, interface_scores, efield_scores))
        combined_scores     = np.mean(combined_scores, axis=0)
        
          
    if print_norm:
        if combined_scores.size > 0:
            print("HIGHSCORE:","{:.2f}".format(np.amax(combined_scores)),end=" ")
            print("Designs:",len(combined_scores),end=" ")
            PARENTS = [i for i in os.listdir(f'{FOLDER_HOME}/{FOLDER_PARENT}') if i[-4:] == ".pdb"]
            print("Parents:",len(PARENTS))
        
    return catalytic_scores, total_scores, interface_scores, efield_scores, combined_scores
        
def boltzmann_selection(self):
        
    parent_indices = set(self.all_scores_df['parent_index'].astype(str).values)
    
    unblocked_all_scores_df = self.all_scores_df
    
    # Remove blocked indices
    for unblock in ["RosettaRelax","ESMfold"]:                    
        unblocked_all_scores_df = unblocked_all_scores_df[unblocked_all_scores_df[f"blocked_{unblock}"] == False]
             
    # Complet ESMfold and RosettaRelax
    filtered_indices = unblocked_all_scores_df[unblocked_all_scores_df['score_taken_from'] != 'RosettaRelax'] # Remove Relaxed Indeces
    
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
        else:
            # Ramp down kbT_boltzmann over time (i.e., with increaseing indices)
            # datapoints = legth of all_scores_df - number of parents generated
            num_pdb_files = len([file for file in os.listdir(self.FOLDER_PARENT) if file.endswith('.pdb')])
            datapoints = max(self.all_scores_df['index'].max() + 1 - num_pdb_files*self.N_PARENT_JOBS, 0)
            kbt_boltzmann = float(max(self.KBT_BOLTZMANN[0] * np.exp(-self.KBT_BOLTZMANN[1]*datapoints), 0.05))
        
        boltzmann_factors = np.exp(combined_potentials / kbt_boltzmann)
        probabilities = boltzmann_factors / sum(boltzmann_factors)
        
        if len(unblocked_all_scores_df) > 0:
            selected_index = int(np.random.choice(unblocked_all_scores_df["index"].to_numpy(), p=probabilities))
        else:
            return None
        
    else:
        
        selected_index = 0

    return selected_index

def check_parent_done(self):
       
    number_of_indices = len(self.all_scores_df)
    parents = [i for i in os.listdir(self.FOLDER_PARENT) if i[-4:] == ".pdb"]
    if number_of_indices < self.N_PARENT_JOBS * len(parents):
        parent_done = False
        logging.debug(f'Parent design not yet done. {number_of_indices+1}/{self.N_PARENT_JOBS * len(parents)} jobs submitted.')
    else:
        parent_done = True
        
    return parent_done
    
def start_parent_design(self):
    
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

    run_design(self, new_index, [self.PARENT_DES_MED])

    save_all_scores_df(self)
    
# Decides what to do with selected index
def start_calculation(self, parent_index):
    
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
        run_design(self, parent_index, ["ESMfold"])  
    
    # Check if RosettaRelax is done    
    elif not f"{self.WT}_RosettaRelax_{parent_index}.pdb" in os.listdir(os.path.join(self.FOLDER_HOME, str(parent_index))):
        logging.info(f"Index {parent_index} has no relaxed structure, starting RosettaRelax.")
        self.all_scores_df.at[parent_index, "blocked_RosettaRelax"] = True 
        run_design(self, parent_index, ["RosettaRelax"])

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

        # Run Design with new_index --> PROBABILITY not 100% accurate!!!!
        if random.random() < self.ProteinMPNN_PROB:  
            self.all_scores_df.at[new_index, 'design_method'] = "ProteinMPNN"
            run_design(self, new_index, ["ProteinMPNN"])
            
        elif random.random() < self.LMPNN_PROB:
            self.all_scores_df.at[new_index, 'design_method'] = "LigandMPNN"
            run_design(self, new_index, ["LigandMPNN"])
        else:                    
            self.all_scores_df.at[new_index, 'design_method'] = "RosettaDesign"
            run_design(self, new_index, ["RosettaDesign"])
        
    save_all_scores_df(self)
        
def create_new_index(self, parent_index):
    
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
                                }, index = [0])
    
    self.all_scores_df = pd.concat([self.all_scores_df, new_index_df], ignore_index=True)

    save_all_scores_df(self)

    # Create the folders for the new index
    os.makedirs(f"{self.FOLDER_HOME}/{new_index}/scripts", exist_ok=True)
           
    logging.debug(f"Child index {new_index} created for {parent_index}.")
    
    return new_index