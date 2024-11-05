import os
import sys
import time
import json
import logging
import pandas as pd
import json
import shutil

from helper_001               import *
from setup_system_001         import *
from main_scripts_001         import *

def initialize_controller(self, FOLDER_HOME):
    
    self.FOLDER_HOME = f'{os.getcwd()}/{FOLDER_HOME}'
    load_main_variables(self, self.FOLDER_HOME)
    
    # Starts the logger
    initialize_logging(self)
    
    # Stops run if variables file not found
    if not os.path.isfile(self.VARIABLES_JSON):
        print(f"Error, {self.VARIABLES_JSON} missing. Please setup AIZYMES with AIZYMES_setup")
        sys.exit()    
    
    if self.PRINT_VAR:
        if os.path.isfile(self.VARIABLES_JSON):
            with open(self.VARIABLES_JSON, 'r') as f: 
                variables_dict = json.load(f)
            for k, v in variables_dict.items():
                print(k.ljust(16), ':', v)
            
    # Stops run if FOLDER_HOME does not match
    if FOLDER_HOME != os.path.basename(self.FOLDER_HOME):
        print(f"Error, wrong FOLDER_HOME! Given {FOLDER_HOME}, required: {os.path.basename(self.FOLDER_HOME)}")
        sys.exit()
        
    if self.PLOT_DATA:
        plot_scores()
        
    # Read in current databases of AIzymes
    self.all_scores_df = pd.read_csv(self.ALL_SCORES_CSV)

    if self.UNBLOCK_ALL: 
        print(f'Unblocking all')
        self.all_scores_df["blocked_ESMfold"] = False
        self.all_scores_df["blocked_RosettaRelax"] = False
   
    # Sleep a bit to make me feel secure. More sleep = less anxiety :)
    time.sleep(0.1)

def prepare_input_files(self):
    
    os.makedirs(self.FOLDER_HOME, exist_ok=True)    
    os.makedirs(self.FOLDER_PARENT, exist_ok=True)
    os.makedirs(self.FOLDER_PLOT, exist_ok=True)

    # Creates general input scripts used by various programs 
    make_main_scripts(self)
        
    # Copy parent structure from Folder Input       
    if not os.path.isfile(f'{self.FOLDER_INPUT}/{self.WT}.pdb'):
        logging.error(f"Input structure {self.FOLDER_INPUT}/{self.WT}.pdb is missing!")
        sys.exit()       
    shutil.copy(f'{self.FOLDER_INPUT}/{self.WT}.pdb', f'{self.FOLDER_PARENT}/{self.WT}.pdb')
        
    # Save input sequence with X as wildcard 
    seq = sequence_from_pdb(f"{self.FOLDER_PARENT}/{self.WT}")
    with open(f'{self.FOLDER_PARENT}/{self.WT}.seq', "w") as f:
        f.write(seq)    
    design_positions = [int(x) for x in self.DESIGN.split(',')]
    
    # Replace seq with X at design positions. Note: Subtract 1 from each position to convert to Python's 0-based indexing
    seq = ''.join('X' if (i+1) in design_positions else amino_acid for i, amino_acid in enumerate(seq))
    with open(f'{self.FOLDER_HOME}/{self.WT}_with_X_as_wildecard.seq', 'w') as f:
        f.writelines(seq)    
    
    # Get the Constraint Residues from enzdes constraints file
    with open(f'{self.FOLDER_INPUT}/{self.CST_NAME}.cst', 'r') as f:
        cst = f.readlines()    
    cst = [i.split()[-1] for i in cst if "TEMPLATE::   ATOM_MAP: 2 res" in i]
    cst = ";".join(cst)
    with open(f'{self.FOLDER_HOME}/cst.dat', 'w') as f:
        f.write(cst)    

def initialize_variables(self):

    # Complete directories
    self.FOLDER_HOME     = f'{os.getcwd()}/{self.FOLDER_HOME}'
    self.FOLDER_PARENT   = f'{self.FOLDER_HOME}/{self.FOLDER_PARENT}'
    self.FOLDER_INPUT    = f'{os.getcwd()}/Input'
    self.USERNAME        = os.getlogin()
    self.LOG_FILE        = f'{self.FOLDER_HOME}/logfile.log'
    self.ALL_SCORES_CSV  = f'{self.FOLDER_HOME}/all_scores.csv'
    self.VARIABLES_JSON  = f'{self.FOLDER_HOME}/variables.json'
    self.FOLDER_PLOT     = f'{self.FOLDER_HOME}/plots' 
        
    # Define system-specific settings
    set_system(self)    
        
    if not os.path.isdir(self.FOLDER_INPUT):
        print(f"ERROR! Input folder missing! Should be {self.FOLDER_INPUT}")
        sys.exit()
        
def initialize_logging(self):

    os.makedirs(self.FOLDER_HOME, exist_ok=True)
    
    # Configure logging file
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Remove all handlers associated with the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Basic configuration for logging to a file
    if self.LOG == "debug":
        logging.basicConfig(filename=self.LOG_FILE, level=logging.DEBUG, format=log_format, datefmt=date_format)
    elif self.LOG == "info":
        logging.basicConfig(filename=self.LOG_FILE, level=logging.INFO, format=log_format, datefmt=date_format)
    else:
        logging.error(f'{self.LOG} not an accepted input for LOG')
        sys.exit()
        
    # Create a StreamHandler for console output
    console_handler = logging.StreamHandler()
    if self.LOG == "debug":
        console_handler.setLevel(logging.DEBUG)
    if self.LOG == "info":
        console_handler.setLevel(logging.INFO)

    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Add the console handler to the root logger
    logging.getLogger().addHandler(console_handler)

def aizymes_setup(self):
        
    # Defines all variables
    initialize_variables(self)
                
    # Starts the logger
    initialize_logging(self)
    
    # Check if setup needs to run
    if input(f'''Do you really want to restart AIzymes from scratch? 
    This will delete all existing files in {self.FOLDER_HOME} [y/n]

    ''') != 'y': 
        print("AIzymes reset aborted.")
        return # Do not reset. 
        
    if self.DESIGN == None:
        logging.error("Please define the designable residues with [DESIGN].")
        sys.exit()
    if self.WT == None:
        logging.error("Please define the name of the parent structure with [WT].")
        sys.exit()
    if self.LIGAND == None:
        logging.error("Please define the name of the ligand with [LIGAND].")
        sys.exit()
    if self.SUBMIT_PREFIX == None:
        logging.error("Please provide a unique prefix for job submission with [SUBMIT_PREFIX].")
        sys.exit()
    if self.SYSTEM == None:
        logging.error("Please define operating system with [SYSTEM]. {GRID,BLUPEBBLE}")
        sys.exit()
        
    if self.N_PARENT_JOBS < self.MAX_JOBS:
        logging.error(f"N_PARENT_JOBS must be > MAX_JOBS. N_PARENT_JOBS: {self.N_PARENT_JOBS}, MAX_JOBS: {self.MAX_JOBS}.")
        sys.exit()
          
    with open(self.LOG_FILE, 'w'): pass  #resets logfile
    
    logging.info(f"Running AI.zymes setup.")
    logging.info(f"Content of {self.FOLDER_HOME} deleted.")
    logging.info(f"Happy AI.zymeing! :)")
   
    for item in os.listdir(self.FOLDER_HOME):
        if item == self.FOLDER_MATCH: continue
        if item == self.FOLDER_PARENT: continue
        if item == self.FOLDER_INPUT: continue           
        item = f'{self.FOLDER_HOME}/{item}'
        if os.path.isfile(item): 
            os.remove(item)
        elif os.path.isdir(item):
            shutil.rmtree(item)
    
    prepare_input_files(self)
        
    #make empyt all_scores_df
    make_empty_all_scores_df(self)

    # Save varliables 
    save_main_variables(self)

def make_empty_all_scores_df(self):
    
    self.all_scores_df = pd.DataFrame(columns=['index', 'sequence', 'parent_index', \
                                              'interface_score', 'total_score', 'catalytic_score', 'efield_score', \
                                              'interface_potential', 'total_potential', 'catalytic_potential', 'efield_potential', \
                                              'relax_interface_score', 'relax_total_score', 'relax_catalytic_score', 'relax_efield_score', \
                                              'design_interface_score', 'design_total_score', 'design_catalytic_score', \
                                              'design_efield_score', 'generation', 'mutations', 'design_method', 'score_taken_from', \
                                              'blocked_ESMfold', 'blocked_RosettaRelax',  'blocked_ElectricFields', \
                                               'cat_resi', 'cat_resn'], dtype=object)
    
    save_all_scores_df(self)