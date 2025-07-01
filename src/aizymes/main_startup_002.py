import os
import sys
import time
import json
import logging
import pandas as pd
import json
import shutil
import datetime
import glob

from helper_002               import *
from setup_system_001         import *
from main_scripts_001         import *

def submit_controller_parallel(self):
    
    jobs = subprocess.run(["squeue", "--me"], capture_output=True, text=True, check=True)
    if self.SUBMIT_PREFIX in jobs.stdout: 
        print(f"ERROR! Job with prefix {self.SUBMIT_PREFIX} is already running. Refusing to start another job in parallel")
        return

    cmd = f'''import sys, os
sys.path.append(os.path.join(os.getcwd(), '../../src'))
from aizymes import *
AIzymes = AIzymes_MAIN(FOLDER_HOME    = '{os.path.basename(self.FOLDER_HOME)}', 
                       LOG            = '{self.LOG}',
                       PRINT_VAR      = False,
                       UNBLOCK_ALL    = True)
AIzymes.controller()
'''
    with open(f"{self.FOLDER_HOME}/start_controller_parallel.py", "w") as f:
        f.write(cmd)

    cmd = self.SUBMIT_HEAD
    
    cmd += f"""#SBATCH --job-name={self.SUBMIT_PREFIX}
#SBATCH --output={self.FOLDER_HOME}/controller.log
#SBATCH --error={self.FOLDER_HOME}/controller.log

export CLUSTER="RAVEN"

set -e  # Exit script on any error

cd {self.FOLDER_HOME}/..

echo "Current Working Directory:" 
pwd 
echo "Job Started:" 
date 

python {self.FOLDER_HOME}/start_controller_parallel.py

""" 
    
    with open(f"{self.FOLDER_HOME}/submit_controller_parallel.sh", "w") as f:
        f.write(cmd)
        
    logging.info(f"Starting parallel controller.")

    ### Start job
    output = subprocess.check_output(
    (f'sbatch {self.FOLDER_HOME}/submit_controller_parallel.sh'),
    shell=True, text=True
    )

def initialize_controller(self, FOLDER_HOME):

    #FOLDER_HOME is the FOLDER_HOME given by user, self.FOLDER_HOME is the one storred in main_variable
    if not os.path.isabs(FOLDER_HOME): FOLDER_HOME = f'{os.getcwd()}/{FOLDER_HOME}'  

    # Checks if self.VARIABLES_JSON exists
    self.VARIABLES_JSON = f'{self.FOLDER_HOME}/variables.json'
    if not os.path.isfile(self.VARIABLES_JSON):
        
        print(f"Error, {self.VARIABLES_JSON} missing. This AI.zymes run is likely corrupted. Remove {FOLDER_HOME} and re-run AIZYMES_setup")
        if input(f'''Do you want to delete {self.FOLDER_HOME} [y/n]
''') == 'y': 
            shutil.rmtree(self.FOLDER_HOME)
            print(f'''{self.FOLDER_HOME} deleted''')
            sys.exit()

    # Load main variables
    load_main_variables(self, FOLDER_HOME)

    # Print out main variables
    if self.PRINT_VAR:
        if os.path.isfile(self.VARIABLES_JSON):
            with open(self.VARIABLES_JSON, 'r') as f: 
                variables_dict = json.load(f)
            for k, v in variables_dict.items():
                print(k.ljust(16), ':', v)
                
    # Starts the logger
    initialize_logging(self)
      
    # Stops run if FOLDER_HOME does not match
    if FOLDER_HOME != self.FOLDER_HOME:
        print(f"Error, wrong FOLDER_HOME! Given {FOLDER_HOME}, required: {self.FOLDER_HOME}")
        sys.exit()
        
    if self.PLOT_DATA:
        plot_scores()
        
    # Read in current databases of AIzymes
    self.all_scores_df = pd.read_csv(self.ALL_SCORES_CSV)
    self.resource_log_df = pd.read_csv(self.RESOURCE_LOG_CSV)

    self.processes = []
    if self.MAX_GPUS > 0:
        self.gpus = {gpu_id: None for gpu_id in range(self.MAX_GPUS)}

    if self.UNBLOCK_ALL: 
        logging.info(f'Unblocking all')
        
        for idx, row in self.all_scores_df.iterrows():

            if self.all_scores_df.at[idx, "blocked"] == "failed": continue
            if self.all_scores_df.at[idx, "blocked"] == "unblocked": continue

            # Remove CRASH.log. Else, the variant will be marked as failed later during update scores
            filename = os.path.join(self.FOLDER_DESIGN, str(index), 'ROSETTA_CRASH.log')
            if os.path.exists(filename):
                os.remove(filename)
    
            # Update "next_steps" --> make sure to remove NaN if "next_steps" is NaN
            if pd.isna(self.all_scores_df.at[idx, "next_steps"]):
                self.all_scores_df.at[idx, "next_steps"] = self.all_scores_df.at[idx, "blocked"]
            else:
                self.all_scores_df.at[idx, "next_steps"] = f"{self.all_scores_df.at[idx, 'blocked']},{self.all_scores_df.at[idx, 'next_steps']}"
            self.all_scores_df.at[idx, 'step_output_variant'] = self.all_scores_df.at[idx, 'step_input_variant']
            self.all_scores_df.at[idx, 'step_input_variant'] = self.all_scores_df.at[idx, 'previous_input_variant_for_reset']

        self.all_scores_df["blocked"] = "unblocked"
        save_all_scores_df(self)
        
    # Sleep a bit to make me feel secure. More sleep = less anxiety :)
    time.sleep(0.1)

def prepare_input_files(self):
    
    os.makedirs(self.FOLDER_HOME, exist_ok=True)    
    os.makedirs(self.FOLDER_PARENT, exist_ok=True)
    os.makedirs(self.FOLDER_PLOT, exist_ok=True)
    os.makedirs(self.FOLDER_DESIGN, exist_ok=True)

    # Creates general input scripts used by various programs 
    make_main_scripts(self)
        
    # Copy parent structure from Folder Input if FOLDER_PARENT is the same as FOLDER_PAR_STRUC  
    if self.FOLDER_PARENT == self.FOLDER_PAR_STRUC:
        
        if not os.path.isfile(f'{self.FOLDER_INPUT}/{self.WT}.pdb'):
            logging.error(f"Input structure {self.FOLDER_INPUT}/{self.WT}.pdb is missing!")
            sys.exit()  
        self.N_INPUT_STRUCTURES = 1
        shutil.copy(f'{self.FOLDER_INPUT}/{self.WT}.pdb', f'{self.FOLDER_PARENT}/{self.WT}.pdb')

    else:

        pdb_files = glob.glob(f'{self.FOLDER_PAR_STRUC}/*.pdb')
        if not pdb_files:
            logging.error(f"No pdb files found in {self.FOLDER_PAR_STRUC}.")
            sys.exit()  
        self.N_INPUT_STRUCTURES = len(pdb_files)
        for pdb_file in pdb_files:
            shutil.copy(pdb_file, self.FOLDER_PARENT)

    self.N_PARENTS = self.N_PARENT_JOBS * self.N_INPUT_STRUCTURES
    if self.N_PARENTS < self.MAX_JOBS:
        logging.error(f"Attention! N_PARENT_JOBS * self.N_INPUT_STRUCTURES must be > MAX_JOBS. N_PARENTS: {self.N_PARENTS}, MAX_JOBS: {self.MAX_JOBS}.")
        
    # Copy .cst file
    if self.CST_NAME is not None:
        if not os.path.isfile(f'{self.FOLDER_INPUT}/{self.CST_NAME}.cst'):
            logging.error(f"Constraint file {self.FOLDER_INPUT}/{self.CST_NAME}.cst is missing!")
            sys.exit()       
        shutil.copy(f'{self.FOLDER_INPUT}/{self.CST_NAME}.cst', f'{self.FOLDER_PARENT}/{self.CST_NAME}.cst')
        
    # Save input sequence with X as wildcard 
    seq = sequence_from_pdb(f"{self.FOLDER_INPUT}/{self.WT}")
    with open(f'{self.FOLDER_PARENT}/{self.WT}.seq', "w") as f:
        f.write(seq)    
    design_positions = [int(x) for x in self.DESIGN.split(',')]
    
    # Replace seq with X at design positions. Note: Subtract 1 from each position to convert to Python's 0-based indexing
    seq = ''.join('X' if (i+1) in design_positions else amino_acid for i, amino_acid in enumerate(seq))
    with open(f'{self.FOLDER_HOME}/{self.WT}_with_X_as_wildecard.seq', 'w') as f:
        f.writelines(seq)    
    
    # Get the Constraint Residues from enzdes constraints file
    if self.CST_NAME != None:
        with open(f'{self.FOLDER_INPUT}/{self.CST_NAME}.cst', 'r') as f:
            cst = f.readlines()    
        cst = [i.split()[-1] for i in cst if "TEMPLATE::   ATOM_MAP: 2 res" in i]
        cst = ";".join(cst)
        with open(f'{self.FOLDER_HOME}/cst.dat', 'w') as f:
            f.write(cst)    

    # Save FIELD_TARGET
    seq = sequence_from_pdb(f"{self.FOLDER_INPUT}/{self.WT}")
    with open(f'{self.FOLDER_PARENT}/field_target.dat', "w") as f:
        f.write(f'{self.FIELD_TARGET}\n')    
                    
def initialize_variables(self):

    # Complete directories
    if not os.path.isabs(self.FOLDER_HOME): # User can either give absolutel path or only folder name!
        self.FOLDER_HOME  = f'{os.getcwd()}/{self.FOLDER_HOME}'
    if not os.path.isabs(self.FOLDER_PARENT): # User can either give absolutel path or only folder name!
        self.FOLDER_PARENT    = f'{self.FOLDER_HOME}/{self.FOLDER_PARENT}'
    if self.FOLDER_PAR_STRUC is None:
        self.FOLDER_PAR_STRUC = self.FOLDER_PARENT
    elif not os.path.isabs(self.FOLDER_PAR_STRUC): # User can either give absolutel path or only folder name!
        self.FOLDER_PAR_STRUC = f'{self.FOLDER_HOME}/{self.FOLDER_PAR_STRUC}'
        
    self.FOLDER_INPUT     = f'{os.getcwd()}/Input'
    self.USERNAME         = os.environ.get("USER", os.environ.get("LOGNAME", "unknown_user"))
    self.LOG_FILE         = f'{self.FOLDER_HOME}/logfile.log'
    self.ALL_SCORES_CSV   = f'{self.FOLDER_HOME}/all_scores.csv'
    self.RESOURCE_LOG_CSV = f'{self.FOLDER_HOME}/resource_log.csv'
    self.VARIABLES_JSON   = f'{self.FOLDER_HOME}/variables.json'
    self.FOLDER_PLOT      = f'{self.FOLDER_HOME}/plots' 
    self.FOLDER_DESIGN    = f'{self.FOLDER_HOME}/designs' 

    # Set weights to 0 if not used
    if "efield" not in self.SELECTED_SCORES:     
        self.WEIGHT_EFIELD = 0
    if "total" not in self.SELECTED_SCORES:      
        self.WEIGHT_TOTAL = 0 
    if "redox" not in self.SELECTED_SCORES:
        self.WEIGHT_REDOX = 0
    if "interface" not in self.SELECTED_SCORES: 
        self.WEIGHT_INTERFACE = 0
    if "catalytic" not in self.SELECTED_SCORES:  
        self.WEIGHT_CATALYTIC = 0
    if "identical" not in self.SELECTED_SCORES:
        self.WEIGHT_IDENTICAL = 0

    # Define system-specific settings
    set_system(self)    
                            
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

def check_settings(self):

    if not os.path.isdir(self.FOLDER_INPUT):
        print(f"ERROR! Input folder missing! Should be {self.FOLDER_INPUT}")
        sys.exit()
    
    if any(method in self.SYS_GPU_METHODS for method in self.DESIGN_METHODS) and self.MAX_GPUS == 0:
        logging.error("Please request GPUs if you are using GPU_METHODS.")
        sys.exit()

    if self.FOLDER_HOME == None:
        logging.error("Please define home folder with [FOLDER_HOME].")
        sys.exit()
        
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
                
    if self.FIELD_TARGET == None and "ElectricFields" in self.DESIGN_METHODS:
        logging.error(f"FIELD_TARGET must be defined for ElectricField calculation! FIELD_TARGET: {self.FIELD_TARGET}.")
        sys.exit()
        
    if len(self.SUBMIT_PREFIX) > 8:
        logging.error(f"SUBMIT_PREFIX cannot be longer than 8 characters. SUBMIT_PREFIX: {self.SUBMIT_PREFIX}.")
        sys.exit()

    if self.LIGAND is None and CST_NAME is not None:    
        logging.error(f"CST_NAME given, but no LIGAND defined!")
        sys.exit()   

    if self.LIGAND is not None:
        for extension in ['params','frcmod','prepi']:
            if self.LIGAND == "HEM": continue     
            if not os.path.isfile(f'{self.FOLDER_INPUT}/{self.LIGAND}.{extension}'):
                logging.error(f'File missing: {self.FOLDER_INPUT}/{self.LIGAND}.{extension}')
                sys.exit()

    for design_method in self.DESIGN_METHODS:
        if "RosettaDesign" not in design_method:
            if "RosettaRelax" not in design_method: 
                logging.error(f"Each Design Method must contain either RosettaDesign or RosettaRelax. Design Method {design_method}")
                sys.exit()  
    if "RosettaDesign" not in self.PARENT_DES_MED:
        if "RosettaRelax" not in self.PARENT_DES_MED: 
            logging.error(f"Each Design Method must contain either RosettaDesign or RosettaRelax. PARENT_DES_MED {self.PARENT_DES_MED}")
            sys.exit()  

    for design_method in self.DESIGN_METHODS:
        methods = [method for method in design_method if method not in self.SYS_STRUCT_METHODS]
        for method in set(methods):
            if methods.count(method) > 1:
                logging.error(f"Non-structural methods must not be duplicated! Duplicate method: {method}")
                sys.exit()              
    methods = [method for method in self.PARENT_DES_MED if method not in self.SYS_STRUCT_METHODS]
    for method in set(methods):
        if methods.count(method) > 1:
            logging.error(f"Non-structural methods must not be duplicated! Duplicate method: {method}")
            sys.exit()  

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
        
    # Check all settings
    check_settings(self)
            
    with open(self.LOG_FILE, 'w'): pass  #resets logfile
    
    logging.info(f"Running AI.zymes setup.")
    logging.info(f"Content of {self.FOLDER_HOME} deleted.")
    logging.info(f"Happy AI.zymeing! :)")
   
    for item in os.listdir(self.FOLDER_HOME):
        if item == self.FOLDER_PARENT: continue
        if item == self.FOLDER_INPUT: continue           
        item = f'{self.FOLDER_HOME}/{item}'
        if os.path.isfile(item): 
            os.remove(item)
        elif os.path.isdir(item):
            shutil.rmtree(item)
    
    prepare_input_files(self)
        
    # Make empyt all_scores_df
    make_empty_all_scores_df(self)
   
    # Make empyt resource_log
    make_empty_resource_log_df(self)
        
    # Add parent designs to all_scores_df
    schedule_parent_design(self)
    
    # Save varliables 
    save_main_variables(self)

def make_empty_all_scores_df(self):
    '''
    Makes the starting all_scores_df dataframe that collects all information about an AI.zymes run
    '''
    columns = ['index', 'parent_index', 'generation', 'total_mutations', 'parent_mutations', 'score_taken_from',
               'design_method', 'blocked', 'next_steps', 
               'final_variant', 'input_variant', 'previous_input_variant_for_reset', 'step_input_variant', 'step_output_variant',
               'sequence']
    if self.CST_NAME is not None:
        columns.extend(['cat_resi', 'cat_resn'])
    for score in self.SELECTED_SCORES:
        columns.append(f'{score}_potential')
        columns.append(f'{score}_score')
        columns.append(f'design_{score}_score')
        columns.append(f'relax_{score}_score')
    self.all_scores_df = pd.DataFrame(columns=columns, dtype=object)
    save_all_scores_df(self)
    
def make_empty_resource_log_df(self):
    '''
    Makes the starting resource_log dataframe to track jobs running over time
    '''

    self.resource_log_df = pd.DataFrame({
        'time':               int(datetime.datetime.now().timestamp()),
        'cpus_used':          np.nan,
        'gpus_used':          np.nan,
        'total_designs':      np.nan,
        'finished_designs':   np.nan,
        'unfinished_designs': np.nan,
        'failed_designs':     np.nan,
        'kbt_boltzmann':      np.nan,
    }, index = [0] , dtype=object)  
    
    save_resource_log_df(self)
    
def schedule_parent_design(self):
    '''
    Adds all parent design runs to the all_score_df dataframe
    '''

    # Do not do parent design if designs were made already (when restarting!)
    if len(self.all_scores_df) != 0: return
        
    # Define intial and final structures
    parent_structures = [i[:-4] for i in os.listdir(self.FOLDER_PARENT) if i[-4:] == ".pdb"]
    final_structure_method = [i for i in self.PARENT_DES_MED if i in self.SYS_STRUCT_METHODS][-1]
    design_method = [i for i in self.PARENT_DES_MED if i in self.SYS_DESIGN_METHODS][0]

    # Make all .seq files
    for parent_structure in parent_structures:
        seq = sequence_from_pdb(f"{self.FOLDER_PARENT}/{parent_structure}")
        with open(f'{self.FOLDER_PARENT}/{parent_structure}.seq', "w") as f:
            f.write(seq) 
            
    # Add all parent indices to all_scores_df
    for parent_structure in parent_structures:
        for n_parent_job in range(self.N_PARENT_JOBS):  

            new_index = create_new_index(self, 
                                         parent_index        = "Parent", 
                                         luca                = f'{self.FOLDER_PARENT}/{parent_structure}',
                                         input_variant       = f'{self.FOLDER_PARENT}/{parent_structure}',
                                         final_method        = final_structure_method,
                                         next_steps          = ",".join(self.PARENT_DES_MED), 
                                         design_method       = design_method)    