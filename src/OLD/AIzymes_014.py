"""
AIzymes Project Main Workflow

This script defines the main AIzymes workflow, including setup, initialization, control, and plotting functions.
It manages the primary processes and configurations required to execute AIzymes functionalities.

Classes:
    AIzymes_MAIN: Manages the main workflow for AIzymes, including setup, initialization, and various control functions.

Functions:
    __init__(): Initializes an instance of the AIzymes_MAIN class.
    setup(): Sets up the AIzymes project environment with specified parameters.
    initialize(): Initializes AIzymes with provided configurations.
    controller(): Controls the AIzymes project based on scoring and normalization parameters.
    plot(): Generates various plots based on AIzymes data.
"""

# -------------------------------------------------------------------------------------------------------------------------
# Import AIzymes modules --------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
    
from main_running_002         import *
from main_startup_001         import *
from plotting_001             import *
from helper_001               import *

# Imports used elsewhere --------------------------------------------------------------------------------------------------
#from main_design_001          import *
#from design_match_001         import *
#from design_ProteinMPNN_001   import *
#from design_LigandMPNN_001    import *
#from design_RosettaDesign_001 import *
#from design_ESMfold_001       import *
#from design_RosettaRelax_001  import *
#from helper_001               import *
#from scoring_efields_001      import *
#from setup_system_001         import *
# -------------------------------------------------------------------------------------------------------------------------

class AIzymes_MAIN:
    """
    Main class for managing AIzymes workflow, including setup, initialization, control, and plotting functions.
    """

    def __init__(self):
        """
        Initializes an instance of the AIzymes_MAIN class.
        """
        return

    def setup(self, FOLDER_HOME, FOLDER_PARENT, CST_NAME, WT, LIGAND, DESIGN,

              # General Job Settings
              MAX_JOBS         = 100, 
              MAX_GPUS         = 0,
              MEMORY           = 128,
              N_PARENT_JOBS    = 3, 
              MAX_DESIGNS      = 10000, 
              KBT_BOLTZMANN    = [0.5, 0.0003],
              MATCH            = None, 
              
              # System Settings
              SUBMIT_PREFIX    = None, 
              SYSTEM           = None,
              RUN_PARALLEL     = False, 
              LOG              = 'debug',   
              
              # General Design Settings
              PARENT_DES_MED   = 'RosettaDesign',
              DESIGN_METHODS   = [],
              EXPLORE          = False,
              
              # General Scoring Settings
              SELECTED_SCORES  = ["total","catalytic","interface","efield"],
              MDMin            = False, 
              
              # RosettaDesign Settings
              CST_WEIGHT       = 1.0, 

              # ProteinMPNN Settings
              ProteinMPNN_PROB = 0.0, 
              ProteinMPNN_BIAS = 0.5, 
              ProteinMPNN_T    = "0.1", 

              # LigandMPNN Settings
              LigandMPNN_PROB  = 0.0,  
              LigandMPNN_BIAS  = 0.5, 
              LigandMPNN_T     = "0.1", 

              # FieldTools Settings
              FIELD_TARGET     = ":5TS@C9 :5TS@H04",
              FIELDS_EXCL_CAT  = False,

              # RosettaMatch Settings
              FOLDER_MATCH     = None,
              
              ):
        """
        Sets up the AIzymes project environment with specified parameters.

        Args:
            FOLDER_HOME (str):          Path to the main folder.
            FOLDER_PARENT (str):        Path to the parent folder.
            CST_NAME (str):             Constraint name.
            WT (str):                   Wild type information.
            LIGAND (str):               Ligand data.
            DESIGN (str):               Design specifications.
            MAX_JOBS (int):             Maximum number of jobs to run concurrently.
            N_PARENT_JOBS (int):        Number of parent jobs.
            MAX_DESIGNS (int):          Maximum number of designs.
            KBT_BOLTZMANN (list):       Boltzmann constant values.
            CST_WEIGHT (float):         Constraint weight.
            
            ProteinMPNN_PROB (float):   Probability parameter for ProteinMPNN.
            ProteinMPNN_BIAS (float):   Bias parameter for ProteinMPNN.
            ProteinMPNN_T (str):        Temperature for ProteinMPNN.
            
            LigandMPNN_PROB (float):    Probability parameter for LMPNN.
            LigandMPNN_BIAS (float):    Bias parameter for ProteinMPNN.
            LigandMPNN_T (str):         Temperature for LMPNN.
            
            FOLDER_MATCH (str):         Path to match folder.
            SUBMIT_PREFIX (str):        Submission prefix.
            SYSTEM (str):               System information.
            MATCH (str):                Match specifications.
            EXPLORE (bool):             Whether to explore parameter space.
            FIELD_TARGET (str):         Target atoms at which to calculate the electric field.
            LOG (str):                  Logging level.
            PARENT_DES_MED (str):       Parent design method.
            MDMin (bool):               Use MD minimization.
            RUN_PARALLEL (bool):        If true, use a single job using MAX_JOBS CPUs that will constantly run for design.
            RUN_INTERACTIVE (bool):     If true, do not submit jobs but run in background.
            FIELDS_EXCL_CAT (bool):     If true, subtract the field of catalytic residue from the electric field score.

            
        """
        for key, value in locals().items():
            if key not in ['self']:  
                setattr(self, key, value)
        
        aizymes_setup(self)
        
        print("AIzymes initiated.")

    def initialize(self, FOLDER_HOME, UNBLOCK_ALL=False, PRINT_VAR=True, PLOT_DATA=False, LOG='debug'):
        """
        Initializes AIzymes with given parameters.

        Args:
            FOLDER_HOME (str): Path to the main folder.
            UNBLOCK_ALL (bool): Flag to unblock all processes.
            PRINT_VAR (bool): Flag to print variables.
            PLOT_DATA (bool): Flag to plot data.
            LOG (str): Logging level.
        """
        for key, value in locals().items():
            if key not in ['self']:  
                setattr(self, key, value)
                              
        initialize_controller(self, FOLDER_HOME)

    def controller(self):
        """
        Controls the AIzymes project based on scoring and normalization parameters.
        """
        
        for key, value in locals().items():
            if key not in ['self']:
                setattr(self, key, value)
                
        start_controller(self)

        
    def submit_controller(self):
        """
        Controls the AIzymes project based on scoring and normalization parameters.
        """

        submit_controller_parallel(self)
        
    def plot(self, main_plots=True, tree_plot=True, landscape_plot=True, print_vals=True, NORM=None, HIGHSCORE_NEGBEST=None):
        """
        Generates plots based on AIzymes data, including main, tree, and landscape plots.

        Args:
            main_plots (bool): Flag to generate main plots.
            tree_plot (bool): Flag to generate tree plot.
            landscape_plot (bool): Flag to generate landscape plot.
            print_vals (bool): Flag to print values on plots.
            NORM (dict): Normalization values for different scores.
            HIGHSCORE_NEGBEST (dict): High score and negative best score for different metrics.
        """
        if NORM is None:
            NORM = {
                'interface_score': [10, 35],
                'total_score': [200, 500], 
                'catalytic_score': [-40, 0], 
                'efield_score': [10, 220],
                'identical_score': [0, 1]
            }
        if HIGHSCORE_NEGBEST is None:
            HIGHSCORE_NEGBEST = {
                'HIGHSCORE_combined_score': 0.814,
                'NEGBEST_combined_score': 0.503,
                'HIGHSCORE_total_score': 0.954,
                'NEGBEST_total_score': 0.209,
                'HIGHSCORE_interface_score': 0.994,
                'NEGBEST_interface_score': 0.935,
                'HIGHSCORE_catalytic_score': 0.994,
                'NEGBEST_catalytic_score': 0.935,
                'HIGHSCORE_efield_score': 0.970,
                'NEGBEST_efield_score': 0.807,
               'HIGHSCORE_identical_score': 1,
                'NEGBEST_identical_score': 0.935
            }

        for key, value in locals().items():
            if key not in ['self']:
                setattr(self, key, value)
                
        if main_plots:
            plot_scores(self, print_vals=print_vals)
            kbt_cst_weights_plotting_function(self)
            
        if tree_plot:
            plot_tree(self)
           
        if landscape_plot:
            landscape_plotting_function(self)

        return

    def best_structures(self, save_structures = False, include_catalytic_score = False, seq_per_active_site = 100, DESIGN = None, WT = None):
        get_best_structures(self, 
                            save_structures = save_structures, 
                            include_catalytic_score = include_catalytic_score,
                            seq_per_active_site = seq_per_active_site,
                            DESIGN = DESIGN,
                            WT = WT)

