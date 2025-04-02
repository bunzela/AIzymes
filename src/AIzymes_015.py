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
# CHANGE COMPARED TO AIzymes_014: -----------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# Include GPUs
# Adjusted Job Schedule, indivdual Jobs and GPU queue
# -------------------------------------------------------------------------------------------------------------------------
# Import AIzymes modules --------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
    
from main_running_003         import *
from main_startup_002         import *
from plotting_002             import *
from helper_002               import *

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

    def setup(self, FOLDER_HOME, FOLDER_PARENT, WT, LIGAND, DESIGN,

              # General Job Settings
              MAX_JOBS            = 100, 
              MAX_GPUS            = 0,
              MEMORY              = 128,
              N_PARENT_JOBS       = 3, 
              MAX_DESIGNS         = 10000, 
              KBT_BOLTZMANN       = [0.5, 0.0003],
              MATCH               = None, 
              
              # System Settings
              SUBMIT_PREFIX       = None, 
              SYSTEM              = None,
              RUN_PARALLEL        = False, 
              LOG                 = 'info',   
              
              # General Design Settings
              PARENT_DES_MED      = ['RosettaDesign','ElectricFields'],
              DESIGN_METHODS      = [[0.7,'RosettaDesign','ElectricFields'],\
                                     [0.3,'ProteinMPNN','ESMfold','RosettaRelax','ElectricFields']],
              EXPLORE             = False,
              RESTRICT_RESIDUES   = None, #2D list defining restricted residues in 1 letter code, e.g.: [[99,'DE'],[103,'H']]
              
              # General Scoring Settings
              SCORING_METHODS     = ['MDMin','ESMfold','RosettaRelax','ElectricFields'], 
              SELECTED_SCORES     = ["total","catalytic","interface","efield"],
              MDMin               = False, 
              
              # RosettaDesign Settings
              CST_WEIGHT          = 1.0, 
              CST_NAME            = None,
              
              # ProteinMPNN Settings
              ProteinMPNN_BIAS    = 0.5, 
              ProteinMPNN_T       = "0.1", 

              # LigandMPNN Settings
              LigandMPNN_BIAS     = 0.5, 
              LigandMPNN_T        = "0.1", 
              
              # SolubleMPNN Settings
              SolubleMPNN_BIAS    = 0.5, 
              SolubleMPNN_T       = "0.1", 

              # FieldTools Settings
              FIELD_TARGET        = None,
              FIELDS_EXCL_CAT     = True,

              # BioDC Settings
              TARGET_REDOX        = 10,
              
              # RosettaMatch Settings
              FOLDER_MATCH        = None,

              # Established Modules list
              # All Methods that redesign a sequence
              SYS_DESIGN_METHODS  = ["RosettaDesign","ProteinMPNN","LigandMPNN","SolubleMPNN"],
              # All Methods that create a structure
              SYS_STRUCT_METHODS  = ["RosettaDesign","MDMin","ESMfold","RosettaRelax",'AlphaFold3INF'], 
              # All Methods that require GPUs
              SYS_GPU_METHODS     = ["ESMfold",'AlphaFold3INF',"ProteinMPNN","LigandMPNN","SolubleMPNN"],
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

    def initialize(self, FOLDER_HOME, UNBLOCK_ALL=False, PRINT_VAR=False, PLOT_DATA=False, LOG='info'):
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
        
    def plot(self, 
             SCORES_V_INDEX=False, 
             STATISTICS=False,
             SCORES_V_GEN=False,
             SCORES_HIST=False,
             PRINT_VALS=False, 
             RESOURCE_LOG=False,
             NORM={},
             HIGHSCORE={},
             NEGBEST={},
             PLOT_TREE=False,
             landscape_plot=False,
             PLOT_SIZE=3,
             TREE_SCORE="combined_score"):
        
        """
        Generates plots based on AIzymes data, including main, tree, and landscape plots.

        Args:
            SCORES_V_INDEX (bool): Flag to generate plots scores vs index.
            SCORES_V_GEN (bool): Flag to generate plots scores vs generation.
            tree_plot (bool): Flag to generate tree plot.
            landscape_plot (bool): Flag to generate landscape plot.
            PRINT_VALS (bool): Flag to print values on plots.
            NORM (dict): Normalization values for different scores.
            HIGHSCORE_NEGBEST (dict): High score and negative best score for different metrics.
        """

        for key, value in locals().items():
            if key not in ['self']:
                setattr(self, key, value)

        make_plots(self)
        
        return

    def best_structures(self,
                        SEQ_PER_ACTIVE_SITE = None, 
                        ACTIVE_SITE = None,
                        N_HITS = 100):

        for key, value in locals().items():
            if key not in ['self']:
                setattr(self, key, value)
                
        get_best_structures(self)

