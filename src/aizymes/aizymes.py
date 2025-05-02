"""
The AIzymes script defines the main AIzymes workflow, including setup, initialization, control, and plotting functions.
It manages the primary processes and configurations required to execute AIzymes functionalities.
"""

from main_running_003         import *
from main_startup_002         import *
from plotting_002             import *
from helper_002               import *

class AIzymes_MAIN:
    """
    Main class for managing AIzymes workflow, including setup, initialization, control, and plotting functions.

    Functions:
        __init__():     Initializes an instance of the AIzymes_MAIN class.
        setup():        Sets up the AIzymes project environment with specified parameters.
        initialize():   Initializes AIzymes with provided configurations.
        controller():   Controls the AIzymes project based on scoring and normalization parameters.
        plot():         Generates various plots based on AIzymes data.
    """

    def __init__(self, FOLDER_HOME, UNBLOCK_ALL=False, PRINT_VAR=False, PLOT_DATA=False, LOG='info'):
        """
        Initializes AIzymes with given parameters.

        Parameters:
            FOLDER_HOME (str):  Path to the main folder.
            UNBLOCK_ALL (bool): Flag to unblock all processes.
            PRINT_VAR (bool):   Flag to print variables.
            PLOT_DATA (bool):   Flag to plot data.
            LOG (str):          Logging level.
        """
        for key, value in locals().items():
            if key not in ['self']:
                setattr(self, key, value)

        if os.path.isdir(FOLDER_HOME):
            initialize_controller(self, FOLDER_HOME)
        else:
            print(f"Folder {FOLDER_HOME} missing. Please run setup().")

    def setup(self,

              # General Design Settings
              WT                  = None,
              LIGAND              = None,
              DESIGN              = None,
              PARENT_DES_MED      = ['RosettaDesign','ElectricFields'],
              DESIGN_METHODS      = [[0.7,'RosettaDesign','ElectricFields'],\
                                     [0.3,'ProteinMPNN','ESMfold','RosettaRelax','ElectricFields']],
              EXPLORE             = False,
              RESTRICT_RESIDUES   = None,
              FOLDER_PARENT       = 'parent',
              FOLDER_PAR_STRUC    = None,

              # General Scoring Settings
              SELECTED_SCORES     = ["total","catalytic","interface","efield"],
              WEIGHT_TOTAL        = 1.0,
              WEIGHT_CATALYTIC    = 1.0,
              WEIGHT_INTERFACE    = 1.0,
              WEIGHT_IDENTICAL    = 1.0,
              
              # General Job Settings
              MAX_JOBS            = 72, 
              MAX_GPUS            = 4,
              MEMORY              = 450,
              N_PARENT_JOBS       = 144, 
              MAX_DESIGNS         = 5000, 
              KBT_BOLTZMANN       = [1.0, 0.5, 0.02],
              SUBMIT_PREFIX       = None, 
              SYSTEM              = "AIzymes.config",
              LOG                 = 'info', 
              REMOVE_TMP_FILES    = False,
                            
              # RosettaDesign Settings
              CST_WEIGHT          = 1.0, 
              CST_NAME            = None,
              CST_DIST_CUTOFF     = False,
              
              # ProteinMPNN settings
              ProteinMPNN_T       = "0.1", 
              ProteinMPNN_BIAS    = 0.5,

              # LigandMPNN settings
              LigandMPNN_T        = "0.1", 
              
              # SolubleMPNN settings
              SolubleMPNN_T       = "0.1", 
              SolubleMPNN_BIAS    = 0.5,
              
              # FieldTools settings
              FIELD_TARGET        = None,
              FIELDS_EXCL_CAT     = True,
              WEIGHT_EFIELD       = 1.0,

              # BioDC settings
              TARGET_REDOX        = None,
              WEIGHT_REDOX        = 1.0,

              # Identical score settings
              IDENTICAL_DESIGN    = False,

              # Established Modules list
              # All Methods that redesign a sequence
              SYS_DESIGN_METHODS  = ["RosettaDesign","ProteinMPNN","LigandMPNN","SolubleMPNN", "Null"],
              # All Methods that create a structure
              SYS_STRUCT_METHODS  = ["RosettaDesign","MDMin","ESMfold","RosettaRelax",'AlphaFold3INF', "Null"], 
              # All Methods that require GPUs
              SYS_GPU_METHODS     = ["ESMfold",'AlphaFold3INF',"ProteinMPNN","LigandMPNN","SolubleMPNN"],
              
              ):
        """
        Sets up the AIzymes project environment with specified parameters.

        Parameters:
            FOLDER_HOME (str):          Path to the main folder.
            WT (str):                   Wild type information.
            LIGAND (str):               Ligand data.
            DESIGN (str):               Design specifications.
            MAX_JOBS (int):             Maximum number of jobs to run concurrently.
            MAX_GPUS (int):             Maximum number of GPUs to use.
            MEMORY (int):               Allocated memory in MB.
            N_PARENT_JOBS (int):        Number of parent jobs.
            MAX_DESIGNS (int):          Maximum number of designs.
            KBT_BOLTZMANN (list):       Boltzmann constant values.
            MATCH (str):                Match specifications.
            SUBMIT_PREFIX (str):        Submission prefix.
            SYSTEM (str):               System information.
            LOG (str):                  Logging level.
            REMOVE_TMP_FILES (bool):    Whether to remove temporary files.
            PARENT_DES_MED (list):      List of parent design methods.
            DESIGN_METHODS (list):      List of design methods with proportions.
            EXPLORE (bool):             Whether to explore parameter space.
            RESTRICT_RESIDUES (list):   2D list defining restricted residues in 1 letter code, e.g.: [[99,'DE'],[103,'H']]
            FOLDER_PARENT (str):        Parent folder name.
            FOLDER_PAR_STRUC (str):     Folder for parent structures.
            SCORING_METHODS (list):     List of scoring methods.
            SELECTED_SCORES (list):     List of selected scores.
            MDMin (bool):               Whether to use MD minimization.
            CST_WEIGHT (float):         Weight for constraints.
            CST_NAME (str):             Name for constraint.
            ProteinMPNN_BIAS (float):   Bias for ProteinMPNN.
            ProteinMPNN_T (str):        Temperature for ProteinMPNN.
            LigandMPNN_BIAS (float):    Bias for LigandMPNN.
            LigandMPNN_T (str):         Temperature for LigandMPNN.
            SolubleMPNN_BIAS (float):   Bias for SolubleMPNN.
            SolubleMPNN_T (str):        Temperature for SolubleMPNN.
            FIELD_TARGET (str):         Target for field calculations.
            FIELDS_EXCL_CAT (bool):     Exclude catalytic field.
            TARGET_REDOX (int):         Target redox value.
            FOLDER_MATCH (str):         Folder for match files.
            IDENTICAL_DESIGN (bool):    Whether designs are identical.
            CST_DIST_CUTOFF (bool):     Whether to apply a distance cutoff for constraints.
            SYS_DESIGN_METHODS (list):  List of design methods.
            SYS_STRUCT_METHODS (list):  List of structure creation methods.
            SYS_GPU_METHODS (list):     List of GPU-required methods.
        """
        for key, value in locals().items():
            if key not in ['self']:  
                setattr(self, key, value)
        
        aizymes_setup(self)

    def controller(self):
        """
        Main script to run AIzymes.
        """
        for key, value in locals().items():
            if key not in ['self']:
                setattr(self, key, value)
        start_controller(self)

    def submit_controller(self):
        """
        Creates submition script to start the controller.
        """
        submit_controller_parallel(self)
        
    def plot(self, 
             SCORES_V_INDEX=True, 
             STATISTICS=True,
             SCORES_V_GEN=True,
             SCORES_HIST=True,
             PRINT_VALS=True, 
             RESOURCE_LOG=False,
             NORM={},
             HIGHSCORE={},
             NEGBEST={},
             PLOT_TREE=True,
             landscape_plot=False,
             PLOT_SIZE=3,
             TREE_SCORE="final_score"):
        """
        Generates plots based on AIzymes data, including main, tree, and landscape plots.

        Parameters:
            SCORES_V_INDEX (bool):  Flag to generate plots scores vs index.
            STATISTICS (bool):      Flag to generate statistical plots.
            SCORES_V_GEN (bool):    Flag to generate plots scores vs generation.
            SCORES_HIST (bool):     Flag to generate histogram plots.
            PRINT_VALS (bool):      Flag to print values on the plots.
            RESOURCE_LOG (bool):    Flag to log resource usage.
            NORM (dict):            Normalization parameters.
            HIGHSCORE (dict):       High score values.
            NEGBEST (dict):         Negative best score values.
            PLOT_TREE (bool):       Flag to generate tree plots.
            landscape_plot (bool):  Flag to generate landscape plots.
            PLOT_SIZE (int):        Size parameter for plots.
            TREE_SCORE (str):       Score metric used for tree plots.
        """
        for key, value in locals().items():
            if key not in ['self']:
                setattr(self, key, value)
        make_plots(self)
        return

    def print_statistics(self, 
                        PRINT_NUMBERS=True,
                        PRINT_COLUMN=True,
                        PRINT_RUNNING=True,
                        PRINT_ROW=True,
                        index=0):
        """
        Prints a range of information from all_scores_df
        """
        for key, value in locals().items():
            if key not in ['self']:
                setattr(self, key, value)
        print_statistics_df(self)
        
    def best_structures(self,
                        SEQ_PER_ACTIVE_SITE = None, 
                        ACTIVE_SITE         = None,
                        N_HITS              = 50,
                        PLOT_SIZE           = 3):
        """
        Retrieves the best structures based on active site criteria and scoring.

        Parameters:
            SEQ_PER_ACTIVE_SITE (int):  Number of sequences per active site.
            ACTIVE_SITE (str):          Comma-seperated list of active site residues.
            N_HITS (int):               Number of top hits to retrieve.
            PLOT_SIZE (int):            Plot size for visual representation.
        """
        for key, value in locals().items():
            if key not in ['self']:
                setattr(self, key, value)
        get_best_structures(self)

    def tar_designs(self):
        
        import tarfile, os, shutil

        with tarfile.open(f"{self.FOLDER_HOME}/designs.tar", "w") as tar:
            tar.add(f"{self.FOLDER_HOME}/designs", arcname="design")
    
    def untar_designs(self):
        
        import tarfile, os, shutil, sys

        tar_path     = os.path.join(self.FOLDER_HOME, "designs.tar")
        extract_root = self.FOLDER_HOME
    
        if not os.path.isfile(tar_path):
            print(f"Error! Tarfile does not exist: {tar_path}")
            sys.exit(1)
    
        # Open for reading and extract all
        with tarfile.open(tar_path, mode="r:*") as tar:
            tar.extractall(path=extract_root)
