# -------------------------------------------------------------------------------------------------------------------------
# Import AIzymes modules --------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
    
from main_running_001         import *
from main_startup_001         import *
from plotting_001             import *

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

class AIzymes_MAIN():

    def __init__(self):
        
        return
         
    def setup(self, FOLDER_HOME, FOLDER_PARENT, CST_NAME, WT, LIGAND, DESIGN,
      MAX_JOBS          = 100,
      N_PARENT_JOBS     = 3,
      MAX_DESIGNS       = 10000,
      KBT_BOLTZMANN     = [0.5, 0.0003],
      CST_WEIGHT        = 1.0,
      ProteinMPNN_PROB  = 0.0,
      ProteinMPNN_BIAS  = 0.0,
      LMPNN_PROB        = 0.0,
      FOLDER_MATCH      = None,
      ProteinMPNN_T     = "0.1",
      LMPNN_T           = "0.1",
      LMPNN_BIAS        = 0.0,
      SUBMIT_PREFIX     = None,
      SYSTEM            = None,
      MATCH             = None,  
      ROSETTA_PATH      = "/home/bunzelh/rosetta_src_2021.16.61629_bundle/main/source/",
      EXPLORE           = False,
      FIELD_TOOLS       = 'src/FieldTools.py',
      LOG               = 'debug',
      PARENT_DES_MED    = 'RosettaDesign'
      ):
                   
        # Automatically assign all parameters to instance variables
        for key, value in locals().items():
            if key not in ['self']:  
                setattr(self, key, value)
        
        # Run aizymes_setup
        aizymes_setup(self)
        
        print('''AIzymes initiated.
''')
        
  
    def initialize(self, FOLDER_HOME,
                   UNBLOCK_ALL     = False, 
                   PRINT_VAR       = True,
                   PLOT_DATA       = False,
                   LOG             = 'debug'):
        
        # Automatically assign all parameters to instance variables
        for key, value in locals().items():
            if key not in ['self']:  
                setattr(self, key, value)
                              
        initialize_controller(self, FOLDER_HOME)
        
    def controller(self,
                   HIGHSCORE       = 0.70,
                   NORM            = {'interface_score': [10, 35],
                                      'total_score': [200, 500], 
                                      'catalytic_score': [-40, 0], 
                                      'efield_score': [10, 220]}):

        # Automatically assign all parameters to instance variables
        for key, value in locals().items():
            if key not in ['self']:
                setattr(self, key, value)
                
        start_controller(self)
        
    def plot(self, main_plots=True, tree_plot=True, landscape_plot=True,
        NORM = {'interface_score': [10, 35],
                                   'total_score': [200, 500], 
                                    'catalytic_score': [-40, 0], 
                                    'efield_score': [10, 220]},
        HIGHSCORE_NEGBEST = {
    'HIGHSCORE.combined_score': 1.0,
    'NEGBEST.combined_score': 0.0,
    'HIGHSCORE.total_score': 10.0,
    'NEGBEST.total_score': -150,
    'HIGHSCORE.interface_score': -30.0,
    'NEGBEST.interface_score': -15.0,
    'HIGHSCORE.efield_score': 250.0,
    'NEGBEST.efield_score': 0.0,
}):
  
        # Automatically assign all parameters to instance variables
        for key, value in locals().items():
            if key not in ['self']:
                setattr(self, key, value)
                
        if main_plots:
            plot_scores(self)
        if tree_plot:
            plot_tree(self)
        if landscape_plot:
            plot_landscape(self)
        
        return