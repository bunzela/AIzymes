
"""
Main Design Module

Coordinates various design steps, managing the workflow of Rosetta, ProteinMPNN, and other modules
within the AIzymes project.

Functions:
    - get_ram: Determines RAM allocation for design steps.
    - run_design: Runs the selected design steps based on configuration.

Modules Required:
    - helper_001, design_match_001, design_ProteinMPNN_001, design_LigandMPNN_001,
      design_RosettaDesign_001, design_ESMfold_001, design_RosettaRelax_001
"""
import logging
import sys

from helper_001               import *

from design_match_001         import *
from design_ProteinMPNN_001   import *
from design_LigandMPNN_001    import *
from design_RosettaDesign_001 import *
from design_ESMfold_001       import *
from design_RosettaRelax_001  import *
from scoring_efields_001      import *

def get_ram(design_steps):
    
    ram = 0
    
    for design_step in design_steps:
        
        if design_step == "ProteinMPNN":
            new_ram = 20
        elif design_step == "RosettaDesign":
            new_ram = 10
        elif design_step == "RosettaRelax": 
            new_ram = 10
        elif design_step == "ESMfold":
            new_ram = 40
        elif design_step == "ElectricFields":
            new_ram = 10
        else:
            logging.error(f"RAM for design_step {design_step} is not defined!")
            sys.exit()
            
        if new_ram > ram: ram = new_ram
            
    return ram

def run_design(self, 
               index,
               design_steps,
               bash = False
              ):
    
    # Expecting list. To make sure individual commmand would also be accepted, convert string to list if string is given
    if not isinstance(design_steps, list):
        variable = [design_steps]
    
    ram = get_ram(design_steps)
    
    cmd = ""
     
    for design_step in design_steps:
                 
        if design_step == "ProteinMPNN":
            
            cmd = prepare_ProteinMPNN(self, index, cmd)
            logging.debug(f"Run ProteinMPNN for index {index}.")
            
        elif design_step == "RosettaDesign":
            
            cmd = prepare_RosettaDesign(self, index, cmd)
            logging.debug(f"Run RosettaDesign for index {index} based on index {index}.")
            
        elif design_step == "RosettaRelax":
            
            cmd = prepare_RosettaRelax(self, index, cmd)
            logging.debug(f"Run RosettaRelax for index {index}.")
            
        elif design_step == "ESMfold":
            
            cmd = prepare_ESMfold(self, index, cmd)
            logging.debug(f"Run ESMfold for index {index}.")
            
        elif design_step == "ElectricFields":
            
            cmd = prepare_efields(self, index, cmd)
            logging.debug(f"Calculating ElectricFields for index {index}.")
            
        else:
            
            logging.error(f"{design_step} is not defined!")
            sys.exit()
                 
    # Write the shell command to a file and submit job                
    job = "_".join(design_steps)
    with open(f'{self.FOLDER_HOME}/{index}/scripts/{job}_{index}.sh','w') as file: file.write(cmd)
    submit_job(self, index=index, job=job, ram=ram, bash=bash)