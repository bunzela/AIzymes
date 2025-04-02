"""
Main Design Module

Coordinates various design steps, managing the workflow of Rosetta, ProteinMPNN, and other modules
within the AIzymes project.

Functions:
    - get_ram: Determines RAM allocation for design steps.
    - run_design: Runs the selected design steps based on configuration.

Modules Required:
    - helper_002, design_match_001, design_ProteinMPNN_001, design_LigandMPNN_001,
      design_RosettaDesign_001, design_ESMfold_001, design_RosettaRelax_001
"""
import logging
import sys

from helper_002               import *

from design_match_001         import *
from design_MPNN_001          import *
from design_RosettaDesign_001 import *
from design_ESMfold_001       import *
from design_RosettaRelax_001  import *
from scoring_efields_001      import *   
from design_MDMin_001         import *  
from design_AlphaFold3_001    import *   
from scoring_BioDC_001        import *      
from design_Boltz_001         import *

def get_ram(design_steps):
    
    ram = 0
    
    for design_step in design_steps:
        
        if design_step == "ProteinMPNN":
            new_ram = 20
        elif design_step == "SolubleMPNN":
            new_ram = 20
        elif design_step == "LigandMPNN":
            new_ram = 20
        elif design_step == "AlphaFold3MSA":
            new_ram = 20
        elif design_step == "AlphaFold3INF":
            new_ram = 20
        elif design_step == "BoltzINF":
            new_ram = 20
        elif design_step == "RosettaDesign":
            new_ram = 10
        elif design_step == "RosettaRelax": 
            new_ram = 10
        elif design_step == "MDMin": 
            new_ram = 10
        elif design_step == "BioDC": 
            new_ram = 10
        elif design_step == "ESMfold":
            new_ram = 40
        elif design_step == "ElectricFields":
            new_ram = 10
        else:
            logging.error(f"RAM for design_step {design_step} is not defined in get_ram() / main_design.py")
            sys.exit()
            
        if new_ram > ram: ram = new_ram
            
    return ram

def run_design(self, 
               index,
               design_steps :list,
               bash = False
              ):
    
    # Expecting list. To make sure individual commmand would also be accepted, convert string to list if string is given
    if not isinstance(design_steps, list):
        variable = [design_steps]
    
    ram = get_ram(design_steps)
    
    cmd = ""
     
    for design_step in design_steps:

        # Assign GPU!
        gpu_id = None
        if self.MAX_GPUS > 0:
            if design_step in self.SYS_GPU_METHODS:
                for idx, job in self.gpus.items():
                    if job is None: 
                        gpu_id = idx
                        break
                if gpu_id == None:
                    logging.error(f"Failed to assign a GPU for {design_step} {index}. GPUs: {self.gpus}. Error in run_design() / main_design.py")
                    sys.exit()    

                # IS THIS THE PROBLEM???
                #gpu_summary = {key: ('busy' if value is not None else None) for key, value in self.gpus.items()}
                #gpu_summary[gpu_id] = 'assigned'
                logging.debug(f"Assigned GPU for {index}_{design_step}. GPUs: {self.gpus}")
        
        if design_step == "ProteinMPNN":
            cmd = prepare_ProteinMPNN(self, index, cmd, gpu_id = gpu_id)
            logging.info(f"Run ProteinMPNN for index {index}.")
        
        elif design_step == "SolubleMPNN":
            cmd = prepare_SolubleMPNN(self, index, cmd, gpu_id = gpu_id)
            logging.info(f"Run SolubleMPNN for index {index}.")

        elif design_step == "LigandMPNN":
            cmd = prepare_LigandMPNN(self, index, cmd, gpu_id = gpu_id)
            logging.info(f"Run LigandMPNN for index {index}.")
        
        elif design_step == "AlphaFold3MSA":
            cmd = prepare_AlphaFold3_MSA(self, index, cmd)
            logging.info(f"Run AlphaFold3MSA for index {index}.")

        elif design_step == "AlphaFold3INF":
            cmd = prepare_AlphaFold3_INF(self, index, cmd, gpu_id = gpu_id)
            logging.info(f"Run AlphaFold3INF for index {index}.")
            
        elif design_step == "BoltzINF":
            cmd = prepare_Boltz_INF(self, index, cmd, gpu_id = gpu_id)
            logging.info(f"Run BoltzINF for index {index}.")
            
        elif design_step == "RosettaDesign":
            cmd = prepare_RosettaDesign(self, index, cmd)
            logging.info(f"Run RosettaDesign for index {index} based on index {index}.")
            
        elif design_step == "RosettaRelax":
            cmd = prepare_RosettaRelax(self, index, cmd)
            logging.info(f"Run RosettaRelax for index {index}.")
        
        elif design_step == "MDMin":
            cmd = prepare_MDMin(self, index, cmd)
            logging.info(f"Run MD minimise for index {index}.")
            
        elif design_step == "ESMfold":
            cmd = prepare_ESMfold(self, index, cmd, gpu_id = gpu_id)
            logging.info(f"Run ESMfold for index {index}.")
            
        elif design_step == "ElectricFields":
            cmd = prepare_efields(self, index, cmd)
            logging.info(f"Calculating ElectricFields for index {index}.")
                        
        elif design_step == "BioDC":
            cmd = prepare_BioDC(self, index, cmd)
            logging.info(f"Calculating Redoxpotentials for index {index}.")
            
        else:
            logging.error(f"{design_step} is not defined! Error in run_design() / main_design.py")
            sys.exit()
                 
    # Write the shell command to a file and submit job                
    job = "_".join(design_steps)
    with open(f'{self.FOLDER_DESIGN}/{index}/scripts/{job}_{index}.sh','w') as file: file.write(cmd)
    submit_job(self, index=index, job=job, ram=ram, bash=bash)