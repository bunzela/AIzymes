
"""
Main Design Module. Coordinates various design steps, managing the workflow of Rosetta, ProteinMPNN, and other modules
within the AIzymes project.

Functions:
    run_design: Runs the selected design step based for the given index.

Modules Required:
    helper_002, design_match_001, design_ProteinMPNN_001, design_LigandMPNN_001,
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

def run_design(self, 
               index : int,
               design_step : str,
               bash = False
              ):
    
    """
    Start a job by creating and executing a submission script based on the current index and design_step.

    Parameters:
        index (int):        The index representing the current design.
        design_step (str):  The design_step to be executed.
    """
                                
    cmd = f""" cd {self.FOLDER_DESIGN}/{index}
pwd
""" 

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
            logging.debug(f"Assigned GPU for {index}_{design_step}. GPUs: {self.gpus}")

    # Add correct design method
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

    elif design_step == "Null":
        cmd = prepare_null(self, index)
        logging.info(f"Prepare Null for index {index}.")
        
    else:
        logging.error(f"{design_step} is not defined! Error in run_design() / main_design.py")
        sys.exit()
                 
    # Write the shell command to a file and submit job  
    with open(f'{self.FOLDER_DESIGN}/{index}/scripts/{design_step}_{index}.sh','w') as file: file.write(cmd)
        
    # Bash submission script parallel in background
    out_file = open(f"{self.FOLDER_DESIGN}/{index}/scripts/{design_step}_{index}.out", "w")
    err_file = open(f"{self.FOLDER_DESIGN}/{index}/scripts/{design_step}_{index}.err", "w")

    process = subprocess.Popen(f'bash -l -c "bash {self.FOLDER_DESIGN}/{index}/scripts/{design_step}_{index}.sh"', 
                               shell=True, 
                               stdout=out_file, 
                               stderr=err_file)
    self.processes.append((process, out_file, err_file)) 

    # Attach Process to GPU list, if its a GPU Job
    with open(f'{self.FOLDER_DESIGN}/{index}/scripts/{design_step}_{index}.sh', "r") as f: script = f.read()
    match = re.search(r'CUDA_VISIBLE_DEVICES\s*=\s*([0-9]+)', script)
    if match: 
        gpu = int(match.group(1))
        self.gpus[gpu] = process 
    
    logging.debug(f'Job started with {self.FOLDER_DESIGN}/{index}/scripts/submit_{design_step}_{index}.sh')  