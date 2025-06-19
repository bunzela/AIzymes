
"""
Contains system specifc information. At the Moment, this is all hard-coded.
In the future, this will be part of the installation of AIzymes.

set_system() contains general variables.
submit_head() constructs the submission header to submit jobs.

"""

import sys
import os
import getpass
import json
from time import sleep

def set_system(self):

    standard_settings = {
        "rosetta_ext"          : "linuxgccrelease",
        "bash_args"            : "",
        "ROSETTA_PATH"         : f"/u/bunzela/bin/rosetta.source.release-371/main/source/",
        "FIELD_TOOLS"          : f'/u/{getpass.getuser()}/bin/AIzymes_dev/src/aizymes/FieldTools.py',
        "FOLDER_ProteinMPNN"   : f'/u/{getpass.getuser()}/bin/ProteinMPNN',
        "FOLDER_ProteinMPNN_h" : f'/u/{getpass.getuser()}/bin/ProteinMPNN/helper_scripts',
        "FOLDER_LigandMPNN"    : f'/u/{getpass.getuser()}/bin/LigandMPNN',
        "FOLDER_Alphafold"     : f'/u/bunzela/bin/alphafold',
        "SUBMIT_HEAD"          : submit_head(self),
    }

    if os.path.isfile(self.SYSTEM):

        # Load settings from config
        with open(self.SYSTEM, 'r') as f:
            loaded_settings = json.load(f)
        
        # Assign to self
        for key, value in loaded_settings.items():
            setattr(self, key, value)
            
    else:

        # MAke sure config file is not save in a path that is deleted during AI.zymes setup
        system_path = os.path.abspath(self.SYSTEM)
        current_dir = os.path.abspath(self.FOLDER_HOME)
        if os.path.commonpath([system_path, current_dir]) == current_dir:
            print(f"Error! {self.SYSTEM} cannot be in current directory or subdirectory thereof")
            sys.exit()
        
        print(f"Config file {self.SYSTEM} missing. Lets create a new config file!\n")

        default = standard_settings["ROSETTA_PATH"]
        self.ROSETTA_PATH = input(f"Path to Rosetta [leavel blank for default: {default}]") or default
        print(f"ROSETTA_PATH set to {self.ROSETTA_PATH}\n")
        if not os.path.isdir(self.ROSETTA_PATH):
            print(f"ERROR, ROSETTA_PATH {self.ROSETTA_PATH} does not exist!")
            sys.exit()
            
        default = standard_settings["FIELD_TOOLS"]
        self.FIELD_TOOLS = input(f"Path to FieldTools [leavel blank for default: {default}]") or default
        print(f"FIELD_TOOLS set to {self.FIELD_TOOLS}\n")
        if not os.path.isfile(self.FIELD_TOOLS):
            print(f"ERROR, FIELD_TOOLS {self.FIELD_TOOLS} does not exist!")
            sys.exit()
        
        default = standard_settings["FOLDER_ProteinMPNN"]
        self.FOLDER_ProteinMPNN = input(f"Path to ProteinMPNN [leavel blank for default: {default}]") or default
        print(f"ProteinMPNN set to {self.FOLDER_ProteinMPNN}\n")
        if not os.path.isdir(self.FOLDER_ProteinMPNN):
            print(f"ERROR, FOLDER_ProteinMPNN {self.FOLDER_ProteinMPNN} does not exist!")
            sys.exit()

        self.FOLDER_ProteinMPNN_h = f'{self.FOLDER_ProteinMPNN}/helper_scripts'
        if not os.path.isdir(self.FOLDER_ProteinMPNN_h):
            print(f"ERROR, FOLDER_ProteinMPNN_h {self.FOLDER_ProteinMPNN_h} does not exist!")
            sys.exit()

        default = standard_settings["FOLDER_LigandMPNN"]
        self.FOLDER_LigandMPNN = input(f"Path to LigandMPNN [leavel blank for default: {default}]") or default
        print(f"FOLDER_LigandMPNN set to {self.FOLDER_LigandMPNN}\n")
        if not os.path.isdir(self.FOLDER_LigandMPNN):
            print(f"ERROR, FOLDER_LigandMPNN {self.FOLDER_LigandMPNN} does not exist!")
            sys.exit()

        default = standard_settings["FOLDER_Alphafold"]
        self.FOLDER_Alphafold = input(f"Path to Alphafold [leavel blank for default: {default}]") or default
        print(f"FOLDER_ALPHAFOLD set to {self.FOLDER_Alphafold}\n")
        if not os.path.isdir(self.FOLDER_Alphafold):
            print(f"ERROR, FOLDER_ALPHAFOLD {self.FOLDER_Alphafold} does not exist!")
            sys.exit()

        default = standard_settings["rosetta_ext"]
        self.rosetta_ext = input(f"Extension for installed Rosetta Programms [leavel blank for default: {default}]") or default
        print(f"rosetta_ext set to {self.rosetta_ext}\n")

        self.bash_args = standard_settings["bash_args"]
        
        default = standard_settings["SUBMIT_HEAD"]
        self.SUBMIT_HEAD = input(f"""Path to controller submission header.
Do not include --job-name, --output, and --error flags!
[leave blank for default header]""") or default
        if self.SUBMIT_HEAD != default:
            if not os.path.isfile(self.SUBMIT_HEAD):
                print(f"ERROR, SUBMIT_HEAD {self.SUBMIT_HEAD} does not exist!")
                sys.exit()
            with open(self.SUBMIT_HEAD, "r") as f:
                self.SUBMIT_HEAD = f.read()
        print(f"\nSUBMIT_HEAD set to\n{self.SUBMIT_HEAD}\n")
        
        # Save settings
        standard_settings = {
            "rosetta_ext"          : self.rosetta_ext,
            "bash_args"            : self.bash_args,
            "ROSETTA_PATH"         : self.ROSETTA_PATH,
            "FIELD_TOOLS"          : self.FIELD_TOOLS,
            "FOLDER_ProteinMPNN"   : self.FOLDER_ProteinMPNN,
            "FOLDER_ProteinMPNN_h" : self.FOLDER_ProteinMPNN_h,
            "FOLDER_LigandMPNN"    : self.FOLDER_LigandMPNN,
            "FOLDER_Alphafold"     : self.FOLDER_Alphafold,
            "SUBMIT_HEAD"          : self.SUBMIT_HEAD
        }
        with open(self.SYSTEM, 'w') as f:
            json.dump(standard_settings, f, indent=4) 
        print(f"Config file saved to {self.SYSTEM}\n")
        print(f"TIP: To avoid making a new config file everytime, store path to config file in SYSTEM during AI.zymes setup")
        print()        
        sleep(1.0)

def submit_head(self):

    
    if self.MAX_GPUS == 0: 
        partitition='general'
    else:
        partitition='gpu'
    
    cmd = f"""#!/bin/bash
#SBATCH --partition={partitition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={self.MAX_JOBS}
#SBATCH --mem={self.MEMORY}G
#SBATCH --time=15:00:00
"""
    if self.MAX_GPUS > 0:
        cmd += f"""#SBATCH --gres=gpu:{self.MAX_GPUS}
"""

    return cmd