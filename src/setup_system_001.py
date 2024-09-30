import sys

def set_system(self):
    
    if self.SYSTEM == 'GRID':       
        
        self.rosetta_ext = "linuxgccrelease"
        self.bash_args = ""
        
    elif self.SYSTEM == 'BLUEPEBBLE':     
        
        self.rosetta_ext = "serialization.linuxgccrelease"
        self.bash_args = ""
        
    elif self.SYSTEM == 'BACKGROUND_JOB': 
        
        self.rosetta_ext = "serialization.linuxgccrelease"
        self.bash_args = ""
        
    elif self.SYSTEM == 'ABBIE_LOCAL':    
        
        self.rosetta_ext = "linuxgccrelease"
        self.bash_args = "OMP_NUM_THREADS=1"
        
    else:
        
        logging.error(f"{self.SYSTEM} not recognized!")
        print(f"{self.SYSTEM} not recognized!")
        sys.exit()
        