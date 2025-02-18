import sys, os
sys.path.append(os.path.join(os.getcwd(), '../../src'))
from AIzymes_014 import *
AIzymes = AIzymes_MAIN()
AIzymes.initialize(FOLDER_HOME    = 'TEST_SCC', 
                   LOG            = 'debug',
                   PRINT_VAR      = False,
                   UNBLOCK_ALL    = True)
AIzymes.controller()
