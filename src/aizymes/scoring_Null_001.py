import os
import subprocess
import numpy as np
import pandas as pd

from helper_002               import *


def prepare_Null(self, index):
    PDB_input = self.all_scores_df.at[int(index), "step_input_variant"] 

    cmd = f"""
cp {PDB_input}.pdb \\
{self.FOLDER_DESIGN}/{index}/{self.WT}_Null_{index}.pdb 

{self.bash_args}python {self.FOLDER_PARENT}/extract_sequence_from_pdb.py \\
    --pdb_in       {self.FOLDER_DESIGN}/{index}/{self.WT}_Null_{index}.pdb \\
    --sequence_out {self.FOLDER_DESIGN}/{index}/{self.WT}_{index}.seq
"""
    return cmd
