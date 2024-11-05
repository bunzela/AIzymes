
"""
Handles RosettaDesign steps to optimize protein design scores and enhance stability
within the AIzymes project.

Functions:
    prepare_RosettaDesign: Prepares RosettaDesign commands for job submission.

Modules Required:
    helper_001
"""
import logging

from helper_001               import *

def prepare_RosettaDesign(self, 
                          new_index,
                          cmd):
    """
    Designs protein structure in {new_index} based on {parent_index} using RosettaDesign.
    
    Args:
    parent_index (str): Index of the parent protein variant to be designed.
    new_index (str):    Index assigned to the resulting design.
    input_suffix (str): Suffix of the input structure to be used for design.
    
    Returns:
    cmd (str): Command to be exected by run_design using submit_job.
    """
    
    # Options for EXPLORE, accelerated script for testing
    if self.EXPLORE:
        ex = ""
    else:
        ex = "-ex1 -ex2"
    
    PDB_input ,_ ,_ = get_PDB_in(self, new_index)
    
    cmd += f"""### RosettaDesign ###
   
# Run RosettaDesign
{self.ROSETTA_PATH}/bin/rosetta_scripts.{self.rosetta_ext} \\
    -s                                        {PDB_input}.pdb \\
    -in:file:native                           {PDB_input}.pdb \\
    -run:preserve_header                      true \\
    -extra_res_fa                             {self.FOLDER_INPUT}/{self.LIGAND}.params \\
    -enzdes:cstfile                           {self.FOLDER_INPUT}/{self.CST_NAME}.cst \\
    -enzdes:cst_opt                           true \\
    -parser:protocol                          {self.FOLDER_HOME}/{new_index}/scripts/RosettaDesign_{new_index}.xml \\
    -out:file:scorefile                       {self.FOLDER_HOME}/{new_index}/score_RosettaDesign.sc \\
    -nstruct                                  1  \\
    -ignore_zero_occupancy                    false  \\
    -corrections::beta_nov16                  true \\
    -overwrite {ex}

# Cleanup
mv {self.FOLDER_HOME}/{new_index}/{os.path.basename(PDB_input)}_0001.pdb \\
   {self.FOLDER_HOME}/{new_index}/{self.WT}_RosettaDesign_{new_index}.pdb 
   
# Get sequence
{self.bash_args}python {self.FOLDER_PARENT}/extract_sequence_from_pdb.py \\
    --pdb_in       {self.FOLDER_HOME}/{new_index}/{self.WT}_RosettaDesign_{new_index}.pdb \\
    --sequence_out {self.FOLDER_HOME}/{new_index}/{self.WT}_{new_index}.seq


"""
                
    # Create XML script for Rosetta Design  
    repeats = "3"
    if self.EXPLORE: repeats = "1"
        
    RosettaDesign_xml = f"""
<ROSETTASCRIPTS>

    <SCOREFXNS>

        <ScoreFunction            name="score"                           weights="beta_nov16" >  
            <Reweight             scoretype="atom_pair_constraint"       weight="1" />
            <Reweight             scoretype="angle_constraint"           weight="1" />    
            <Reweight             scoretype="dihedral_constraint"        weight="1" />        
            <Reweight             scoretype="res_type_constraint"        weight="1" />              
        </ScoreFunction>
       
        <ScoreFunction            name="score_unconst"                   weights="beta_nov16" >        
            <Reweight             scoretype="atom_pair_constraint"       weight="0" />
            <Reweight             scoretype="dihedral_constraint"        weight="0" />
            <Reweight             scoretype="angle_constraint"           weight="0" />              
        </ScoreFunction>

        <ScoreFunction            name="score_final"                     weights="beta_nov16" >    
            <Reweight             scoretype="atom_pair_constraint"       weight="1" />
            <Reweight             scoretype="angle_constraint"           weight="1" />    
            <Reweight             scoretype="dihedral_constraint"        weight="1" />               
        </ScoreFunction>
   
   </SCOREFXNS>
   
    <RESIDUE_SELECTORS>
   
        <Index                    name="sel_design"
                                  resnums="{self.DESIGN}" />
"""
    
    # Add residue number constraints from REMARK (via all_scores_df['cat_resi'])
    cat_resis = str(self.all_scores_df.at[new_index, 'cat_resi']).split(';')
    for idx, cat_resi in enumerate(cat_resis): 
        
        RosettaDesign_xml += f"""
        <Index                    name="sel_cat_{idx}"
                                  resnums="{int(float(cat_resi))}" />
"""
        
    RosettaDesign_xml += f"""
        <Not                      name="sel_nothing"
                                  selector="sel_design" />
    </RESIDUE_SELECTORS>
   
    <TASKOPERATIONS>
   
        <OperateOnResidueSubset   name="tsk_design"                      selector="sel_design" >
                                  <RestrictAbsentCanonicalAASRLT         aas="GPAVLIMFYWHKRQNEDST" />
        </OperateOnResidueSubset>
"""
    
    # Add residue identity constraints from constraint file
    with open(f'{self.FOLDER_HOME}/cst.dat', 'r') as f: cat_resns = f.read()
    cat_resns = cat_resns.split(";")
    
    for idx, cat_resn in enumerate(cat_resns): 
        RosettaDesign_xml += f"""
        <OperateOnResidueSubset   name="tsk_cat_{idx}"                   selector="sel_cat_{idx}" >
                                  <RestrictAbsentCanonicalAASRLT         aas="{cat_resn}" />
        </OperateOnResidueSubset>
"""
    
    tsk_cat = []
    for idx, cat_res in enumerate(cat_resns): 
        tsk_cat += [f"tsk_cat_{idx}"]
    tsk_cat = ",".join(tsk_cat)
        
    RosettaDesign_xml += f"""
       
       
        <OperateOnResidueSubset   name="tsk_nothing"                     selector="sel_nothing" >
                                  <PreventRepackingRLT />
        </OperateOnResidueSubset>
       
    </TASKOPERATIONS>

    <FILTERS>
   
        <HbondsToResidue          name="flt_hbonds"
                                  scorefxn="score"
                                  partners="1"
                                  residue="1X"
                                  backbone="true"
                                  sidechain="true"
                                  from_other_chains="true"
                                  from_same_chain="false"
                                  confidence="0" />
    </FILTERS>
   
    <MOVERS>
       
        <FavorSequenceProfile     name="mv_native"
                                  weight="{self.CST_WEIGHT}"
                                  use_native="true"
                                  matrix="IDENTITY"
                                  scorefxns="score" />  

        <EnzRepackMinimize        name="mv_cst_opt"
                                  scorefxn_repack="score"
                                  scorefxn_minimize="score_final"
                                  cst_opt="true"
                                  task_operations="tsk_design,tsk_nothing,{tsk_cat}" />
                               
        <AddOrRemoveMatchCsts     name="mv_add_cst"
                                  cst_instruction="add_new"
                                  cstfile="{self.FOLDER_INPUT}/{self.CST_NAME}.cst" />

        <FastDesign               name                   = "mv_design"
                                  disable_design         = "false"
                                  task_operations        = "tsk_design,tsk_nothing,{tsk_cat}"
                                  repeats                = "{repeats}"
                                  ramp_down_constraints  = "false"
                                  scorefxn               = "score" />
        
        <FastDesign               name                   = "mv_design_no_native"
                                  disable_design         = "false"
                                  task_operations        = "tsk_design,tsk_nothing,{tsk_cat}"
                                  repeats                = "1"
                                  ramp_down_constraints  = "false"
                                  scorefxn               = "score" />
                                  
        <FastRelax                name                   = "mv_relax"
                                  disable_design         = "true"
                                  task_operations        = "tsk_design,tsk_nothing,{tsk_cat}"
                                  repeats                = "1"
                                  ramp_down_constraints  = "false"
                                  scorefxn               = "score_unconst" />  
                                  
        <InterfaceScoreCalculator name                   = "mv_inter"
                                  chains                 = "X"
                                  scorefxn               = "score_final" />
                                 
    </MOVERS>

    <PROTOCOLS>
        <Add mover_name="mv_add_cst" />
        <Add mover_name="mv_cst_opt" />
        <Add mover_name="mv_native" />
        <Add mover_name="mv_design" />
        <Add mover_name="mv_relax" />
        <Add mover_name="mv_inter" />
    </PROTOCOLS>
   
</ROSETTASCRIPTS>

"""
    # Write the XML script to a file
    with open(f'{self.FOLDER_HOME}/{new_index}/scripts/RosettaDesign_{new_index}.xml', 'w') as f:
        f.writelines(RosettaDesign_xml)               

    return cmd