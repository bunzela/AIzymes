"""
Prepares commands to run RosettaDesign methods for protein design.

Functions:
    prepare_RosettaDesign: Prepares RosettaDesign commands for job submission.
"""

import os
import logging

def prepare_RosettaDesign(self, 
                          index,
                          cmd):
    """
    Designs protein structure in {index} using RosettaDesign.
    
    Args:
    index (str): Index assigned to the resulting design.
    
    Returns:
    cmd (str): Command to be exected by run_design using submit_job.
    """
    
    # Options for EXPLORE, accelerated script for testing
    if self.EXPLORE:
        ex = ""
    else:
        ex = "-ex1 -ex2"
    
    PDB_input = self.all_scores_df.at[int(index), "step_input_variant"]

    cmd += f"""### RosettaDesign ###
   
# Run RosettaDesign
{self.ROSETTA_PATH}/bin/rosetta_scripts.{self.rosetta_ext} \\
    -s                                        {PDB_input}.pdb \\
    -in:file:native                           {PDB_input}.pdb \\
    -run:preserve_header                      true """
    if self.LIGAND not in ['HEM']:
        cmd += f"""\\
    -extra_res_fa                             {self.FOLDER_INPUT}/{self.LIGAND}.params """
    if self.CST_NAME is not None:
        cmd += f"""\\
    -enzdes:cstfile                           {self.FOLDER_PARENT}/{self.CST_NAME}.cst \\
    -enzdes:cst_opt                           true """
    cmd += f"""\\
    -parser:protocol                          {self.FOLDER_DESIGN}/{index}/scripts/RosettaDesign_{index}.xml \\
    -out:file:scorefile                       {self.FOLDER_DESIGN}/{index}/score_RosettaDesign.sc \\
    -nstruct                                  1  \\
    -ignore_zero_occupancy                    false \\
    -corrections::beta_nov16                  true \\
    -overwrite {ex}

# Cleanup
mv {self.FOLDER_DESIGN}/{index}/{os.path.basename(PDB_input)}_0001.pdb \\
   {self.FOLDER_DESIGN}/{index}/{self.WT}_RosettaDesign_{index}.pdb 
   
# Get sequence
{self.bash_args}python {self.FOLDER_PARENT}/extract_sequence_from_pdb.py \\
    --pdb_in       {self.FOLDER_DESIGN}/{index}/{self.WT}_RosettaDesign_{index}.pdb \\
    --sequence_out {self.FOLDER_DESIGN}/{index}/{self.WT}_{index}.seq

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
    if self.CST_NAME is not None:
        cat_resis = str(self.all_scores_df.at[index, 'cat_resi']).split(';')
        for idx, cat_resi in enumerate(cat_resis): 
            RosettaDesign_xml += f"""
        <Index                    name="sel_cat_{idx}"
                                  resnums="{int(float(cat_resi))}" />
"""

    if self.RESTRICT_RESIDUES is not None:
        for idx, rest_resi in enumerate(self.RESTRICT_RESIDUES): 
            RosettaDesign_xml += f"""
        <Index                    name="sel_rest_{idx}"
                                  resnums="{int(float(rest_resi[0]))}" />
"""

    allowed_aa = ''.join(sorted(set('ACDEFGHIKLMNPQRSTVWY') - set(self.FORBIDDEN_AA)))
    
    RosettaDesign_xml += f"""
        <Not                      name="sel_nothing"
                                  selector="sel_design" />
    </RESIDUE_SELECTORS>
   
    <TASKOPERATIONS>
   
        <OperateOnResidueSubset   name="tsk_design"                      selector="sel_design" >
                                  <RestrictAbsentCanonicalAASRLT         aas="{allowed_aa}" />
        </OperateOnResidueSubset>
"""
    
    # Add residue identity constraints from constraint file
    if self.CST_NAME is not None:

        with open(f'{self.FOLDER_HOME}/cst.dat', 'r') as f:
            cat_resns = f.read()
        cat_resns = cat_resns.split(";")
    
        for idx, cat_resn in enumerate(cat_resns): 
            RosettaDesign_xml += f"""
        <OperateOnResidueSubset   name="tsk_cat_{idx}"                   selector="sel_cat_{idx}" >
                                  <RestrictAbsentCanonicalAASRLT         aas="{cat_resn}" />
        </OperateOnResidueSubset>
"""

    if self.RESTRICT_RESIDUES is not None:
        for idx, rest_resi in enumerate(self.RESTRICT_RESIDUES):
            RosettaDesign_xml += f"""
        <OperateOnResidueSubset   name="tsk_rest_{idx}"                  selector="sel_rest_{idx}" >
                                  <RestrictAbsentCanonicalAASRLT         aas="{rest_resi[1]}" />
        </OperateOnResidueSubset>
"""
    
    # Generate list of tsk_cat
    if self.CST_NAME is not None:
        tsk_cat = []
        for idx, cat_res in enumerate(cat_resns): 
            tsk_cat += [f"tsk_cat_{idx}"]
        tsk_cat = f",{','.join(tsk_cat)}"
    else:
        tsk_cat = ''

    # Generate list of tsk_rest
    if self.RESTRICT_RESIDUES is not None:
        tsk_rest = []
        for idx, _ in enumerate(self.RESTRICT_RESIDUES): 
            tsk_rest += [f"tsk_rest_{idx}"]
        tsk_rest = f",{','.join(tsk_rest)}"
    else:
        tsk_rest = ''
        
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
"""
    if self.CST_NAME is not None:
        RosettaDesign_xml += f"""
        <AddOrRemoveMatchCsts     name="mv_add_cst"
                                  cst_instruction="add_new"
                                  cstfile="{self.FOLDER_INPUT}/{self.CST_NAME}.cst" />

        <EnzRepackMinimize        name="mv_cst_opt"
                                  scorefxn_repack="score"
                                  scorefxn_minimize="score_final"
                                  cst_opt="true"
                                  task_operations="tsk_design,tsk_nothing{tsk_cat}{tsk_rest}" />
"""
    RosettaDesign_xml += f"""
        <FastDesign               name                   = "mv_design"
                                  disable_design         = "false"
                                  task_operations        = "tsk_design,tsk_nothing{tsk_cat}{tsk_rest}"
                                  repeats                = "{repeats}"
                                  ramp_down_constraints  = "false"
                                  scorefxn               = "score" />
                                  
        <FastRelax                name                   = "mv_relax"
                                  disable_design         = "true"
                                  task_operations        = "tsk_design,tsk_nothing{tsk_cat}{tsk_rest}"
                                  repeats                = "1"
                                  ramp_down_constraints  = "false"
                                  scorefxn               = "score_unconst" />  
                                  
        <InterfaceScoreCalculator name                   = "mv_inter"
                                  chains                 = "X"
                                  scorefxn               = "score_final" />
                                 
    </MOVERS>

    <PROTOCOLS>"""
    if self.CST_NAME is not None:
        RosettaDesign_xml += f"""
        <Add mover_name="mv_add_cst" />
        <Add mover_name="mv_cst_opt" />"""
    RosettaDesign_xml += f"""
        <Add mover_name="mv_native" />
        <Add mover_name="mv_design" />
        <Add mover_name="mv_relax" />
        <Add mover_name="mv_inter" />
    </PROTOCOLS>
   
</ROSETTASCRIPTS>

"""
    # Write the XML script to a file
    with open(f'{self.FOLDER_DESIGN}/{index}/scripts/RosettaDesign_{index}.xml', 'w') as f:
        f.writelines(RosettaDesign_xml)               

    return cmd