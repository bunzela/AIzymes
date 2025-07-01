# Full Manual

## Input preperation

Input file in same directory as Jupyter notebook. 

### Input files for Rosettta

Add description how to make .params and .cst files

Does input structure need REMARK match?

### Input files Amber (MDMin, FieldTools)

Add description how to make .frcmod and .prepi files

## Create an Instance of the AI.zymes class

Add description here to what this cell is doing

```
FOLDER_HOME = '/raven/ptmp/bunzela/AIzymes/TEST'
import sys, os
from AIzymes_015 import *
AIzymes = AIzymes_MAIN(FOLDER_HOME = FOLDER_HOME)
```

## Set up AI.zymes for design

Add description here to what this cell is doing AIzymes.setu

**AIzymes.setup() options**<br>
WT                  <d> Name of the protein scaffold.<br>
                    <t> Must fit to the input pdb structure in **Input**.<br>
                    <t> if **FOLDER_PAR_STRUC** is set (see below), a **WT** structure<br>
                    <t> is still required to create a reference sequence.<br>
LIGAND              <d> Ligand name. Must fit all ligand input files in the folder **Input**.<br>
DESIGN              <d> Comma-seperated string of residues that will be targeted with **RosettaDesign**.<br>
                    <t> Usually, these are active-site or ligand-binding residues.<br>
                    <t> Residue numbering starts with 1.<br>
PARENT_DES_MED      <d> List of Methods to be used for the re-design of the parent input structures.<br>
                    <t> Default: ['RosettaDesign','ElectricFields']<br>
DESIGN_METHODS      <d> List of Lists defining the design methods after parent design is completed.<br>
                    <t> In addition to the design methods, each list must start with a float.<br>
                    <t> The flot indictes the probability of the specific design methdos to be used.<br>
                    <t> Default: <br>
                    <t> [[0.7,'RosettaDesign','ElectricFields'],\<br>
                    <t> [0.3,'ProteinMPNN','ESMfold','RosettaRelax','ElectricFields']]<br>
EXPLORE             <d> Set to True to activate EXPLORE modus.<br>
                    <t> During EXPLORE, design will be performed faster<br>
                    <t> Only use this EXPLORE for testing and not for actually design!<br>
                    <t> Default: None<br>
RESTRICT_RESIDUES   <d>     <br>
FOLDER_PARENT       <d><br>
                    <t> Default: 'parent'<br>
FOLDER_PAR_STRUC    <d><br>

### General settings

### General design settings

### RosettaDesign settings

### RosettaDRelax settings

### ProteinMPNN settings

### AlphaFold3 settings

### ESMFold settings

### MDMin settings

MDMin relies on Amber to minimize the input structure. MD minimization is requested by adding **MDMin** to **DESIGN_METHODS**.



### Electric Field settings

Selection for electric fields using **efield** in **SELECTED_SCORES**. Electric field calculation is requested by adding **ElectricFields** to **DESIGN_METHODS**.<br>

**AIzymes.setup() options**
WEIGHT_EFIELD       <d> Adjust the weight of the **efield_score**<br>        
FIELDS_EXCL_CAT     <d> Excluding the field effect of catalytic residues, recommended False<br>    
FIELD_TARGET        <d> Define the bond along which the electric field is calculated.<br> 
                    <t> For Kemp this was: **:5TS@C9 :5TS@H04**.<br>
                    <t> Atoms along which the field is calculated are defined using Amber selection masks.<br>
                    <t> **":"** indicates the residue name or number, wheras **"@"** describes the atom name.<br> 

> [!NOTE] 
> In its current version, AI.zymes aims to design the highest possible poisitive electric field.
> To design negative electic fields, swap the two atoms defined in the **FIELD_TARGET**.

### BioDC settings

*work in progress...*

## Submit controller to run AI.zymes

Add description here to what this cell is doing

```
AIzymes.submit_controller()              
```

## Analysis tools to check AI.zymes

Add description here to what this cell is doing

```
AIzymes.plot(...)          
AIzymes.print_statistics()               
```

## Extract best structures after AI.zymes run

Add description here to what this cell is doing

```
AIzymes.best_structures()      
```