import sys
import os
import subprocess
import numpy as np  
import pickle
import datetime
import mdtraj as md
import parmed as pmd

def ResNr_from_ResName(ResName, Names):
    for Name_i in range(0,len(Names),1):
        if ResName == Names[Name_i][1]:
            return Names[Name_i][2]

def Calc_Electrostatic_Potential_Energy(_ABC,_XYZ,_Charge,_Target_Charge):
    # Electrostatic Potential energy at _xyz from Atom _abc with _Charge and Target_Charges U = k*q*Q/r
    _Couloumb_Const = 1.00 #(A.U.)
    _Vector = _ABC-_XYZ        
    _Length = np.linalg.norm(_Vector)
    _Length_Bohr = _Length/0.529177249           # Bohr
    _Energy = (_Couloumb_Const*float(_Charge)*float(_Target_Charge)/(_Length_Bohr))   #### WRONG!!!!!!!!!!
    ### 1 a.u. = 2625.5 kJ/mol
    _Energy=_Energy*2625.5    
    return(_Energy)

def Field_of_ABC_at_XYZ_projected_onto_VEC(_ABC,_XYZ,_VEC,_Charge,arg_energy_out,Target_Charges):
    
    # Field at _xyz from Atom _abc with _Charge E = k*Q/r/r
    _Couloumb_Const = 1.00 #(A.U.)
    _Vector = _ABC-_XYZ     
    _Length = np.linalg.norm(_Vector)
    _Normalized_Vector = (-1.0)*_Vector/_Length
    _Length_Bohr = _Length/0.529177249           # Bohr
    _Field = (_Couloumb_Const*float(_Charge)/_Length_Bohr**2)*_Normalized_Vector
    _Field = np.dot(_Field,_VEC)
    ### 1 a.u. = 5.14x10^9 V/cm
    ### 1 a.u. = 5.14x10^3 MV/cm
    _Field=_Field*5.14*1000 / 100

    _Energy = 0.0               
    return _Field,_Energy

def Field_of_ABC_at_XYZ(_ABC,_XYZ,_Charge,arg_energy_out,Target_Charges,arg_vector_out):
    
    # Field at _xyz from Atom _abc with _Charge E = k*Q/r/r 
    _Couloumb_Const = 1.00 #(A.U.)
    _Vector = _ABC-_XYZ     
    _Length = np.linalg.norm(_Vector)
    _Length_Bohr = _Length/0.529177249           # Bohr
    _Field = (-1)*(_Couloumb_Const*float(_Charge)/(_Length_Bohr**2))
    ### 1 a.u. = 5.14x10^9 V/cm
    ### 1 a.u. = 5.14x10^3 MV/cm
    _Field=_Field*5.14*1000 / 100
    
    _Energy = 0.0
    _Field_Vectors = []
    _Field_Vector = "x"
    if arg_energy_out != "False":
        _Energy = Calc_Electrostatic_Potential_Energy(_ABC,_XYZ,_Charge,Target_Charges[0])
    if arg_vector_out != "False": 
        _Normalized_Vector = _Vector/_Length
        _Field_Vector = (-1.0)*(_Couloumb_Const*float(_Charge)/_Length_Bohr**2)*_Normalized_Vector
        #print("Field magnitude:",(-1.0)*(_Couloumb_Const*float(_Charge)/_Length_Bohr**2))
        #print("Field direction:",_Normalized_Vector,"length of vector",np.linalg.norm(_Normalized_Vector))
        #print("Field vector:",_Field_Vector,"length of vector",np.linalg.norm(_Field_Vector))
        
    return _Field,_Energy,_Field_Vector

def fkt_Update_Charges(QM_Charges,QM_Dict,Frame_i,Names,Charges):
    stop="X"
    for QM_Charge_i in range(0,len(QM_Charges),1):
        if len(QM_Charges[QM_Charge_i])==2:
            if QM_Charges[QM_Charge_i][1]==str(Frame_i+1):
                start = QM_Charge_i+1
            if QM_Charges[QM_Charge_i][1]==str(Frame_i+2):
                stop = QM_Charge_i
                QM_Charges = QM_Charges[start:stop]
                break
    if stop == "X":
        QM_Charges = QM_Charges[start:]   
    
    for QM_Atom_i in range(0,len(QM_Dict),1):
        for Atom_i in range(0,len(Names),1):
            Name_tmp = Names[Atom_i][0]+"_"+Names[Atom_i][1]+"_"+Names[Atom_i][2]
            if Name_tmp == QM_Dict[QM_Atom_i]:      #QM_Atom_i index in QM_Charges to be changed from 
                Charges[Atom_i]=QM_Charges[QM_Atom_i][2]
                break
    return Charges

def fkt_Load_QM_Charges(arg_qm_charges,arg_qm_dict):
    with open(arg_qm_charges) as f:
        QM_Charges = [i.split() for i in f.readlines()]
    with open(arg_qm_dict) as f:
        QM_Dict = [i.split("\n")[0] for i in f.readlines()]
    return QM_Charges,QM_Dict

def decompose_fields(Absolut_Fields,Absolut_Field,Names,Atom_Index,arg_solvent):
    #### Total Field
    Absolut_Fields["Total"]+=Absolut_Field
    #### Solvent Field
    if Names[Atom_Index][1] in arg_solvent:
        Absolut_Fields["Solvent"]+=Absolut_Field
        Absolut_Fields[Names[Atom_Index][1]]+=Absolut_Field
    else:
        Absolut_Fields["Protein"]+=Absolut_Field
        Absolut_Fields[Names[Atom_Index][1]+"_"+Names[Atom_Index][2]]+=Absolut_Field
    return Absolut_Fields

def fkt_calc_fields(FIELDS,FIELD,XYZ,VEC,Frame,Charges,Names,Field_Components,arg_solvent,Self_i,arg_exclude_atoms, \
                    arg_energy_out,Target_Index,ENERGIES,arg_vector_out,VECTORS):

### VARIABLES STRUCTURE
    Absolut_Fields = {}
    for Field_Component in Field_Components:
        Absolut_Fields[Field_Component] = 0.0
    Absolut_Energies = {}
    for Field_Component in Field_Components:
        Absolut_Energies[Field_Component] = 0.0

    Target_Charges = Target_Index[FIELD]
    Target_Charges = [Charges[i] for i in Target_Charges]
    for Atom_Index in range(0,len(Frame),1):   
#### Exclude Self-residue!!! 
        if arg_exclude_atoms == "False":
            if not Self_i.isdigit():
                Self_i = ResNr_from_ResName(Self_i,Names)
            if Names[Atom_Index][2] == Self_i: continue        
        else:
            if Atom_Index in arg_exclude_atoms[FIELD]: continue
                
                
        #### Field_of_ABC_at_XYZ(_ABC,_XYZ,_Charge,arg_energy_out,arg_vector_out,Target_Charges):
        if len(FIELD.split("_")) == 1:   #POINT CALCULATION
            Absolut_Field, Absolut_Energy, Vectors  = Field_of_ABC_at_XYZ(Frame[Atom_Index],XYZ,Charges[Atom_Index], \
                                                                   arg_energy_out, Target_Charges, arg_vector_out)
            
        ### Field_of_ABC_at_XYZ_projected_onto_VEC(_ABC,_XYZ,_VEC,_Charge,arg_energy_out,Target_Charges)
        if len(FIELD.split("_")) == 2:   #VECTOR CALCULATION
            Absolut_Field, Absolut_Energy = Field_of_ABC_at_XYZ_projected_onto_VEC(Frame[Atom_Index],XYZ,VEC,\
                                                                                   Charges[Atom_Index],arg_energy_out,\
                                                                                   Target_Charges)
      
        Absolut_Energies      = decompose_fields(Absolut_Energies,Absolut_Energy,Names,Atom_Index,arg_solvent)
        Absolut_Fields        = decompose_fields(Absolut_Fields,Absolut_Field,Names,Atom_Index,arg_solvent)
        
    for Field_Component in Field_Components:
        FIELDS[FIELD][Field_Component].append(Absolut_Fields[Field_Component])
        if arg_energy_out != "False":
            ENERGIES[FIELD][Field_Component].append(Absolut_Energies[Field_Component])
    if len(FIELD.split("_")) == 1: 
        if arg_vector_out != "False":
            VECTORS[FIELD].append(Vectors)
    return FIELDS,ENERGIES,VECTORS

def fkt_Field_Components(FIELDS,Names,arg_solvent):
    
    Field_Components = ["Total","Protein","Solvent"]+arg_solvent
    
    for Name in Names:
        if Name[1] not in arg_solvent:
            if Name[1]+"_"+Name[2] not in Field_Components: Field_Components += [Name[1]+"_"+Name[2]] 

    for Field in FIELDS:
        for Field_Component in Field_Components:
            FIELDS[Field][Field_Component]=[]
    
    return FIELDS,Field_Components

def fkt_get_Target_VEC(Target_XYZ):
    
    Target_VEC = {}

    for Target in Target_XYZ:
        if len(Target_XYZ[Target])==2:
            Target_VEC[Target]={}
            XYZ_START = Target_XYZ[Target][0]
            XYZ_STOP  = Target_XYZ[Target][1]
            Target_VEC[Target]["Center"] = (XYZ_START+XYZ_STOP)/2 
            Target_VEC[Target]["Vector"] = (XYZ_STOP-XYZ_START)/np.linalg.norm(XYZ_STOP-XYZ_START)
            
    return Target_VEC

def fkt_Target_Index(Names,arg_target,FIELDS):
    
    Target_Index={}
    
    with open(arg_target) as f:
        Targets = [i.split() for i in f.readlines()]
        
    for Target in range(len(Targets)-1,-1,-1): #Delete empty lines
        if len(Targets[Target]) == 0:
            Targets = Targets[:Target]+Targets[Target+1:]
    
    for Target_i in range(0,len(Targets),1):
        Target_Name = "_".join(Targets[Target_i])
        FIELDS[Target_Name]={}
        for Atom_i in range(0,len(Targets[Target_i]),1): #If this is a vector, it contains two Atoms!
            Atom_ResidueNr = Targets[Target_i][Atom_i].split("@")[0][1:]
            Atom_Name      = Targets[Target_i][Atom_i].split("@")[1]
            #print Targets[Target_i][Atom_i], Atom_ResidueNr, Atom_Name
            
            if not Atom_ResidueNr.isdigit(): Atom_ResidueNr = ResNr_from_ResName(Atom_ResidueNr,Names)
    
            for Name_i in range(0,len(Names),1):
                if Atom_ResidueNr == Names[Name_i][2]:      #Check if Residue Number matches
                    if Atom_Name == Names[Name_i][0]:       #Check if Atom Name matches
                        Targets[Target_i][Atom_i] = Name_i  #Update Number in Target
                        break
        Target_Index[Target_Name] = Targets[Target_i]
    return Target_Index,FIELDS #Passed to Target_Index in main!

def fkt_Exlude_Index(Names,arg_exclude_atoms,arg_target):
    
    Exclude_Index={}
    
    with open(arg_exclude_atoms) as f:
        Excludes = [i.split() for i in f.readlines()]
    with open(arg_target) as f:
        Targets = [i.split() for i in f.readlines()]

    for Excludes_i in range(0,len(Excludes),1):
        Exclude_Index["_".join(Targets[Excludes_i])]=Excludes[Excludes_i]
        for Atom_i in range(0,len(Excludes[Excludes_i]),1):    #If this is a vector, it contains two Atoms!
            Atom_ResidueNr = Excludes[Excludes_i][Atom_i].split("@")[0][1:]
            if not Atom_ResidueNr.isdigit(): Atom_ResidueNr = ResNr_from_ResName(Atom_ResidueNr,Names)
            Atom_Name      = Excludes[Excludes_i][Atom_i].split("@")[1]
            for Name_i in range(0,len(Names),1):
                if Atom_ResidueNr == Names[Name_i][2]:      #Check if Residue Number matches
                    if Atom_Name == Names[Name_i][0]:       #Check if Atom Name matches
                        Exclude_Index["_".join(Targets[Excludes_i])][Atom_i] = Name_i  #Update Number in Target
                        break
    arg_exclude_atoms =  Exclude_Index              
    return arg_exclude_atoms #Passed to Target_Index in main!

def fkt_get_Target_XYZ(Frame,Target_Index):
#Target_Index is a libary
#Frame is
#Make empty library
    Target_XYZ = {}

    for Target in Target_Index:
        Target_XYZ[Target]=[]
        for Atom_i in range(0,len(Target_Index[Target]),1): #If this is a vector, it contains two Atoms!
            Target_XYZ[Target]=Target_XYZ[Target]+[Frame[Target_Index[Target][Atom_i]]]
                
    return Target_XYZ #Passed to Target_Index in main!

def fkt_Load_Trajectory(arg_nc, arg_parm, arg_TIP4P):
    
    # Load topology using ParmEd
    parm = pmd.load_file(arg_parm)
    
    # Extract charges from ParmEd topology
    Charges = np.array([atom.charge for atom in parm.atoms])
    
    # Load trajectory using MDTraj
    traj = md.load(arg_nc, top=arg_parm)
    Trajectory = traj.xyz  # Coordinates in nanometers
    
    # Extract atomic and residue information
    Names = [[str(atom.name), str(atom.residue.name), str(atom.residue.index + 1)] for atom in traj.topology.atoms]

    if arg_TIP4P == "True":
        TIP4P_indices = [
            i for i, name in enumerate(atom.name for atom in traj.topology.atoms)
            if name == "EPW" and atom.residue.name == "WAT"
        ]
        for index in TIP4P_indices:
            Charges[index - 3] = np.inf  # Ignore charge of linker atoms
            for frame in Trajectory:
                frame[index - 3] = [np.inf, np.inf, np.inf]

        Trajectory = np.array([
            frame[np.all(frame != [np.inf, np.inf, np.inf], axis=1)]
            for frame in Trajectory
        ])
        Charges = Charges[np.isfinite(Charges)]

    return Trajectory, Charges, np.array(Names)

def make_pdb(arg_nc, arg_parm, arg_pdb, arg_use_qmcharges, arg_TIP4P, arg_qm_mask, arg_out):
    traj = md.load(arg_nc, top=arg_parm)
    
    if arg_TIP4P == "True":
        TIP4P_indices = [
            i for i, atom in enumerate(traj.topology.atoms)
            if atom.name == "EPW" and atom.residue.name == "WAT"
        ]
        # Remove extra points from both topology and trajectory
        traj = traj.atom_slice([i for i in range(traj.n_atoms) if i not in TIP4P_indices])
    
    traj.save_pdb(arg_pdb)
        
def load_qm_mask(arg_qm_mask):
    with open(arg_qm_mask) as f:
        arg_qm_mask = f.read()[:-1]
    return arg_qm_mask

def usage():
    print("                                                                                                              ")
    print("|------------------------------------------------------------------------------------------------------------|")
    print("|-------------                                FieldTools Usage:                                 -------------|")
    print("|------------------------------------------------------------------------------------------------------------|")
    print("|-nc             : Specify trajectory file                                                                   |")
    print("|-parm           : Specify parameter file                                                                    |")
    print("|-out            : Specify output file                                                                       |")
    print("|-energy_out     : Optional: Calculate Energies from Fields                                                  |")
    print("|-target         : Specify target file                                                                       |")
    print("|-solvent        : Optional: Comma seperated list of non-protein residues              [default WAT,Na+,Cl- ]|")
    print("|-exclude_atoms  : Optional: Specify what atoms will be excluded from Field calculation                      |")
    print("|                : If not set, Field calculation will exclude the full residue of the first atom defined     |")
    print("|                : for each target.                                                                          |")
    print("|-TIP4P          : Optional: Recognize TIP4P waters                                    [default False       ]|")
    print("|-verbose        : Optional: Display additional information                            [default False       ]|")
    print("|-use_qm_charges : Optional: Specify if qmcharges should be read                       [default False       ]|")
    print("|-vector_out     : Optional: Save Field Vectors to specified file                      [default False       ]|")
    print("|------------------------------------------------------------------------------------------------------------|")
    print("|-------------        If -use_qm_charges = true, the following options have to be set:          -------------|")
    print("|------------------------------------------------------------------------------------------------------------|")
    print("|-qm_mask        : Specify qm region (amber selection mask)                                                  |")
    print("|-qm_charges     : Specify qmcharges file (qm partial charges of each frame, made with QMChargesTools)       |")
    print("|-qm_dict        : Specify qmdict file (containing qmatom names in .pdb and gaussian)                        |")
    print("|------------------------------------------------------------------------------------------------------------|")
    print("|-------------                              How to define -target:                              -------------|")
    print("|------------------------------------------------------------------------------------------------------------|")
    print("|-target should point to a file, each line in that file defines target to calculate the Field                |")
    print("|Targets are selected using the amber selection syntax                                                       |")
    print("|To calculate the magnitude of the field at a specific atom:     :47@OG                                      |")
    print("|To calculate the magnitude of the field projected along a bond: :47@OG :47@OH                               |")
    print("|------------------------------------------------------------------------------------------------------------|")
    print("|-------------                                      INFO                                        -------------|")
    print("|------------------------------------------------------------------------------------------------------------|")
    print("|Script recognizes if total field at one point or projection onto a bond is to be calculated                 |")
    print("|based on the input in -target                                                                               |")
    print("|-------------                                                                                               |")
    print("|Script excludes all Field contributions of the Residue at which the field is calculated.                    |")
    print("|For Vector Projections, this is only the first residue number!                                              |")
    print("|-------------                                                                                               |")
    print("|Solvent = All atoms as defined in -solvent,                                                                 |")
    print("|          for these residues contributions will also be grouped based on the residue name                   |")
    print("|Protein = All atoms that are not part of -solvent,                                                          |")
    print("|          for the protein, per-reside fields will be calculated                                             |")
    print("|-------------                                                                                               |")
    print("|Linker Atom charge is ignored!                                                                              |")
    print("|------------------------------------------------------------------------------------------------------------|") 
    
def inputParser(arg):
    
    arg_nc            = False                                                                         ### Add new flag here
    arg_parm         = False
    arg_out           = False
    arg_energy_out    = "False"
    arg_target        = False
    arg_solvent       = "WAT,Na+,Cl-"
    arg_exclude_atoms = "False" 
    arg_TIP4P         = "False"
    arg_verbose       = "False"
    arg_use_qmcharges = "False"
    arg_vector_out    = "False"
    arg_qm_mask       = False
    arg_qm_charges    = False
    arg_qm_dict       = False  
    
    if(len(arg) == 1) or arg[1] in ["-help","--help","-h"]:
        usage()
        quit()
    
    for index in range(1,len(arg)-1,2):
        param = arg[index]
        if param=="-nc"               : arg_nc            = str(arg[index+1])         ### Add new flag here
        elif param=="-parm"           : arg_parm          = str(arg[index+1])
        elif param=="-out"            : arg_out           = str(arg[index+1])
        elif param=="-energy_out"     : arg_energy_out    = str(arg[index+1])
        elif param=="-target"         : arg_target        = str(arg[index+1])
        elif param=="-solvent"        : arg_solvent       = str(arg[index+1])
        elif param=="-exclude_atoms"  : arg_exclude_atoms = str(arg[index+1])
        elif param=="-qm_dict"        : arg_qm_dict       = str(arg[index+1])
        elif param=="-TIP4P"          : arg_TIP4P         = str(arg[index+1])
        elif param=="-verbose"        : arg_verbose       = str(arg[index+1])
        elif param=="-use_qm_charges" : arg_use_qmcharges = str(arg[index+1])
        elif param=="-vector_out"     : arg_vector_out    = str(arg[index+1])
        elif param=="-qm_mask"        : arg_qm_mask       = str(arg[index+1])
        elif param=="-qm_charges"     : arg_qm_charges    = str(arg[index+1])
        elif param=="-qm_dict"        : arg_qm_dict       = str(arg[index+1])
        else:
            print ("Error! "+param+" not a valid option.")
            usage()
            quit()  
                                                                                                      ### Add new flag here
    if arg_nc        == False: print ("ERROR. Please specify the name of the trajectory file with -nc")  
    if arg_parm     == False: print ("ERROR. Please specify the name of the parameter file with -parm")
    if arg_out       == False: print ("ERROR. Please specify the name of the output file with -out")
    if arg_target    == False: print ("ERROR. Please specify the name of the target file with -target.                                       \n -> See examples with --help")
    if arg_use_qmcharges == "True":
        if arg_qm_mask    == False: print ("ERROR. Please specify qm mask with -qm_mask")
        if arg_qm_charges == False: print ("ERROR. Please specify charge of the qm region with -qm_charge")
        if arg_qm_dict    == False: print ("ERROR. Please specify qm_dict with -qm_dict")
        
    if False in [arg_nc,arg_parm,arg_out,arg_target]:                                                ### Add new flag here
        usage()
        quit()
    if arg_use_qmcharges == "True":
        if False in [arg_qm_mask,arg_qm_charges,arg_qm_dict]:                                         ### Add new flag here
            usage()
            quit()
    
    print ("-nc              Trajectory file      : ",arg_nc)                                         ### Add new flag here
    print ("-parm            Parameter file       : ",arg_parm)
    print ("-out             Output file          : ",arg_out)
    print ("-target          Field target         : ",arg_target)
    print ("-arg_solvent     Non-protein residues : ",arg_solvent)
    if arg_exclude_atoms != "False":
        print ("-exclude_atoms   Atoms excluded       : ",arg_exclude_atoms)
    else:
        print ("-exclude_atoms   Atoms excluded       : Full residue defined in the first Atom for each Target.")
    if arg_TIP4P == "True":
        print ("-TIP4P           Recognize TIP4P wat  : ",arg_TIP4P)
    if arg_energy_out != "False":
        print ("-energy_out      Energy from Fields   : ",arg_energy_out)
    if arg_verbose == "True":
        print ("-arg_verbose     Verbose information  : ",arg_verbose)
    if arg_use_qmcharges == "True":
        print ("-use_qm_charges  QM_theory            :  True")
        print ("-qm_mask         QM region            : ",arg_qm_mask)
        print ("-qm_charges      QM charges           : ",arg_qm_charges)
        print ("-qm_dict         QM dict              : ",arg_qm_dict) 
    if arg_vector_out != "False":
        print ("-vector_out      Vector output file   : ",arg_vector_out)
    return arg_nc,arg_parm,arg_out,arg_energy_out,arg_target,arg_solvent,arg_exclude_atoms,arg_TIP4P,arg_verbose,      \
           arg_use_qmcharges,arg_qm_mask,arg_qm_charges,arg_qm_dict,arg_vector_out                    ### Add new flag here

def python_main(args):                                                                           
    arg_nc,arg_parm,arg_out,arg_energy_out,arg_target,arg_solvent,arg_exclude_atoms,arg_TIP4P,arg_verbose,       \
    arg_use_qmcharges,arg_qm_mask,arg_qm_charges,arg_qm_dict,arg_vector_out=inputParser(args)     ### Add new flag here
    arg_solvent = arg_solvent.split(",")

    print ("\nField calculation RUNNING : ",datetime.datetime.now()) 

    #Load qm_mask.in
    if arg_use_qmcharges == "True":
        arg_qm_mask = load_qm_mask(arg_qm_mask)

    ### Empty dictionary storing all Fields
    FIELDS = {}
    ENERGIES = {}
    VECTORS = {}
    
######################## not necessary anymore! Keept in case this might become interesting again
    ### Make pdb of system
    #arg_pdb = ".".join(arg_out.split(".")[:-1])+"_full.pdb"
    #make_pdb(arg_nc,arg_param,arg_pdb,arg_use_qmcharges,arg_TIP4P,arg_qm_mask,arg_out)
######################## not necessary anymore!

    ### Load QM charges
    if arg_use_qmcharges == "True":
        QM_Charges,QM_Dict = fkt_Load_QM_Charges(arg_qm_charges,arg_qm_dict) 
        if arg_verbose=="True": print ("QM_Dict :", QM_Dict,"\n" )     
        if arg_verbose=="True": print ("QM_Charges :", QM_Charges[:50],"\n")

    ### Load Trajectory
    Trajectory, Charges, Names = fkt_Load_Trajectory(arg_nc,arg_parm,arg_TIP4P)
    if arg_verbose=="True": print ("Trajectory :", Trajectory[0])
    if arg_verbose=="True": print ("Charges :", Charges)
    if arg_verbose=="True": print ("Names :", Names)
    if len(Trajectory[0]) != len(Charges) or len(Trajectory[0]) != len(Names):
        print ("Error! Trajectory, Charges, and Atom Name Lists have different length")
        print ("Trajectory :", len(Trajectory[0]))
        print ("Charges :", len(Charges))
        print ("Names :", len(Names),"\n","\n")
        usage()
        quit()
    
    if arg_verbose=="True": print ("Topology information (index,Coords,Charges,Names):")
    for i in range(0,len(Names),1):
        if arg_verbose=="True": print (i, Trajectory[0][i], Charges[i], Names[i] )

    ### Load Index of Target Atoms for Field calculation
    Target_Index,FIELDS = fkt_Target_Index(Names,arg_target,FIELDS)
    if arg_verbose=="True": print ("Target_Index:", Target_Index,"\n")
    if arg_verbose=="True": print ("Field_Dict:", FIELDS,"\n")

    ### Load Index of Excluded Atoms
    if arg_exclude_atoms != "False":
        arg_exclude_atoms = fkt_Exlude_Index(Names,arg_exclude_atoms,arg_target)
        if arg_verbose=="True": print ("Excluded_Atoms:", arg_exclude_atoms,"\n")
            
    ### Define Field Components (Total,Protein,Solvent,Each residue,Each solvent type) to calculate
    FIELDS,Field_Components = fkt_Field_Components(FIELDS,Names,arg_solvent)
    if arg_verbose=="True": print ("Field_Dict:", FIELDS,"\n")
    if arg_energy_out != "False":
        for FIELD in FIELDS.keys():
            ENERGIES[FIELD] = {}
            for Field_Component in FIELDS[FIELD].keys():
                ENERGIES[FIELD][Field_Component] = []
        if arg_verbose=="True": print ("Energy_Dict:", ENERGIES,"\n") 
    if arg_vector_out != "False":
        for FIELD in FIELDS.keys():
            VECTORS[FIELD] = []
           
        if arg_verbose=="True": print ("Vector_Dict:", VECTORS,"\n") 

    ### Loop through Trajectory to calculate Fields
    for Frame_i in range(0,len(Trajectory),1):
        
    ### Get XYZ coordinates of the Target Atoms
        Target_XYZ=fkt_get_Target_XYZ(Trajectory[Frame_i],Target_Index)
        if arg_verbose=="True": print ("Target_XYZ:",Target_XYZ,"\n")
        Target_VEC=fkt_get_Target_VEC(Target_XYZ)
        if arg_verbose=="True": print ("Target_VEC:",Target_VEC,"\n")    
    ### Update QM_Charges
        if arg_use_qmcharges == "True":
            Charges = fkt_Update_Charges(QM_Charges,QM_Dict,Frame_i,Names,Charges)  
            if arg_verbose=="True": print ("Charges :", Charges,"\n")    
            
    ### Loop through FIELDS to calculate each Target
        for FIELD in FIELDS:
            if len(FIELD.split("_")) == 1:   #POINT CALCULATION
                XYZ = Target_XYZ[FIELD][0]
                VEC = ""
            if len(FIELD.split("_")) == 2:   #VECTOR CALCULATION
                XYZ = Target_VEC[FIELD]['Center']
                VEC = Target_VEC[FIELD]['Vector']
            Self_i = FIELD.split("@")[0][1:]
            
    ### Calculate Fields - This is where the magic happens!
            FIELDS,ENERGIES,VECTORS = fkt_calc_fields(FIELDS,FIELD,XYZ,VEC,Trajectory[Frame_i],Charges,Names, \
                                              Field_Components,arg_solvent,Self_i,arg_exclude_atoms,arg_energy_out, \
                                              Target_Index,ENERGIES, arg_vector_out, VECTORS)
            if arg_verbose=="True": 
                for Field_Component in FIELDS[FIELD]:
                    print (FIELD,Field_Component,FIELDS[FIELD][Field_Component]) 
                if arg_energy_out != "False":    
                    for Field_Component in ENERGIES[FIELD]:
                        print (FIELD,Field_Component,ENERGIES[FIELD][Field_Component])    
    
    pickle.dump(FIELDS,open(arg_out,"wb"))   
    if arg_energy_out != "False":
        pickle.dump(ENERGIES,open(arg_energy_out,"wb"))
    if arg_vector_out != "False":
        pickle.dump(VECTORS,open(arg_vector_out,"wb"))
    print ("\nField calculation DONE : ",datetime.datetime.now() )      
    
def main():
    ########################
    ### THE FINAL SCRIPT ###
    ########################
    ### Read in arguments                                                                             
    arg_nc,arg_param,arg_out,arg_energy_out,arg_target,arg_solvent,arg_exclude_atoms,arg_TIP4P,arg_verbose,       \
    arg_use_qmcharges,arg_qm_mask,arg_qm_charges,arg_qm_dict,arg_vector_out=inputParser(sys.argv)     ### Add new flag here
    arg_solvent = arg_solvent.split(",")

    print ("\nField calculation RUNNING : ",datetime.datetime.now()) 

    #Load qm_mask.in
    if arg_use_qmcharges == "True":
        arg_qm_mask = load_qm_mask(arg_qm_mask)

    ### Empty dictionary storing all Fields
    FIELDS = {}
    ENERGIES = {}
    VECTORS = {}
    
######################## not necessary anymore! Keept in case this might become interesting again
    ### Make pdb of system
    #arg_pdb = ".".join(arg_out.split(".")[:-1])+"_full.pdb"
    #make_pdb(arg_nc,arg_param,arg_pdb,arg_use_qmcharges,arg_TIP4P,arg_qm_mask,arg_out)
######################## not necessary anymore!

    ### Load QM charges
    if arg_use_qmcharges == "True":
        QM_Charges,QM_Dict = fkt_Load_QM_Charges(arg_qm_charges,arg_qm_dict) 
        if arg_verbose=="True": print ("QM_Dict :", QM_Dict,"\n" )     
        if arg_verbose=="True": print ("QM_Charges :", QM_Charges[:50],"\n")

    ### Load Trajectory
    Trajectory, Charges, Names = fkt_Load_Trajectory(arg_nc,arg_param,arg_TIP4P)
    if arg_verbose=="True": print ("Trajectory :", Trajectory[0])
    if arg_verbose=="True": print ("Charges :", Charges)
    if arg_verbose=="True": print ("Names :", Names)
    if len(Trajectory[0]) != len(Charges) or len(Trajectory[0]) != len(Names):
        print ("Error! Trajectory, Charges, and Atom Name Lists have different length")
        print ("Trajectory :", len(Trajectory[0]))
        print ("Charges :", len(Charges))
        print ("Names :", len(Names),"\n","\n")
        usage()
        quit()
    
    if arg_verbose=="True": print ("Topology information (index,Coords,Charges,Names):")
    for i in range(0,len(Names),1):
        if arg_verbose=="True": print (i, Trajectory[0][i], Charges[i], Names[i] )

    ### Load Index of Target Atoms for Field calculation
    Target_Index,FIELDS = fkt_Target_Index(Names,arg_target,FIELDS)
    if arg_verbose=="True": print ("Target_Index:", Target_Index,"\n")
    if arg_verbose=="True": print ("Field_Dict:", FIELDS,"\n")

    ### Load Index of Excluded Atoms
    if arg_exclude_atoms != "False":
        arg_exclude_atoms = fkt_Exlude_Index(Names,arg_exclude_atoms,arg_target)
        if arg_verbose=="True": print ("Excluded_Atoms:", arg_exclude_atoms,"\n")
            
    ### Define Field Components (Total,Protein,Solvent,Each residue,Each solvent type) to calculate
    FIELDS,Field_Components = fkt_Field_Components(FIELDS,Names,arg_solvent)
    if arg_verbose=="True": print ("Field_Dict:", FIELDS,"\n")
    if arg_energy_out != "False":
        for FIELD in FIELDS.keys():
            ENERGIES[FIELD] = {}
            for Field_Component in FIELDS[FIELD].keys():
                ENERGIES[FIELD][Field_Component] = []
        if arg_verbose=="True": print ("Energy_Dict:", ENERGIES,"\n") 
    if arg_vector_out != "False":
        for FIELD in FIELDS.keys():
            VECTORS[FIELD] = []
           
        if arg_verbose=="True": print ("Vector_Dict:", VECTORS,"\n") 

    ### Loop through Trajectory to calculate Fields
    for Frame_i in range(0,len(Trajectory),1):
        
    ### Get XYZ coordinates of the Target Atoms
        Target_XYZ=fkt_get_Target_XYZ(Trajectory[Frame_i],Target_Index)
        if arg_verbose=="True": print ("Target_XYZ:",Target_XYZ,"\n")
        Target_VEC=fkt_get_Target_VEC(Target_XYZ)
        if arg_verbose=="True": print ("Target_VEC:",Target_VEC,"\n")    
    ### Update QM_Charges
        if arg_use_qmcharges == "True":
            Charges = fkt_Update_Charges(QM_Charges,QM_Dict,Frame_i,Names,Charges)  
            if arg_verbose=="True": print ("Charges :", Charges,"\n")    
            
    ### Loop through FIELDS to calculate each Target
        for FIELD in FIELDS:
            if len(FIELD.split("_")) == 1:   #POINT CALCULATION
                XYZ = Target_XYZ[FIELD][0]
                VEC = ""
            if len(FIELD.split("_")) == 2:   #VECTOR CALCULATION
                XYZ = Target_VEC[FIELD]['Center']
                VEC = Target_VEC[FIELD]['Vector']
            Self_i = FIELD.split("@")[0][1:]
            
    ### Calculate Fields - This is where the magic happens!
            FIELDS,ENERGIES,VECTORS = fkt_calc_fields(FIELDS,FIELD,XYZ,VEC,Trajectory[Frame_i],Charges,Names, \
                                              Field_Components,arg_solvent,Self_i,arg_exclude_atoms,arg_energy_out, \
                                              Target_Index,ENERGIES, arg_vector_out, VECTORS)
            if arg_verbose=="True": 
                for Field_Component in FIELDS[FIELD]:
                    print (FIELD,Field_Component,FIELDS[FIELD][Field_Component]) 
                if arg_energy_out != "False":    
                    for Field_Component in ENERGIES[FIELD]:
                        print (FIELD,Field_Component,ENERGIES[FIELD][Field_Component])    
    
    pickle.dump(FIELDS,open(arg_out,"wb"))   
    if arg_energy_out != "False":
        pickle.dump(ENERGIES,open(arg_energy_out,"wb"))
    if arg_vector_out != "False":
        pickle.dump(VECTORS,open(arg_vector_out,"wb"))
    print ("\nField calculation DONE : ",datetime.datetime.now() )              
            
if __name__ == "__main__":
    main()