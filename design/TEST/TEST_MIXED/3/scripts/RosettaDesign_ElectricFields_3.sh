### RosettaDesign ###
   
# Run RosettaDesign
/home/bunzelh/rosetta_src_2021.16.61629_bundle/main/source//bin/rosetta_scripts.linuxgccrelease \
    -s                                        /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/parent/7vuu.pdb \
    -in:file:native                           /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/parent/7vuu.pdb \
    -run:preserve_header                      true \
    -extra_res_fa                             /home/bunzelh/AIzymes/design/TEST/Input/5TS.params \
    -enzdes:cstfile                           /home/bunzelh/AIzymes/design/TEST/Input/5TS_enzdes_planar_tAB100.cst \
    -enzdes:cst_opt                           true \
    -parser:protocol                          /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/3/scripts/RosettaDesign_3.xml \
    -out:file:scorefile                       /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/3/score_RosettaDesign.sc \
    -nstruct                                  1  \
    -ignore_zero_occupancy                    false  \
    -corrections::beta_nov16                  true \
    -overwrite 

# Cleanup
mv /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/3/7vuu_0001.pdb \
   /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/3/7vuu_RosettaDesign_3.pdb 
   
# Get sequence
python /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/parent/extract_sequence_from_pdb.py \
    --pdb_in       /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/3/7vuu_RosettaDesign_3.pdb \
    --sequence_out /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/3/7vuu_3.seq


### ElectricFields ###

# Delete everything after 'CONECT' and remove hydrogens 
sed -n '/CONECT/q;p' /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/3/7vuu_RosettaDesign_3.pdb > \
                     /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/3/ElectricFields/7vuu_RosettaDesign_3.pdb
sed -i '/ H /d'      /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/3/ElectricFields/7vuu_RosettaDesign_3.pdb

# Make AMBER files
tleap -f             /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/3/ElectricFields/7vuu_RosettaDesign_3_tleap.in > \
                     /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/3/ElectricFields/7vuu_RosettaDesign_3_tleap.out
mv leap.log          /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/3/ElectricFields/7vuu_RosettaDesign_3_tleap.log

# Add field calculation command
python   /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/../../../src/FieldTools.py \
-nc      /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/3/ElectricFields/7vuu_RosettaDesign_3.rst7 \
-parm    /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/3/ElectricFields/7vuu_RosettaDesign_3.parm7 \
-out     /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/3/ElectricFields/7vuu_RosettaDesign_3_fields.pkl \
-target  /home/bunzelh/AIzymes/design/TEST/TEST_MIXED/parent/field_target.dat \
-solvent WAT


