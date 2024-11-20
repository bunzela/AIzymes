### RosettaDesign ###
   
# Run RosettaDesign
/home/bunzelh/rosetta_src_2021.16.61629_bundle/main/source//bin/rosetta_scripts.linuxgccrelease \
    -s                                        /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/7vuu_RosettaRelax_4.pdb \
    -in:file:native                           /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/4/7vuu_RosettaRelax_4.pdb \
    -run:preserve_header                      true \
    -extra_res_fa                             /home/bunzelh/AIzymes/design/TEST/Input/5TS.params \
    -enzdes:cstfile                           /home/bunzelh/AIzymes/design/TEST/Input/5TS_enzdes_planar_tAB100.cst \
    -enzdes:cst_opt                           true \
    -parser:protocol                          /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/scripts/RosettaDesign_7.xml \
    -out:file:scorefile                       /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/score_RosettaDesign.sc \
    -nstruct                                  1  \
    -ignore_zero_occupancy                    false  \
    -corrections::beta_nov16                  true \
    -overwrite 

# Cleanup
mv /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/7vuu_RosettaRelax_4_0001.pdb \
   /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/7vuu_RosettaDesign_7.pdb 
   
# Get sequence
python /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/parent/extract_sequence_from_pdb.py \
    --pdb_in       /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/7vuu_RosettaDesign_7.pdb \
    --sequence_out /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/7vuu_7.seq


### ElectricFields ###

# Delete everything after 'CONECT' and remove hydrogens 
sed -n '/CONECT/q;p' /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/7vuu_RosettaDesign_7.pdb > \
                     /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/ElectricFields/7vuu_RosettaDesign_7.pdb
sed -i '/ H /d'      /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/ElectricFields/7vuu_RosettaDesign_7.pdb

# Make AMBER files
tleap -f             /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/ElectricFields/7vuu_RosettaDesign_7_tleap.in > \
                     /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/ElectricFields/7vuu_RosettaDesign_7_tleap.out
mv leap.log          /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/ElectricFields/7vuu_RosettaDesign_7_tleap.log

# Add field calculation command
python   /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/../../../src/FieldTools.py \
-nc      /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/ElectricFields/7vuu_RosettaDesign_7.rst7 \
-parm    /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/ElectricFields/7vuu_RosettaDesign_7.parm7 \
-out     /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/7/ElectricFields/7vuu_RosettaDesign_7_fields.pkl \
-target  /home/bunzelh/AIzymes/design/TEST/TEST_ROSETTA/parent/field_target.dat \
-solvent WAT


