### RosettaDesign ###
   
# Run RosettaDesign
/scratch-scc/projects/scc_mmtm_bunzel/rosetta.source.release-371/main/source//bin/rosetta_scripts.linuxgccrelease \
    -s                                        /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/parent/7vuu.pdb \
    -in:file:native                           /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/parent/7vuu.pdb \
    -run:preserve_header                      true \
    -extra_res_fa                             /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/Input/5TS.params \
    -enzdes:cstfile                           /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/Input/5TS_enzdes_planar_tAB100.cst \
    -enzdes:cst_opt                           true \
    -parser:protocol                          /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/scripts/RosettaDesign_0.xml \
    -out:file:scorefile                       /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/score_RosettaDesign.sc \
    -nstruct                                  1  \
    -ignore_zero_occupancy                    false  \
    -corrections::beta_nov16                  true \
    -overwrite 

# Cleanup
mv /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/7vuu_0001.pdb \
   /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/7vuu_RosettaDesign_0.pdb 
   
# Get sequence
python /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/parent/extract_sequence_from_pdb.py \
    --pdb_in       /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/7vuu_RosettaDesign_0.pdb \
    --sequence_out /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/7vuu_0.seq


### ElectricFields ###

# Delete everything after 'CONECT' and remove hydrogens 
sed -n '/CONECT/q;p' /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/7vuu_RosettaDesign_0.pdb > \
                     /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaDesign_0.pdb
sed -i '/ H /d'      /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaDesign_0.pdb

# Make AMBER files
tleap -f             /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaDesign_0_tleap.in > \
                     /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaDesign_0_tleap.out
mv leap.log          /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaDesign_0_tleap.log

# Add field calculation command
python   /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/../../../src/FieldTools.py \
-nc      /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaDesign_0.rst7 \
-parm    /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaDesign_0.parm7 \
-out     /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/0/ElectricFields/7vuu_RosettaDesign_0_fields.pkl \
-target  /mnt/vast-standard/home/hansadrian.bunzel01/u14852/AIzymes/design/TEST/TEST_SCC/parent/field_target.dat \
-solvent WAT


