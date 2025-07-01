# Quick Start

> [!TIP] 
> Get started with AI.zymes using the notebook **example_notebooks/Example_KSI.ipynb**.

The notebook **Example_KSI.ipynb** can be used to improve the promoscious Kemnp eliminase activity of KSI. Here, each step of this notebook is explained briefly. For detailed instructions on how to use AI.zymes, check **Full Manual**.

## Create an Instance of the AI.zymes class

To start AI.zymes, you first need to create an Instance of the AI.zymes class and set **FOLDER_HOME**.
All data generated during the AI.zymes run are storred in **FOLDER_HOME**.
Asides from the design structures, **FOLDER_HOME** contains information on the settings chosen during design (**variables.json**)
and a database containing the sequences and scores of all designs (**all_scores.csv**).

```
# Import AI.zymes
from aizymes import * 
         
# Create instance of AIzymes_MAIN
AIzymes = AIzymes_MAIN(FOLDER_HOME = 'Example_KSI') 
```

## Set up AI.zymes for design

Befor starting design, the basic folder structure and input files need to be set up with **AIzymes.setup**. The input settings used here are discussed in detail in **Full Manual**.

> [!IMPORTANT] 
> All inputfiles provided need to be in the folder **Input** in the same directory as where you running the code.
> Note that structures can also be loaded from a different folder using **FOLDER_PAR_STRUC**.

> [!NOTE] 
> If the config file specified with **SYSTEM** does not exist,
> **AIzymes.setup** will prompt for inputs to generate a new config file.
> The config file contains system-specific information neccesary to run AI.zymes.

> [!WARNING] 
> AI.zymes currently only works with SLURM.
> If you want to use a different submission system, please contact Adrian.Bunzel@mpi-marburg.mpg.de.

```
AIzymes.setup(
# General Design Settings
    WT                = "1ohp",
    LIGAND            = "5TS",
    DESIGN            = "11,14,15,18,38,54,55,58,63,65,80,82,84,97,99,101,112,114,116",
    PARENT_DES_MED    = ['RosettaDesign',
                         'ESMfold','MDMin','RosettaRelax','ElectricFields'],
    DESIGN_METHODS    = [[0.5,'SolubleMPNN',\
                              'ESMfold','MDMin','RosettaRelax','ElectricFields'],\
                         [0.5,'RosettaDesign',\
                              'ESMfold','MDMin','RosettaRelax','ElectricFields']],
# General Scoring Settings    
    SELECTED_SCORES   = ["total","catalytic","interface","efield", "identical"],    
# General Job Settings
    MAX_JOBS          = 72,
    MAX_GPUS          = 4,
    MEMORY            = 450,
    N_PARENT_JOBS     = 144,
    MAX_DESIGNS       = 5000,
    KBT_BOLTZMANN     = [1.0, 0.5, 0.02],   
    SUBMIT_PREFIX     = "KSI_TEST", 
    SYSTEM            = "AIzymes.config",
# RosettaDesign Settings
    CST_NAME          = "5TS_enzdes_planar_tAB100", 
    CST_DIST_CUTOFF   = 40.,
# FieldTools settings
    FIELD_TARGET      = ":5TS@C9 :5TS@H04",
 )                     
```

## Submit controller to run AI.zymes

Once the setup is completet you need to start the controller.
**AIzymes.submit_controller()** will launch am HPC job that runs the controller.

```
AIzymes.submit_controller()
```

> [!TIP]  
> You can check the progress of AI.zymes and trouble-shoot errors with the controller.log.
> Alternatively, you can also check the output and errors of individual designs.

```
!tail -n 50 {FOLDER_HOME}/controller.log
!ls {FOLDER_HOME}/designs/0/scripts
```

## Analysis tools to check AI.zymes

AI.zymes comes with several tools to analyze the design runs. Use **AIzymes.plot** to create a range of plots, showing how the scores evolved over time.
Alternatively, you can use **AIzymes.print_statistics()** to print out key information from the **all_scores.csv** file.


> [!NOTE] 
> **NORM** is used to ensure comarability between design runs by normalizing all results to the same values.

```
AIzymes.plot(NORM = {
                'total_score': [310, 380], 
                'catalytic_score': [-30, -4],
                'interface_score': [20, 30],
                'final_score': [-30, 0],
                'efield_score': [-40, 30],
                'identical_score': [0, 1.05],
             })                   
```

> [!TIP] 
> **AIzymes.print_statistics()** can be very helpfull to trouble-shoot!

```
AIzymes.print_statistics()
```

## Extract best structures after AI.zymes run

Finally, to extract the hits from your AI.zymes run, execute **AIzymes.best_structures()**

> [!NOTE] 
> Structures will be saved with their final score as prefix, helping to identify the best designs.

```
AIzymes.best_structures()      
```
