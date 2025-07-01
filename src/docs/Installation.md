# Installation

We recommend installing AI.zymes with pip. For code development, AI.zymes can also be cloned from the GitHub repository. To install AI.zymes with pip, run:

```
pip install aizymes
```

AI.zymes works by wrapping around various bioengineering tools that need to be installed seperately. A list of all established tools in AI.zymes is given in the section **Available Tools**. For some of these tools, their paths need to be defined to connect with them. To that end, AI.zymes will create a **.config** file when you run AIzymes for first time. 

> [!Note] 
> Several tools, such as Rosetta, need to be installed seperately. See **Available Tools** for guidelines on how to find the installation instructions.

> [!WARNING] 
> AI.zymes currently only works with SLURM.
> If you want to use a different submission system, please contact Adrian.Bunzel@mpi-marburg.mpg.de.

## Installation via GitHub

To install AI.zymes via GitHub, clone the repository, create the AI.zymes python environment, and add the repository to your pythonpath. To clone AI.zyme, run:

```
git clone https://github.com/bunzela/AIzymes.git
```

You can either create your own AI.zymes environment, or install all required packages in your existing environment.

```
cd AIzymes
# To build new environemnt
conda env create -f environment.yml --name AIzymes 
# Alternative to install packages in curent environemnt:
# conda env update -f environment.yml --prune
```

Add the path to AI.zymes to your .bashrc. Replace **PATH_TO_AIZYMES** with the corect path to AI.zymes.

```
echo 'export PYTHONPATH="$PYTHONPATH:PATH_TO_AIZYMES/src/aizymes"' >> ~/.bashrc
source ~/.bashrc
```
