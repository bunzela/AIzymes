# AI.zymes

> [!NOTE]
> We are happy to tailor **AI.zymes** to your system! Contact [Adrian Bunzel](mailto:Adrian.Bunzel@mpi-marburg.mpg.de) for specific requests!

> [!NOTE]
> The **AIzymes_Manual.pdf** contains all information to get started. The manual is still work in progess. Feel free to reach out if you have any specific questions.

Welcome to the code repository for **AI.zymes — a modular platform for evolutionary enzyme design**.

**AI.zymes** integrates a suite of state-of-the-art tools for enzyme engineering, including:
- 🛠️ **Protein design** (e.g. RosettaDesign, ProteinMPNN, LigandMPNN)  
- 🔮 **Structure prediction** (e.g. ESMFold, RosettaRelax, MD minimization)
- ⚡ **Electrostatic Catalysis** (e.g. FieldTools)

Built with modularity in mind, **AI.zymes** allows you to easily plug in new methods or customize workflows for diverse bioengineering goals — from enzyme evolution to structure-function exploration.

We are currently working on improving the accessibility of **AI.zymes**, including a full user manual and installation instructions. Stay tuned!

## 📥 Getting Started

**AIzymes_Manual.pdf** contains all information to get started with **AI.zymes**. We are actively looking for collaborators and enthusiastic users! If you're interested in using **AI.zymes** or exploring joint projects, **please reach out** — we'd love to hear from you:

**Contact:**  
📧 [Adrian Bunzel](mailto:Adrian.Bunzel@mpi-marburg.mpg.de)  
Max Planck Institute for Terrestrial Microbiology

## 📝 Citation

If you use **AI.zymes** in your research, please cite:

**AI.zymes – A Modular Platform for Evolutionary Enzyme Design**  

Lucas P. Merlicek, Jannik Neumann, Abbie Lear, Vivian Degiorgi, Moor M. de Waal, Tudor-Stefan Cotet, Adrian J. Mulholland, and H. Adrian Bunzel
**Angewandte Chemie International Edition** 2025, https://doi.org/10.1002/anie.202507031

## 🛠️ Installation

Check **AIzymes_Manual.pdf** for detailed installation instructions.

Briefly, we recommend installing **AI.zymes** with pip. 

```
pip install aizymes
```

To use **AI.zymes** in Python, import:

```
from aizymes import *
```

For code development, AI.zymes can also be cloned from the GitHub repository:

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

> [!NOTE]
> Replace $HOME/AIzymes/src with the actual path if you have cloned the repository elsewhere.

---

> *AI.zymes is in active development! Contributions, feedback, and collaborations are very welcome! We are happy to assist you with geting AI.zymes to run on your systems.*

---

> *License: MIT-NC – non-commercial academic use only. Commercial use requires permission. See LICENSE.txt.*
