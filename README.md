# AI.zymes

Welcome to the source code repository for **AI.zymes — a modular platform for evolutionary enzyme design**.

**AI.zymes** integrates a suite of state-of-the-art tools for enzyme engineering, including:
- 🛠️ **Protein design** (e.g. RosettaDesign, ProteinMPNN, LigandMPNN)  
- 🔮 **Structure prediction** (e.g. ESMFold, RosettaRelax, MD minimization)
- ⚡ **Electrostatic Catalysis** (e.g. FieldTools)

Built with modularity in mind, **AI.zymes** allows you to easily plug in new methods or customize workflows for diverse bioengineering goals — from enzyme evolution to structure-function exploration.

We are currently working on improving the accessibility of **AI.zymes**, including a full user manual and installation instructions. Stay tuned!

## 📥 Getting Started

We are actively looking for collaborators and enthusiastic users! If you're interested in using **AI.zymes** or exploring joint projects, **please reach out** — we'd love to hear from you:

**Contact:**  
📧 [Adrian Bunzel](mailto:Adrian.Bunzel@mpi-marburg.mpg.de)  
Max Planck Institute for Terrestrial Microbiology

## 📝 Citation

If you use **AI.zymes** in your research, please cite:

**AI.zymes – A Modular Platform for Evolutionary Enzyme Design**  
Lucas P. Merlicek, Jannik Neumann, Abbie Lear, Vivian Degiorgi, Moor M. de Waal, Tudor-Stefan Cotet, Adrian J. Mulholland, and H. Adrian Bunzel
**Angewandte Chemie International Edition**, 2025 *(accepted)*

---

## 🛠️ Developer Notes

**AI.zymes** is currently not distributed as a Python package (e.g. via `pip`). To use the platform in your own scripts or Jupyter notebooks, you need to ensure the source directory is included in your Python module search path.

Add the following line to your `.bashrc` (or `.bash_profile`) to permanently include the source directory:

```
export PYTHONPATH="$PYTHONPATH:$HOME/AIzymes/src"
```

This ensures that Python can locate modules like AIzymes_015.py when importing:

```
from AIzymes_015 import *
```

> [!NOTE]
> Replace $HOME/AIzymes/src with the actual path if you have cloned the repository elsewhere.

---

> *This project is in active development. Contributions, feedback, and collaborations are very welcome!*
