# alchemy-pyscf
Data generation program for Tencent alchemy project.

Tencent Alchemy project targets at quantum chemistry molecular properties using
all available deep learning tools. The molecular properties are computed with
quantum chemistry calculations at DFT level of theory. The tools developed in
alchemy-pyscf library supports various 2D and 3D molecular input format, such as
SMILES strings, z-matrix, xyz format and SDF format. Equilibrium molecular
geometry can be obtained. On top of the equilibrium geometry, the library allows
you to generate the 17 properties of QM9 dataset, the properties of excited
states, the properties of chiral molecule and so forth.

The code has been used to generate the data for [alchemy
contest](https://alchemy.tencent.com/), and our
[paper](https://rlgm.github.io/papers/31.pdf).

The source code of data generation will be released after the Alchemy contest.


Getting started
===============

Alchemy-pyscf depends on the Openbabel, pybel, pyberny, geomeTRIC, PySCF
libraries.  These packages can be installed through pip or conda.

An optimized version of alchemy-pyscf can be found at Tencent scientific
computation platform SimHub. Based on the open-source implementation in this
repo, parameters and code are tuned wrt the architecture of Tencent cloud hardware
in the optimized version. It can performs up to 2x faster than the open-source
implementation.
