#!/usr/bin/env python
# encoding: utf-8
# File Name: utils.py
# Author: Jiezhong Qiu
# Create Time: 2019/01/26 15:40
# TODO:

import numpy as np
import pybel

def smi2xyz(smi, forcefield="mmff94", steps=50):
    """
    Example:
        utils.smi2xyz("CNC(C(C)(C)F)C(C)(F)F")
        returns:
C          1.17813        0.06150       -0.07575
N          0.63662        0.20405        1.27030
C         -0.86241        0.13667        1.33270
C         -1.46928       -1.21234        0.80597
C         -0.94997       -2.44123        1.55282
C         -2.99527       -1.22252        0.74860
F         -1.08861       -1.36389       -0.50896
C         -1.34380        0.44926        2.78365
C         -0.84421        1.76433        3.34474
F         -2.70109        0.48371        2.84063
F         -0.94986       -0.53971        3.63106
H          0.78344        0.82865       -0.74701
H          0.99920       -0.92873       -0.50038
H          2.26559        0.18049       -0.03746
H          1.03185       -0.51750        1.87094
H         -1.24335        0.93908        0.68721
H         -1.29943       -2.47273        2.58759
H         -1.27996       -3.36049        1.05992
H          0.14418       -2.47324        1.55471
H         -3.35862       -0.36599        0.16994
H         -3.34471       -2.11983        0.22567
H         -3.46364       -1.21709        1.73400
H         -1.20223        2.60547        2.74528
H         -1.22978        1.89248        4.36213
H          0.24662        1.79173        3.40731
    """
    mol = pybel.readstring("smi", smi)
    mol.addh() # add hydrogens, if this function is not called, pybel will output xyz string with no hydrogens.
    mol.make3D(forcefield=forcefield, steps=steps) 
    # possible forcefields: ['uff', 'mmff94', 'ghemical']
    mol.localopt()
    return _to_pyscf_atom(mol)

def _to_pyscf_atom(mol):
    atoms = []
    for atom in mol.atoms:
        atoms.append([atom.atomicnum, atom.coords])
    return _remove_dup_atoms(atoms)

def _remove_dup_atoms(atoms):
    xyz = np.array([a[1] for a in atoms]).round(3)
    c = xyz[:,0] * 10000 + xyz[:,1] * 100 + xyz[:,2]
    idx = np.unique(c, return_index=True)[1]
    return [atoms[i] for i in idx]
