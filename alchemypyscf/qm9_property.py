#!/usr/bin/env python
# Copyright 2019 Tencent America LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

#import os, sys
#sys.path.insert(0, os.path.abspath(__file__ + '/..'))

from typing import Iterator, Union, List
import time
import numpy as np

from pyscf import gto, scf, dft, nist
# Activate all pyscf features
from pyscf import __all__

from utils import smi2xyz
from . import pyscf_geomopt
from . import thermo
from . import utils


def qm9_properties(smi: str,
                   properties: Iterator[str] = None,
                   basis = {'default': '6-31g(2df,p)', 'Br': '6-311g*', 'I':'6-311g*'},
                   xc: str = 'b3lyp') -> dict:
    '''
    Compute the equilibrium geometry of the molecule represented by the given
    SMILES string and the molecular properties (including the 15 properties
    defined by the QM9 dataset). The molecular properties are

    ----------------------- --------------------------
    key                       comments
    ----------------------- --------------------------
    'polarizability'        : Isotropic polarizability (au)
    'polarizability_tensor' : (au)
    'dipole'                : Magnitude of dipole moment (Debye)
    'dipole_vector'         : (Debye)
    'mulliken_charge'       : Atomic charge based on Mulliken population analysis
    'meta_mulliken_charge'  : Atomic charge of meta Mulliken population analysis
                              (which has better transferability than Mulliken charge)
    'r2'                    : <r^2> (au)
    'HOMO'                  : orbital energy (au)
    'LUMO'                  : orbital energy (au)
    'rot_const'             : Rotational constants A, B, C (GHz)
    'ZPVE'                  : Zero point vibrational energy (au)
    'U0'                    : Internal energy (au) at 0K
    'U'                     : Internal energy (au) at 298.15K
    'H'                     : Enthalpy (au) at 298.15K
    'G'                     : Free energy (au) at 298.15K
    'Cv'                    : Heat capacity (au) at 298.15K
    'smi'                   : SMILES string
    'xyz'                   : Equilibrium geometry (angstrom)
    'Etot'                  : Total DFT energy (au)
    'force'                 : Atomic force (au) for the equilibrium geometry. Should be ~ 0
    ----------------------- --------------------------

    Args:
        smi (str): SMILES string

    Kwargs:
        properties (list): A list of properties to compute. If not
            specified, all properties listed above will be computed by
            default. Keys listed above can be used to specify the properties.

        basis (str or dict): basis set for quantum chemistry calculations. It
            supports various basis input format:
            * string for Pople basis like 3-21G, 6-311+G*, 6-31G(2d,p), ...
            * string for Dunning basis like cc-pvdz, aug-ccpcvdz, ...
            * string for more basis functions. The complete list of basis sets
              can be found in https://github.com/pyscf/pyscf/tree/master/pyscf/gto/basis
            * a dict to specify basis for specific elements such as
              {'C': '6-31g', 'H': 'sto-3g'}
            * see also PySCF document for other supported basis input format
              https://github.com/pyscf/pyscf/blob/master/examples/gto/04-input_basis.py

        xc (str): DFT XC functional. It can be any functionals provided by
            libxc library, like Slater, pbe, pbe0, b3lyp, bp86, camb3lyp,
            tpss,... It also supports custom functional '0.5*PBE0+0.5*B3LYP'.
            see also PySCF library for the complete lists of XC functionals
            https://github.com/pyscf/pyscf/blob/master/pyscf/dft/libxc.py

    Returns:
        A property dict for the properties specified in the argument
        "properties". The dict keys are the names listed above.
    '''

    print("smi: %s" % smi)
    atom_xyz = smi2xyz(smi)
    mol = gto.Mole()
    mol.atom = atom_xyz
    mol.spin = None  # guess spin
    mol.max_memory = _system_max_memory()
    mol.basis = basis
    mol.build()

    dft_time = time.time()
    mol, e_tot, grad = pyscf_geomopt.run_dft(mol, with_df=True, xc=xc)
    dft_time = time.time() - dft_time

    metadata = {}
    metadata['smi'] = smi
    metadata['xc'] = xc
    metadata['basis'] = basis
    metadata['xyz'] = mol.tostring('xyz')
    metadata['Etot'] = e_tot
    metadata['force'] = -grad

    property_time = time.time()
    metadata['properties'] = _qm9_property(mol, xc, properties)
    property_time = time.time() - property_time

    metadata['dft_time'] = dft_time
    metadata['property_time'] = property_time

    return convert_nparray(metadata)

def geomopt(smi: str,
            basis = {'default': '6-31g(2df,p)', 'Br': '6-311g*', 'I':'6-311g*'},
            xc: str = 'b3lyp') -> dict:
    '''
    Compute the equilibrium geometry of the molecule represented by the given
    SMILES string. It's exactly the first step of qm9_properties. The computed
    properties are

    'smi'   : SMILES string
    'xyz'   : Equilibrium geometry (angstrom)
    'Etot'  : Total DFT energy (au)
    'force' : Atomic force (au) for the equilibrium geometry. Should be ~ 0

    Args:
        smi (str): SMILES string

    Kwargs:
        basis (str or dict): basis set for quantum chemistry calculations. It
            supports various basis input format:
            * string for Pople basis like 3-21G, 6-311+G*, 6-31G(2d,p), ...
            * string for Dunning basis like cc-pvdz, aug-ccpcvdz, ...
            * string for more basis functions. The complete list of basis sets
              can be found in https://github.com/pyscf/pyscf/tree/master/pyscf/gto/basis
            * a dict to specify basis for specific elements such as
              {'C': '6-31g', 'H': 'sto-3g'}
            * see also PySCF document for other supported basis input format
              https://github.com/pyscf/pyscf/blob/master/examples/gto/04-input_basis.py

        xc (str): DFT XC functional. It can be any functionals provided by
            libxc library, like Slater, pbe, pbe0, b3lyp, bp86, camb3lyp,
            tpss,... It also supports custom functional '0.5*PBE0+0.5*B3LYP'.
            see also PySCF library for the complete lists of XC functionals
            https://github.com/pyscf/pyscf/blob/master/pyscf/dft/libxc.py

    Returns:
        A dict for the results including 'smi', 'xyz', 'Etot', 'force'
    '''
    print("smi: %s" % smi)
    atom_xyz = smi2xyz(smi)
    mol = gto.Mole()
    mol.atom = atom_xyz
    mol.spin = None  # guess spin
    mol.max_memory = _system_max_memory()
    mol.basis = basis
    mol.build()

    dft_time = time.time()
    mol, e_tot, grad = pyscf_geomopt.run_dft(mol, with_df=True)
    dft_time = time.time() - dft_time

    metadata = {}
    metadata['smi'] = smi
    metadata['xc'] = xc
    metadata['basis'] = basis
    metadata['xyz'] = mol.tostring('xyz')
    metadata['Etot'] = e_tot
    metadata['force'] = -grad
    return convert_nparray(metadata)


def tddft(geom: str,
          nstates: int = 5,
          basis = {'default': '6-31g(2df,p)', 'Br': '6-311g*', 'I':'6-311g*'},
          xc: str = 'b3lyp',
          density_fit: bool = False) -> dict:
    '''
    For the input molecular geometry, compute the TDDFT (TDA) excited states and
    properties of excited states including:

    -------------------------------- ------------------
    key                                comments
    -------------------------------- ------------------
    'tddft'                          : Energy of excited states (eV)
    'oscillator_strength'            : Oscillator strength for each state
    'transition_dipole'              : Transition dipole vector (au) for each state
    'transition_velocity_dipole'     : Transition velocity dipole vector (au) for each state
    'transition_magnetic_dipole'     : Transition magnetic dipole vector (au) for each state
    'transition_quadrupole'          : Transition quadrupole tensor (au) for each state
    'transition_velocity_quadrupole' : Transition velocity quadrupole tensor (au) for each state
    'transition_magnetic_quadrupole' : Transition magnetic quadrupole tensor (au) for each state
    'HOMO'                           : orbital energy (au)
    'LUMO'                           : orbital energy (au)
    'Etot'                           : Total DFT energy (au)
    -------------------------------- ------------------

    Args:
        geom (str): Molecular gemoetry. It can be Cartesian coordinates or
            Z-matrix input.

    Kwargs:
        nstates (int): Number of excited states to compute.

        basis (str or dict): basis set for quantum chemistry calculations. It
            supports various basis input format:
            * string for Pople basis like 3-21G, 6-311+G*, 6-31G(2d,p), ...
            * string for Dunning basis like cc-pvdz, aug-ccpcvdz, ...
            * string for more basis functions. The complete list of basis sets
              can be found in https://github.com/pyscf/pyscf/tree/master/pyscf/gto/basis
            * a dict to specify basis for specific elements such as
              {'C': '6-31g', 'H': 'sto-3g'}
            * see also PySCF document for other supported basis input format
              https://github.com/pyscf/pyscf/blob/master/examples/gto/04-input_basis.py

        xc (str): DFT XC functional. It can be any functionals provided by
            libxc library, like Slater, pbe, pbe0, b3lyp, bp86, camb3lyp,
            tpss,... It also supports custom functional '0.5*PBE0+0.5*B3LYP'.
            see also PySCF library for the complete lists of XC functionals
            https://github.com/pyscf/pyscf/blob/master/pyscf/dft/libxc.py

        density_fit (bool): whether or not to use density fitting to compute
            2-electron integrals

    Returns:
        placeholder
    '''
    mol = gto.M(atom=geom, basis=basis, spin=None)
    mol.max_memory = _system_max_memory()

    property_time = time.time()
    mf = mol.RKS()
    if density_fit:
        mf = mf.density_fit()
    if xc[:3].upper() == 'CAM':
        mf._numint.libxc = dft.xcfun

    mf.xc = xc
    mf.kernel()
    td = mf.TDA().set(nstates=nstates, conv_tol=1e-4)
    de = td.kernel()[0]
    de *= nist.HARTREE2EV

    results = {}
    results['geom'] = geom
    results['nstates'] = nstates
    results['xc'] = xc
    results['basis'] = basis
    results['tddft'] = de

    results['oscillator_strength'           ] = td.oscillator_strength()
    results['transition_dipole'             ] = td.transition_dipole()
    results['transition_velocity_dipole'    ] = td.transition_velocity_dipole()
    results['transition_magnetic_dipole'    ] = td.transition_magnetic_dipole()
    results['transition_quadrupole'         ] = td.transition_quadrupole()
    results['transition_velocity_quadrupole'] = td.transition_velocity_quadrupole()
    results['transition_magnetic_quadrupole'] = td.transition_magnetic_quadrupole()

    mo_occ = np.asarray(mf.mo_occ)
    mo_energy = np.asarray(mf.mo_energy)
    results['HOMO'] = mo_energy[mo_occ> 0].max()
    results['LUMO'] = mo_energy[mo_occ==0].min()
    results['Etot'] = mf.e_tot

    property_time = time.time() - property_time
    return convert_nparray(results)


def pyscf(method: str = 'DFT',
          geom: str = None,
          basis = 'sto-3g',
          charge: int = 0,
          spin: int = None,
          max_memory: float = 8000,
          xc: str = 'b3lyp',
          verbose: int = 0,
          properties: Iterator[str] = None,
          params: dict = None) -> dict:
    '''A simple driver to run single point calculations using PySCF library.

    Kwargs:
        method (str): A string of regular quantum chemistry method. It can be
            HF, DFT, KS, MP2, CCSD, CISD, CAS(n,m)

        geom (str): Molecular gemoetry. It can be Cartesian coordinates or
            Z-matrix input.

        basis (str or dict): basis set for quantum chemistry calculations. It
            supports various basis input format:
            * string for Pople basis like 3-21G, 6-311+G*, 6-31G(2d,p), ...
            * string for Dunning basis like cc-pvdz, aug-ccpcvdz, ...
            * string for more basis functions. The complete list of basis sets
              can be found in https://github.com/pyscf/pyscf/tree/master/pyscf/gto/basis
            * a dict to specify basis for specific elements such as
              {'C': '6-31g', 'H': 'sto-3g'}
            * see also PySCF document for other supported basis input format
              https://github.com/pyscf/pyscf/blob/master/examples/gto/04-input_basis.py

        charge (int): charge for ion or cation.

        spin (int): If not specified, spin will be guessed based on the
            neutral molecule

        xc (str): DFT XC functional. It can be any functionals provided by
            libxc library, like Slater, pbe, pbe0, b3lyp, bp86, camb3lyp,
            tpss,... It also supports custom functional '0.5*PBE0+0.5*B3LYP'.
            see also PySCF library for the complete lists of XC functionals
            https://github.com/pyscf/pyscf/blob/master/pyscf/dft/libxc.py

        verbose (int) : from 0 to 7, print level in the output.

        properties (str): A list of properties to be computed. The elements
            can be 'Mulliken population', 'density matrix', 'force', 'thermo'

        params (dict): Extra parameters of the method to be set. Depending on
            the method, it can be 'conv_tol, 'max_cycle', 'frozen', 'x2c',
            'density_fit', etc

    Returns:
        dict of results. If the argument properties is set, relevant
        properties will be computed and added to results dict.
    '''
    method = method.upper()
    mol = gto.Mole()
    mol.basis = basis
    mol.charge = charge
    mol.ecp = ecp
    mol.spin = spin
    mol.verbose = verbose
    mol.max_memory = _system_max_memory()
    mol.build()

    if method[:3] == 'CAS':
        method, cas = method.split('(')
        nelecas, ncas = cas.replace(' ', '').split(')')[0].split(',')
        model = getattr(mol, method)(int(ncas), int(nelecas))
    else:
        model = getattr(mol, method)

    if isinstance(model, dft.rks.KohnShamDFT)
        # set KS attributes
        model.xc = xc

    if params:
        for key, val in params.items():
            attr = getattr(model, key, None)
            if callable(attr):
                model = attr(val)
            elif attr:  # To ensure attributes were defined in the model
                setattr(model, key, val)

    model.kernel()

    results = {}
    results['Etot'] = model.e_tot

    if properties:
        # population analysis
        fp = set(x[:3].upper() for x in properties)

        dm = model.make_rdm1()
        if 'DEN' in fp:
            results['density matrix'] = dm

        if 'MUL' in fp:
            results['mulliken_charge'] = scf.hf.mulliken_pop(dm=dm)[1]
            results['meta_mulliken_charge'] = scf.hf.mulliken_meta.pop(dm=dm)[1]

        if 'FOR' in fp:
            grad = model.nuc_grad_method().kernel()
            results['force'] = -grad

        if 'THE' in fp and isinstance(model, scf.hf.SCF):
            results['thermo'] = _theomo_chem(mf)

    return convert_nparray(results)


def convert_nparray(dat):
    if isinstance(dat, (tuple, list)):
        return [convert_nparray(x) for x in dat]
    elif isinstance(dat, set):
        return set([convert_nparray(x) for x in dat])
    elif isinstance(dat, dict):
        return dict([(k, convert_nparray(v)) for k, v in dat.items()])
    elif isinstance(dat, np.ndarray):
        return dat.tolist()
    else:
        return dat


def _qm9_property(mol, xc='b3lyp', properties=None):
    '''Ground state properties and thermochemistry data'''

    mf = mol.apply('KS').run(xc=xc)

    run_all = False
    if properties:
        # Format properties
        fp = set(x[:3].upper() for x in properties)
    else:
        run_all = True

    results = {}
    if run_all or 'POL' in fp:
        pol = mf.Polarizability().polarizability()
        # isotropic polarizability
        results['polarizability'] = pol.trace() / 3
        results['polarizability_tensor'] = pol

    if run_all or 'DIP' in fp:
        dip = mf.dip_moment()
        results['dipole'        ] = np.linalg.norm(dip)
        results['dipole_vector' ] = dip

    if run_all or 'MUL' in fp or 'MET' in fp:
        results['mulliken_charge'] = mf.mulliken_pop()[1]
        results['meta_mulliken_charge'] = mf.pop()[1]

    # <r^2>
    if run_all or 'R2' in fp:
        dm = mf.make_rdm1(ao_repr=True)
        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        charge_center = np.einsum('i,ix->x', charges, coords) / charges.sum()
        with mol.with_common_origin(charge_center):
            if getattr(dm, 'ndim', None) == 2:
                results['r2'] = np.einsum('ij,ij->', dm, mol.intor('int1e_r2'))
            else:
                results['r2'] = np.einsum('sij,ij->', dm, mol.intor('int1e_r2'))

    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    results['HOMO'] = mo_energy[mo_occ> 0].max()
    results['LUMO'] = mo_energy[mo_occ==0].min()

# rotational constants A, B, C
# zero point vibrational energy
# U0 internal energy at 0K
# U internal energy at 298.15K
# enthalpy at 298.15K
# Free energy at 298.15K
# Heat capacity at 298.15K
    if run_all or fp.intersection(['ROT', 'ZPV', 'U0', 'U', 'H', 'G', 'CV']):
        results.update(_theomo_chem(mf))
    return results

def _theomo_chem(mf):
    hess = mf.Hessian().kernel()
    freq_info = thermo.harmonic_analysis(mf.mol, hess)
    thermo_info = thermo.thermo(mf, freq_info['freq_au'], 298.15, 101325)
    results = {}
    results['rot_const'] = thermo_info['rot_const'][0]
    results['ZPVE'     ] = thermo_info['ZPE'   ][0]
    results['U0'       ] = thermo_info['E_0K'  ][0]
    results['U'        ] = thermo_info['E_tot' ][0]
    results['H'        ] = thermo_info['H_tot' ][0]
    results['G'        ] = thermo_info['G_tot' ][0]
    results['Cv'       ] = thermo_info['Cv_tot'][0]
    return results

def _system_max_memory():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / 1024**2  # to MB
    except:
        return 4000
