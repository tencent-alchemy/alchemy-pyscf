#!/usr/bin/env python

import os, sys
#sys.path.insert(0, os.path.abspath(__file__ + '/..'))

import numpy as np
from pyscf import gto, dft, nist
# Activate property driver
from pyscf import __all__
from server import public

@public
def tddft(xyz: str, xc: str = 'camb3lyp', basis: str = '6-31+G*', nstates: int = 5):
    '''
    Compute the TDDFT energy of excited states and the transition dipole of
    each excited states.

    Args:
        xc : DFT functional
        nstates : Number of excited states
    '''
    mol = gto.M(atom=xyz, basis=basis, verbose=4)

    mf = mol.RKS().density_fit()
    if 'cam' in xc:
        mf._numint.libxc = dft.xcfun
    mf.xc = xc
    mf.conv_tol = 1e-7
    mf.kernel()
    td = mf.TDA().set(nstates=nstates, conv_tol=1e-4)
    de = td.kernel()[0] * nist.HARTREE2EV

    results = {}
    results['xc'] = xc
    results['basis'] = basis
    results['xyz'] = xyz
    results['unit'] = 'Angstrom'
    results['tddft'] = de

    results['oscillator_strength'] = td.oscillator_strength()
    results['transition_dipole'] = td.transition_dipole()
    results['transition_velocity_dipole'] = td.transition_velocity_dipole()
    results['transition_magnetic_dipole'] = td.transition_magnetic_dipole()

    dip = mf.dip_moment()
    results['dipole_vector'] = dip
    results['mulliken_charge'] = mf.mulliken_pop(verbose=0)[1]
    results['meta_mulliken_charge'] = mf.pop(verbose=0)[1]

    dm = mf.make_rdm1(ao_repr=True)
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    charge_center = np.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_origin(charge_center):
        if getattr(dm, 'ndim', None) == 2:
            results['r2'] = np.einsum('ij,ij->', dm, mol.intor('int1e_r2'))
        else:
            results['r2'] = np.einsum('sij,ij->', dm, mol.intor('int1e_r2'))

    mo_occ = np.asarray(mf.mo_occ)
    mo_energy = np.asarray(mf.mo_energy)
    results['HOMO'] = mo_energy[mo_occ > 0].max()
    results['LUMO'] = mo_energy[mo_occ == 0].min()
    return convert_to_nparray(results)


def convert_to_nparray(dat):
    if isinstance(dat, (tuple, list)):
        return [convert_to_nparray(x) for x in dat]
    elif isinstance(dat, set):
        return set([convert_to_nparray(x) for x in dat])
    elif isinstance(dat, dict):
        return dict([(k, convert_to_nparray(v)) for k, v in dat.items()])
    elif isinstance(dat, np.ndarray):
        return dat.tolist()
    else:
        return dat
