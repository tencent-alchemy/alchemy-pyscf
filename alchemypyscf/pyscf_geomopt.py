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

import time
import copy
import numpy
import functools
from pyscf import lib
from pyscf import scf
from pyscf.df import df_jk
from pyscf.geomopt import berny_solver
from pyscf.geomopt import geometric_solver

from berny import berny
# Must be the in-house berny solver
assert(berny.__version__ == '0.2.1m')


# Patch SCF kernel. The new kernel can call the SOSCF solver automatically if
# the default DIIS scheme does not converge.
scf_kernel = scf.hf.SCF.kernel
def kernel(self, dm0=None, **kwargs):
    scf_kernel(self, dm0, **kwargs)
    if not self.converged:
        with lib.temporary_env(self, level_shift=.2):
            scf_kernel(self, dm0, **kwargs)
    if not self.converged:
        mf1 = self.newton().run()
        del mf1._scf  # mf1._scf leads to circular reference to self
        self.__dict__.update(mf1.__dict__)
    return self.e_tot
scf.hf.SCF.kernel = kernel

berny_params = {
        'gradientmax': 0.45e-3,  # Eh/Angstrom
        'gradientrms': 0.15e-3,  # Eh/Angstrom
        'stepmax': 1.8e-3,       # Angstrom
        'steprms': 1.2e-3,       # Angstrom
        'trust': 0.3,
        'dihedral': True,
        'superweakdih': False,
        'interpolation': False,
}
geomeTRIC_params = {  # They are default settings
        'convergence_energy': 1e-6,  # Eh
        'convergence_gmax': 4.5e-4 * lib.param.BOHR,  # Eh/Bohr
        'convergence_grms': 1.5e-4 * lib.param.BOHR,  # Eh/Bohr
        'convergence_dmax': 1.8e-3,  # Angstrom
        'convergence_drms': 1.2e-3,  # Angstrom
}

def nuc_grad_method(mf):
    from .grad import rhf, uhf, rks, uks
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
        if getattr(mf, 'xc', None):
            return uks.Gradients(mf)
        else:
            return uhf.Gradients(mf)
    else:
        if getattr(mf, 'xc', None):
            return rks.Gradients(mf)
        else:
            return rhf.Gradients(mf)

def Hessian(mf):
    from .hessian import rhf, uhf, rks, uks
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
        if getattr(mf, 'xc', None):
            return uks.Hessian(mf)
        else:
            return uhf.Hessian(mf)
    else:
        if getattr(mf, 'xc', None):
            return rks.Hessian(mf)
        else:
            return rhf.Hessian(mf)

# Monkey patch the df_jk.density_fit function of pyscf.df module. The
# modification will affect to all .density_fit() method.
df_jk._DFHF.nuc_grad_method = nuc_grad_method
df_jk._DFHF.Hessian = Hessian


def optimize(mf, callback=None, max_cycle=50, check_conv=True):
    opt = geometric_solver.GeometryOptimizer(mf)
    opt.params = geomeTRIC_params
    if callable(callback):
        opt.callback = functools.partial(callback, solver="geomeTRIC")
    opt.max_cycle = max_cycle
    opt.kernel()

    if check_conv and not opt.converged:
        mf.stdout.write('geomeTRIC is not converged. Switch to berny solver\n')
        mf.mol = opt.mol
        opt = berny_solver.GeometryOptimizer(mf)
        opt.params = berny_params
        if callable(callback):
            opt.callback = functools.partial(callback, solver="berny")
        opt.max_cycle = max_cycle
        opt.kernel()

    if check_conv and not opt.converged:
        raise RuntimeError('geomopt not converged')
    return opt


def run_dft(mol, with_df=False, xc='b3lyp', **kwargs):
    mf = mol.apply('KS')
    mf.xc = xc
    mf.conv_tol = 1e-9
    if with_df:
        mf = mf.density_fit()

    g_scanner = mf.nuc_grad_method().as_scanner()
    opt = optimize(g_scanner, max_cycle=200)
    e_tot = g_scanner.e_tot
    grad = g_scanner.de
    return opt.mol, e_tot, grad

