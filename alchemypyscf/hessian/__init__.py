from . import rhf, uhf, rks, uks

# Replace the nuc_grad_method in df
def Hessian(mf):
    from pyscf import scf
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
        if getattr(mf, 'xc', None):
            return uks.Hessian(mf)
        else:
            return uhf.Hessian(mf)
    elif isinstance(mf, scf.rhf.RHF):
        if getattr(mf, 'xc', None):
            return rks.Hessian(mf)
        else:
            return rhf.Hessian(mf)
    else:
        raise NotImplementedError

from pyscf.df.df_jk import _DFHF
_DFHF.Hessian = Hessian
