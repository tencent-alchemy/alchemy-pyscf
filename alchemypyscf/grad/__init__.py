from . import rhf, uhf, rks, uks

# Replace the nuc_grad_method in df
def nuc_grad_method(mf):
    from pyscf import scf
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
        if getattr(mf, 'xc', None):
            return uks.Gradients(mf)
        else:
            return uhf.Gradients(mf)
    elif isinstance(mf, scf.rhf.RHF):
        if getattr(mf, 'xc', None):
            return rks.Gradients(mf)
        else:
            return rhf.Gradients(mf)
    else:
        raise NotImplementedError

from pyscf.df.df_jk import _DFHF
_DFHF.nuc_grad_method = _DFHF.Gradients = nuc_grad_method
