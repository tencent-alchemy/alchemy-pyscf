#!/usr/bin/env python

'''
Computing the 15 properties reported in QM9 paper
'''

from alchmeypyscf import qm9_properties

# Computing QM9 properties for ethanol
smiles = 'CCO'
results = qm9_properties(smiles)
for k, v in results.items():
    print(k, v)
