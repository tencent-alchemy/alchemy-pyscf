#!/usr/bin/env python

'''
Computing the 15 properties reported in QM9 paper
'''

from alchmeypyscf import qm9_properties

# Computing QM9 properties for ethanol

results = qm9_properties('CCO')
for k, v in results.items():
    print(k, v)
