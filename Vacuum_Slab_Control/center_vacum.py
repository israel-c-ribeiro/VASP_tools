# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from ase.io import read, write
from ase.build import surface

# Import optmized bulk
slab = read('POSCAR_file')

# Create surface slab
#slab = surface(bulk, (1, 1, 1), 2)
slab.center(axis=2)

write('POSCAR_file_vac15', slab)
