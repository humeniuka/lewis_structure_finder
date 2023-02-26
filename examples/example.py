#!/usr/bin/env python
#
# The example from the README.
#

####
from lewis_structures.LewisStructures import lewis_structures
from lewis_structures.XYZ import read_xyz, connectivity_matrix

# read molecule from xyz-file and determine atom connectivity
atomlist = read_xyz('examples/benzene.xyz')[0]
ConMat = connectivity_matrix(atomlist)

# maximum recursion depth when looking for equivalent Lewis structures
max_depth=4
  
# find equivalent Lewis structures and compute the average
structures, structure_avg = lewis_structures(atomlist, ConMat,
                                             charge=0, max_depth=max_depth)

# For an conjugated system the average bond orders will be fractional numbers,
# e.g. 1.5 for benzene.

# The bond orders and formal charges of the Lewis structure can be exported
# to a Tripos Mol2 file.
structure_avg.write_mol2('benzene.mol2')

# Finally plot all Lewis structures
import matplotlib.pyplot as plt

for i,structure in enumerate(structures):
    structure.plot(title="Lewis structure %d" % (i+1))
plt.show()
    
structure_avg.plot(title="average Lewis structure")
plt.show()


####
from lewis_structures.LewisStructures import fragment_lewis_structure
# find equivalent Lewis structures and compute the average
structure_avg = fragment_lewis_structure(atomlist, ConMat,
                                         charge=0, max_depth=max_depth)
