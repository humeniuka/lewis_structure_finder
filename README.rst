Bond Order Assignment using Linear Programming
----------------------------------------------

One or more valid Lewis structures are determined for a molecule
based only on the elements and atom connectivities.
Bond orders, formal charges and lone electron pairs are assigned
with the help of linear programming as described in Froeyen and Herdewijn (2005) [1]_


Requirements
------------

Required python packages:

 * numpy, matplotlib

   
Installation
------------
The package is installed with

.. code-block:: bash

   $ pip install -e .

in the top directory.


Getting Started
---------------

While it is not necessary to use an optimized geometry, all atoms (including hydrogens)
should have reasonable 3D positions, so that it is possible to detect which atoms
are connected based on the distance matrix.

Only the element type and atom connectivities are needed to determine the Lewis structures.
The following example computes the two Lewis structures of benzene:

.. code-block:: python

  from lewis_structures.LewisStructures import lewis_structures
  from lewis_structures.XYZ import read_xyz, connectivity_matrix

  # read molecule from xyz-file and determine atom connectivity
  atomlist = read_xyz('examples/benzene.xyz')[0]
  ConMat = connectivity_matrix(atomlist)

  # maximum recursion depth when looking for equivalent Lewis structures
  max_depth=4
  
  # find equivalent Lewis structures and compute the average
  structures, structure_avg = lewis_structures(atomlist, ConMat, max_depth=max_depth)

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


----------
References
----------
.. [1] Froeyen,M. and Herdewijn,P.
    "Correct Bond Order Assignment in a Molecular Framework Using Integer Linear Programming with Application to Molecules Where Only Non-Hydrogen Atom Coordinates Are Available",
    J. Chem. Inf. Model., 2005, 45, 1267-1274.
    https://doi.org/10.1021/ci049645z
       
.. [2] Jiri Matousek, Bernd Gaertner
    "Understanding and Using Linear Programming", Springer, 2007.

