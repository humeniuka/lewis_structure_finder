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

in the top directory. To verify the proper functioning of the code
a set of tests should be run with

.. code-block:: bash

   $ cd tests
   $ python -m unittest


Getting Started
---------------

The package may be used

.. code-block:: python

  from lewis_structures import 
  

		        

The molecule is read from a file in the XYZ format.
While it is not necessary to use an optimized geometry, all atoms (including hydrogens)
should have reasonable 3D positions, so that it is possible to detect which atoms
are connected based on the distance matrix.

Only the element type and atom connectivities are needed to determine the Lewis structures.

   
----------
References
----------
.. [1] Froeyen,M. and Herdewijn,P.
    "Correct Bond Order Assignment in a Molecular Framework Using Integer Linear Programming
     with Application to Molecules Where Only Non-Hydrogen Atom Coordinates Are Available",
    J. Chem. Inf. Model., 2005, 45, 1267-1274.
    https://doi.org/10.1021/ci049645z
       
.. [2] Jiri Matousek, Bernd Gaertner
    "Understanding and Using Linear Programming", Springer, 2007.

