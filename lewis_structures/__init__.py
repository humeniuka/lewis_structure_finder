"""
find and plot Lewis structures

Bond orders are assigned to the molecular structure 
only based on the atom types and the connectivity.

References
----------
 Froeyen,M. and Herdewijn,P.    J. Chem. Inf. Model., 2005, 45, 1267-1274.
"""

__version__ = '0.0.1'

__all__ = ['XYZ', 'lewis_structures']

import XYZ
from LewisStructures import lewis_structures
