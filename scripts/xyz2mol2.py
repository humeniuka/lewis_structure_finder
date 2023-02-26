#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from lewis_structures import XYZ
from lewis_structures import LewisStructures

if __name__ == "__main__":
    import sys
    import os.path

    description = """
     converts an XYZ file to a Tripos mol2 file after assigning bond orders
     and formal charges.

     The bond orders and partial charges are determined by averaging
     over all equivalent Lewis structures that can be found.
    """
    
    parser = argparse.ArgumentParser(description=description)
    # input
    parser.add_argument('xyz_file',
        help=('Input file with 3D coordinates in XYZ format. The atom connectivity is determined from the distance matrix.'))
    # output
    parser.add_argument('mol2_file',
        help='Output file with bond orders and charges in Tripos Mol2 format.')

    # options
    parser.add_argument('--max_depth', type=int, default=4,
        help="""maximum recursion depth when looking for equivalent Lewis structures.
      The larger the depth, the more Lewis structures are found. Unless `max_depth`
      is very large, it is not guaranteed that all Lewis structures will be found.""")

    parser.add_argument('--charge', type=int, default=0,
        help='total charge of molecule')

    parser.add_argument('--plot', action='store_true',
        help='plot the average Lewis structure')

    args = parser.parse_args()

    atomlist = XYZ.read_xyz(args.xyz_file)[0]
    ConMat = XYZ.connectivity_matrix(atomlist)

    if args.charge == 0:
        structure_avg = LewisStructures.fragment_lewis_structure(
            atomlist,
            ConMat,
            charge=args.charge,
            max_depth=args.max_depth)
    else:
        # For charged system it is not clear which fragment the charge should
        # be assigned to, so we have to treat the whole system.
        structures, structure_avg = LewisStructures.lewis_structures(
            atomlist,
            ConMat,
            charge=args.charge,
            max_depth=args.max_depth)
        
    # save MOL2
    structure_avg.write_mol2(args.mol2_file)

    # optionally plot the Lewis structure
    if (args.plot == True):
        import matplotlib.pyplot as plt

        structure_avg.plot(title="average Lewis structure")
        plt.show()

