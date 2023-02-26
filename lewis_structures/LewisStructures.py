#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
find and plot Lewis structures

Bond orders are assigned to the molecular structure 
only based on the atom types and the connectivity.

References
----------
 Froeyen,M. and Herdewijn,P.    J. Chem. Inf. Model., 2005, 45, 1267-1274.
"""
from __future__ import print_function

from lewis_structures import AtomicData
from lewis_structures.LinearProgramming import LinearProgram
from lewis_structures import MolecularCoords
from lewis_structures import MolecularGraph as MG
from lewis_structures import TriposMol2
from lewis_structures import XYZ


import numpy as np
import numpy.linalg as la
import logging
import copy

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(module)-12s] %(message)s", level='DEBUG')


class LewisStructure:
    def __init__(self, atomlist, bonds, bond_orders, lone_pairs, formal_charges):
        self.atomlist = atomlist
        self.bonds = bonds
        self.bond_orders = bond_orders
        self.lone_pairs = lone_pairs
        self.formal_charges = formal_charges

    def __add__(self, other):
        """
        combine the Lewis structure from two molecular fragments into a single one
        """
        nat = len(self.atomlist)
        # The atom indices of the second Lewis structures have to be shifted
        # by `nat`.
        other_bonds = [(I+nat,J+nat) for (I,J) in other.bonds]
        return LewisStructure(self.atomlist+other.atomlist,
                              self.bonds+other_bonds,
                              self.bond_orders+other.bond_orders,
                              self.lone_pairs+other.lone_pairs,
                              self.formal_charges+other.formal_charges)
    
    def plot(self, title=""):
        """
        plot Lewis structure using matplotlib
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=0.0, azim=0.0)
        ax.set_axis_off()

        # Bring molecule into standard orientation, shift its center of mass
        # to the origin and align the principal axes of inertia with the
        # cartesian axes.
        atomlist = MolecularCoords.standard_orientation(self.atomlist)
        
        # ATOMS
        # color scheme for elements
        colors = {'H': 'black', 'C': 'grey', 'N': 'blue', 'O': 'red'}
        # show element names of atoms
        for (Z,pos),charge in zip(atomlist, self.formal_charges):
            atname = AtomicData.element_name(Z).upper()
            x,y,z = pos
            label = atname.upper()
            # formal charges
            if charge == 0:
                pass
            elif charge == 1:
                label += r"$^+$"
            elif charge == -1:
                label += r"$^-$"
            elif int(charge) == charge:
                label += r"$^{%+d}$" % charge
            else:
                # fractional charges
                label += r"$^{%+4.3f}$" % charge
                
            ax.text(x,y,z, label, None,
                    verticalalignment='center', horizontalalignment='center',
                    color=colors.get(atname, 'green'))

        # distances between all atoms
        D = XYZ.distance_matrix(atomlist)

        # limits of viewport
        rad = 0.5*D.max()
        ax.set_xlim((-rad,rad))
        ax.set_ylim((-rad,rad))
        ax.set_zlim((-rad,rad))
        
        # BONDS
        for (I,J),bo in zip(self.bonds, self.bond_orders):
            # The bond is drawn as a vector from rI to rJ
            rI = np.array(atomlist[I][1])
            rJ = np.array(atomlist[J][1])
            # K is the atom closest to I and J.
            indices = np.argsort(D[I,:]+D[J,:])
            for K in indices:
                if (K == I) or (K == J):
                    continue
                break
            rK = np.array(atomlist[K][1])
            # The double and triple bonds are drawn in the plane defined
            # by the atoms I,J,K
            n,a,b = MolecularCoords.construct_tripod(rI,rJ,rK)
            # axis n  is parallel to the bond, a is perpendicular to the bond

            if bo == 1:
                # single bond
                points = np.array([rI + 0.1*D[I,J]*n, rI + 0.9*D[I,J]*n])
                ax.plot(points[:,0],points[:,1], points[:,2], lw=1, color="black")
            elif bo == 2:
                # double bond
                # upper line
                points = np.array([rI + 0.1*D[I,J]*n + 0.15*a, rI + 0.9*D[I,J]*n + 0.15*a])
                ax.plot(points[:,0],points[:,1], points[:,2], lw=1, color="black")
                # lower line
                points = np.array([rI + 0.1*D[I,J]*n - 0.15*a, rI + 0.9*D[I,J]*n - 0.15*a])
                ax.plot(points[:,0],points[:,1], points[:,2], lw=1, color="black")
            elif bo == 3:
                # triple bond
                # upper line
                points = np.array([rI + 0.1*D[I,J]*n + 0.25*a, rI + 0.9*D[I,J]*n + 0.25*a])
                ax.plot(points[:,0],points[:,1], points[:,2], lw=1, color="black")
                # middle line
                points = np.array([rI + 0.1*D[I,J]*n, rI + 0.9*D[I,J]*n])
                ax.plot(points[:,0],points[:,1], points[:,2], lw=1, color="black")                
                # lower line
                points = np.array([rI + 0.1*D[I,J]*n - 0.25*a, rI + 0.9*D[I,J]*n - 0.25*a])
                ax.plot(points[:,0],points[:,1], points[:,2], lw=1, color="black")
            elif bo == 1.5:
                # resonant bond
                # upper line
                points = np.array([rI + 0.1*D[I,J]*n + 0.15*a, rI + 0.9*D[I,J]*n + 0.15*a])
                ax.plot(points[:,0],points[:,1], points[:,2], lw=1, color="black")
                # lower line
                points = np.array([rI + 0.1*D[I,J]*n - 0.15*a, rI + 0.9*D[I,J]*n - 0.15*a])
                ax.plot(points[:,0],points[:,1], points[:,2], lw=1, color="black", ls="-.")
            else:
                # all other bonds (with non-integer bond order)
                # upper line
                points = np.array([rI + 0.1*D[I,J]*n + 0.15*a, rI + 0.9*D[I,J]*n + 0.15*a])
                ax.plot(points[:,0],points[:,1], points[:,2], lw=1, color="black", ls="--")
                # lower line
                points = np.array([rI + 0.1*D[I,J]*n - 0.15*a, rI + 0.9*D[I,J]*n - 0.15*a])
                ax.plot(points[:,0],points[:,1], points[:,2], lw=1, color="black", ls="-.")
                        
    def __str__(self):
        Nat = len(self.atomlist)
        txt  = "Bond Orders\n"
        txt += "===========\n"
        for i,bond in enumerate(self.bonds):
            bo = self.bond_orders[i]
            I,J = bond
            #
            Zi = self.atomlist[I][0]
            Zj = self.atomlist[J][0]
            
            txt += "BOND %s%d-%s%d \t BO=%f\n" % (AtomicData.atom_names[Zi-1], I+1, AtomicData.atom_names[Zj-1], J+1, bo)
        txt += "Free Electron Pairs\n"
        txt += "===================\n"
        for I in range(0, Nat):
            fI = self.lone_pairs[I]
            if (fI > 0):
                Zi = self.atomlist[I][0]
                atname = AtomicData.atom_names[Zi-1]
                txt += "Atom %s%d  number of lone pairs = %d\n" % (atname, I+1,fI)
        txt += "Formal Charges\n"
        txt += "==============\n"
        for I in range(0, Nat):
            fcI = self.formal_charges[I]
            if abs(fcI) > 0:
                Zi = self.atomlist[I][0]
                atname = AtomicData.atom_names[Zi-1]
                # change sign of fcI so that having less electrons leads to a positive number
                txt += "Atom %s%d  formal charge = %f\n" % (atname, I+1, -fcI)
        return txt
            
    def write_mol2(self, filename):
        """
        save the Lewis structure (atom positions, bond orders, formal charges)
        to file in the Tripos Mol2 format.
        """
        TriposMol2.write_mol2(filename,
                              self.atomlist,
                              self.bonds,
                              self.bond_orders,
                              self.formal_charges)
        
                
def solve_linprog_simplex(c, A_ub, b_ub, A_eq, b_eq, **kwds):
    """
    solve a linear program of the form 

       maximize    c^T.x 
       subject to  A_ub.x <= b_ub
                   A_eq.x == b_eq

    using the simplex algorithm and find all optimal solutions x* 

    Returns
    -------
    xs     :  list of ndarray
      optimal solutions
    """
    # Note that the order of equality and inequality constraints
    # is swapped in this function call.
    lp = LinearProgram(c, A_eq, b_eq, A_ub, b_ub)
    # First we find one optimal solution x* with z*=z(x*) ...
    x, z = lp.solve()
    # ... then we enumerate all other solutions with the same z.
    xs, z = lp.all_optimal_solutions(**kwds)
    
    return xs

def is_metal_ion(Z):
    if Z in [12, 28, 30]: # magnesium, nickel, zinc
        return True
    else:
        return False

def fragment_lewis_structure(atomlist_full, ConMat, charge=0, max_depth=5):
    """
    If the geometry contains disconnected fragments, the Lewis structures are
    assigned for each fragment separately. For identical fragments, which have
    the same connectivity and ordering of atoms, a Lewis structure is determined
    only once and copied to all fragments.

    Only the average Lewis structures are combined, since the 

    Parameters
    ----------
    For explanations about the parameters see `lewis_structures(...)`.

    Returns
    -------
    structure_avg :  instance of LewisStructure
      average Lewis structure for the entire molecule with fractional bond orders
    """
    logger.info("identifying disconnected fragments")
    fragment_graphs = MG.atomlist2graph(atomlist_full, conmat=ConMat)
    # list of indices into atomlist of atoms belonging to each fragment
    fragment_indices = [np.array(MG.graph2indeces(g), dtype=int) for g in fragment_graphs]
    # Ordering of atoms within each fragment
    fragment_ordering = [np.argsort(indices) for indices in fragment_indices]
    # atomic coordinates of fragments
    fragments = []
    for g, ordering in zip(fragment_graphs, fragment_ordering):
        atomlist = MG.graph2atomlist(g, atomlist_full)
        # reorder atoms
        atomlist = [atomlist[i] for i in ordering]
        fragments.append(atomlist)
    nfrag = len(fragments)
    if nfrag > 0:
        assert charge == 0, "For charged systems it is not clear which fragment the charge should be assigned to."
    # canonical Morgan names for fragments
    fragment_labels = [MG.identifier( *MG.morgan_ordering(atomlist) ) for atomlist in fragments]
    logger.info("Found %d fragment(s). An averaged Lewis structure will be assigned to each of them." % nfrag)
    # determine an average Lewis structure for each fragment
    fragment_lewis = []
    # We keep a cache of fragments for which Lewis structure have been assigned already. If
    # the same fragment comes up again, we just use the cached Lewis structure.
    cache = {}
    for atomlist, indices, ordering, label in zip(fragments, fragment_indices,
                                                  fragment_ordering, fragment_labels):
        logger.info("Lewis structure for fragment '%s'" % label)
        # connectivity within a fragment
        idx = indices[ordering]
        fragment_conmat = ConMat[idx,:][:,idx]
        
        # Did we see a similar fragment before?
        stored = cache.get(label, None)
        if (not stored is None) and (la.norm(stored['connectivity'] - fragment_conmat)) == 0:
            # The order of the atoms in two fragments has to be the same, otherwise we cannot
            # copy the Lewis structures from one to the other.
            logger.info("Fragment has already been assigned.")
            structure_avg = stored["Lewis-structure"]
            # Only replace geometry
            structure_avg = copy.deepcopy(structure_avg)
            structure_avg.atomlist = atomlist
            
            fragment_lewis.append(structure_avg)
            continue

        # assign Lewis structure
        structures, structure_avg = lewis_structures(atomlist, fragment_conmat,
                                                     charge=0, max_depth=max_depth)
        # store Lewis structure in cache
        cache[label] = {"Lewis-structure" : structure_avg,
                        "connectivity" : fragment_conmat}
        
        fragment_lewis.append(structure_avg)

    # combine all Lewis structures
    structure_avg = sum(fragment_lewis[1:], fragment_lewis[0])

    print("--- Average Lewis Structure of Entire Molecule ----")
    print(structure_avg)
    
    return structure_avg
        
    
def lewis_structures(atomlist, ConMat, charge=0, max_depth=5):
    """
    Determine Lewis structure only based on element types and connectivity.
    The problem of distributing the electrons over the bonds in an optimal fashion
    is formulated as a integer linear program following the article

    Froeyen,~M. Herdewijn,P.
    'Correct Bond Order Assignment in a Molecular Framework Using Integer Linear
    Programming with Application to Molecules Where Only Non-Hydrogen Atom Coordinates
    Are Available'
    J. Chem. Inf. Model., 2005, 45, 1267-1274.

    Parameters:
    -----------
    atomlist: list of Nat tuples (Zi,[xi,yi,zi]) with atomic number and positions
    ConMat: connectivity matrix, if atom I is bonded to atom J, then ConMat[I,J] = 1

    Optional:
    ---------
    charge  :  int
      total charge of molecule
    max_depth  :  int > 0
      maximum recursion depth when looking for equivalent Lewis structures.
      The larger the depth, the more Lewis structures are found. Unless max_depth
      is very large, it is not guaranteed that all Lewis structures will be found.

    Results:
    --------
    structures    :  list of instances of LewisStructures
    structure_avg :  average Lewis structure with fractional bond orders
    """
    # Xij is the number of bonds between atoms i and j
    # Xii is the number of lone pairs on atom i
    
    Nat = len(atomlist)
    # octet electrons
    OCT = 0.0
    # number of valence electrons
    V = 0.0 - charge
    ks = []
    valence_electrons = []
    for i,(Zi,posi) in enumerate(atomlist):
        atname = AtomicData.atom_names[Zi-1]
        # octet electrons
        if (Zi == 1):
            #
            ki = 2
        elif (Zi == 30):
            # Zinc 2+
            ki = 10
        else:
            ki = 8
        ks.append(ki)
        OCT += ki
        # valence electrons
        vi = AtomicData.valence_electrons[atname]
        valence_electrons.append(vi)
        V += vi
    # number of bond electrons
    B = OCT-V
    # number of electrons not participating in bonds
    F = V-B
    # constraints
    bonds = []
    # create list of all bonds (I,J)
    i = 0
    for I in range(0, Nat):
        for J in range(I+1, Nat):
            if ConMat[I,J] == 1:
                #
                Zi = atomlist[I][0]
                Zj = atomlist[J][0]
                if is_metal_ion(Zi) or is_metal_ion(Zj):
                    # neglect ionic bonds
                    logger.info("neglect ionic bond between %s%d and %s%d" % (AtomicData.atom_names[Zi-1], I+1, AtomicData.atom_names[Zj-1], J+1))
                    continue
                #
                bonds.append( (I,J) )

                logger.debug("BOND %d  = %s%d-%s%d" % (i, AtomicData.atom_names[Zi-1], I+1, AtomicData.atom_names[Zj-1], J+1))
                i += 1
    # number of bonds + number of free electron pairs = number of variables Xk
    Nb = len(bonds)
    N = Nb + Nat
    # EQUALITY CONSTRAINTS
    constraintsD = []
    constraintsB = []
    # 1.  sum_(i,j) X_ij = V
    #     2 * (bonds + lone pairs) = total number of electrons
    d = 2 * np.ones(N)
    b = V
    constraintsD.append(d)
    constraintsB.append(b)
    # 2. total number of bond electrons
    d = np.zeros(N)
    for i in range(0, Nb):
        d[i] = 2    # 2 per bond
    b = B
    constraintsD.append(d)
    constraintsB.append(b)
    # 3. total number of free electrons
    d = np.zeros(N)
    for i in range(Nb,N):
        d[i] = 2    # 2 per lone pair
    b = F
    constraintsD.append(d)
    constraintsB.append(b)
    # 4. octet rule for each atom
    for I in range(0, Nat):
        d = np.zeros(N)
        d[Nb+I] = 2 # 2 electrons per lone pair
        for i,bond in enumerate(bonds):
            if bond[0] == I or bond[1] == I:
                d[i] = 2   # 2 electrons per bond 
        b = ks[I]

        constraintsD.append(d)
        constraintsB.append(b)
    # 5. hydrogen and carbon should have no formal charges, zinc should have formal charge +2
    Nc = 0 # number of hydrogens and carbons
    formal_charge_constraints = {1: 0,
                                 6: 0,
                                 30: +2} # hydrogen and carbon: 0, zinc: +2
    for I in range(0, Nat):
        Zi = atomlist[I][0]
        if Zi in formal_charge_constraints.keys():
            Nc += 1
            # hydrogen or carbon
            d = np.zeros(N)
            b = valence_electrons[I] - formal_charge_constraints[Zi]  # no formal charge FC=valence electrons - electrons attributed to atom I 
            # electrons in bonds attached to atom I
            for i,bond in enumerate(bonds):
                if bond[0] == I or bond[1] == I:
                    d[i] = 1  # bond electrons are split equally between partners
            # electrons in free electron pairs attached to atom I
            d[Nb+I] = 2 # all 2 electrons from a lone pair belong to the atom

            constraintsD.append(d)
            constraintsB.append(b)

    # UPPER BOUND CONSTRAINTS
    ub_constraintsD = []
    ub_constraintsB = []
    
    # 1. Each bond should have at least a bond order of 1
    #      bo(i) >= 1    <=>  -bo(i) <= -1
    #
    for i,bond in enumerate(bonds):
        d = np.zeros(N)
        d[i] = -1
        b = -1
        ub_constraintsD.append(d)
        ub_constraintsB.append(b)

    # 2. nitrogen/oxygen should have a formal charge of at most -1/-2
    Nfc = 0 # number of atoms which can have formal charges
    fc_lower_limit = {7: -1, 8: -2}  # nitrogen can have at most 1 addition electron, oxygen at most 2
    for I in range(0, Nat):
        Zi = atomlist[I][0]
        if Zi in fc_lower_limit.keys():
            Nfc += 1
            #
            d = np.zeros(N)
            b = valence_electrons[I]-fc_lower_limit[Zi]  # formal charge FC=-(valence electrons - electrons attributed to atom I) of at most 1 or 2 electrons
            # electrons in bonds attached to atom I
            for i,bond in enumerate(bonds):
                if bond[0] == I or bond[1] == I:
                    d[i] = 1  # bond electrons are split equally between partners
            # electrons in free electron pairs attached to atom I
            d[Nb+I] = 2 # all 2 electrons from a lone pair belong to the atom

            ub_constraintsD.append(d)
            ub_constraintsB.append(b)

    # 3. nitrogen should have at most +1 and oxygen should never have a positive formal charge
    fc_upper_limit = {7: +1, 8: 0}
    for I in range(0, Nat):
        Zi = atomlist[I][0]
        if Zi in fc_upper_limit.keys():
            d = np.zeros(N)
            b = valence_electrons[I]-fc_upper_limit[Zi]  # formal charge FC=-(valence electrons - electrons attributed to atom I) of at least 0
            # electrons in bonds attached to atom I
            for i,bond in enumerate(bonds):
                if bond[0] == I or bond[1] == I:
                    d[i] = 1  # bond electrons are split equally between partners
            # electrons in free electron pairs attached to atom I
            d[Nb+I] = 2 # all 2 electrons from a lone pair belong to the atom

            ub_constraintsD.append(-d)
            ub_constraintsB.append(-b)

    logger.info("number of atoms: %d" % Nat)
    logger.info("number of bonds: %d" % Nb)
    logger.info("number of valence electrons V=%d" % V)
    logger.info("number of bond electrons B=%d" % B)
    logger.info("number of variables: %d" % N)

    logger.info("number of equality constraints: %d" % len(constraintsB))
    logger.info("number of upper-bound constraints: %d" % len(ub_constraintsB))
    
    # OBJECTIVE LINEAR FUNCTION
    c = np.ones(N)
    # equality constraints
    Nc = len(constraintsB)
    A_eq = np.array(constraintsD)  # number of bonds = number of bonding electron pairs = 2*number of electrons
    b_eq = np.array(constraintsB)
    # upper-bound constraints
    Nc_ub = len(ub_constraintsB)
    A_ub = np.reshape(np.array(ub_constraintsD), (Nc_ub, N))
    b_ub = np.reshape(np.array(ub_constraintsB), (Nc_ub,))

    Xs = solve_linprog_simplex(c, A_ub, b_ub, A_eq, b_eq,
                               max_depth=max_depth)

    def solution2lewis(X):
        """

        convert solution vector X into bond orders, lone pairs and formal charges

        Parameters
        ----------
        X          :  ndarray (shape (Nb+Nat))
          solution vector of linear program

        Returns
        -------
        structure  :  instance of LewisStructure

        """
        # Ideally all solutions should be integer, however the simplex algorithm might
        # give non-integer solutions. The most common case is a bond order of 1.5 in
        # aromatic rings. Therefore we round the solution to 1-digit, so as to remove
        # small numerical errors but keep the large deviations.
        X = list(np.round(X,1))
        bond_orders = X[:Nb]
        lone_pairs = X[Nb:Nb+Nat]
        formal_charges = []
        for I in range(0, Nat):
            fcI = 0.0
            for i,bond in enumerate(bonds):
                # bonding electrons are shared
                if bond[0] == I or bond[1] == I:
                    fcI += bond_orders[i]
            fcI += 2.0*lone_pairs[I]
            fcI -= valence_electrons[I]
            formal_charges.append( -fcI )

        structure = LewisStructure(atomlist, bonds, bond_orders, lone_pairs, formal_charges)
        return structure
        
    # loop over equivalent solutions, each solution corresponds to a Lewis structure
    structures = []
    for i,X in enumerate(Xs):
        print("---- Lewis Structure %d ----" % (i+1))
        structure = solution2lewis(X)
        print(structure)
        structures.append(structure)

    # average over all solutions
    Xavg = list(np.mean(Xs, axis=0))
    # The fractional bond orders obtained from the average solution are a generalization
    # of Pauling bond orders for general organic molecules.
    # For benzoid hydrocarbons the Pauling bond orders are defined as
    #  BO_ij = K_ij/K
    # where K is the total number of Kekule structures and K_ij is the number of
    # Kekule structures with a double bond between carbons i and j.
    structure_avg = solution2lewis(Xavg)
    print("---- Average Lewis Structure ----")
    print(structure_avg)
    
    return structures, structure_avg


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    
    xyz_file = args[0]
    if len(args) > 1:
        max_depth = int(args[1])
    else:
        max_depth = 4
    
    atomlist = XYZ.read_xyz(xyz_file)[0]
    ConMat = XYZ.connectivity_matrix(atomlist)
    structures, structure_avg = lewis_structures(atomlist, ConMat, max_depth=max_depth)

    #structure_avg = fragment_lewis_structure(atomlist, ConMat, max_depth=max_depth)

    structure_avg.write_mol2('/tmp/average.mol2')
    
    import matplotlib.pyplot as plt

    for i,structure in enumerate(structures):
        structure.plot(title="Lewis structure %d" % (i+1))
        plt.show()
    
    structure_avg.plot(title="average Lewis structure")
    plt.show()
