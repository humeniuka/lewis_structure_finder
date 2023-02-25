"""\
 adapted from 

 Code for reading/writing Xmol XYZ files.
 
 This program is part of the PyQuante quantum chemistry program suite.

 Copyright (c) 2004, Richard P. Muller. All Rights Reserved. 

 PyQuante version 1.2 and later is covered by the modified BSD
 license. Please see the file LICENSE that is part of this
 distribution. 
"""
from __future__ import print_function

import numpy as np
import numpy.linalg as la

from lewis_structures.AtomicData import atom_names, atomic_number, bohr_to_angs, covalent_radii


def read_xyz_it(filename, units="Angstrom", fragment_id=atomic_number):
    """
    same as read_xyz, with the difference that an iterator to the geometries
    is returned instead of a list.
    For very large MD trajectories not all geometries can be kept in memory.  
    The iterator returns one geometry at a time which can be processed in a pipeline.

    Parameters:
    ===========
    filename: path to xyz-file
    units: specify the units of coordinates in xyz-file, "Angstrom" or "bohr"
    fragment_id: a function that takes the name of the atom and assigns a
       number to it, usually the atomic number Z.

    Returns:
    ========
    iterator to individual structures
    """
    assert units in ["bohr", "Angstrom", "hartree/bohr", ""]
    fh = open(filename)
    igeo = 1
    while 1:
        line = fh.readline()
        if not line:
            # end of file reached
            break
        words = line.strip().split()
        if words[0] == "Tv":
            # skip lattice vectors
            continue
        try:
            nat = int(words[0])
        except ValueError as e:
            print(e)
            raise Exception("Probably wrong number of atoms in xyz-file '%s'" % filename)
        # skip title
        title = fh.readline()
        # read coordinates of nat atoms
        atoms = []
        for i in range(nat):
            line = fh.readline()
            words = line.split()
            atno = fragment_id(words[0])
            x,y,z = map(float,words[1:4])
            if units == "Angstrom":
                x,y,z = map(lambda c: c/bohr_to_angs, [x,y,z])
            atoms.append((atno,(x,y,z)))
        igeo += 1
        yield atoms

def read_xyz(filename, units="Angstrom", fragment_id=atomic_number):
    """
    read geometries from xyz-file

    Parameters:
    ===========
    filename: path to xyz-file
    units: specify the units of coordinates in xyz-file, "Angstrom" or "bohr"
    fragment_id: a function that takes the name of the atom and assigns a
       number to it, usually the atomic number Z.
    
    Returns:
    ========
    list of structures, each structure is a list of atom numbers and positions 
       [(Z1, (x1,y1,z1)), (Z2,(x2,y2,z2)), ...]
    """
    geometries = []
    for atoms in read_xyz_it(filename, units=units, fragment_id=fragment_id):
        geometries.append(atoms)
    return geometries

def write_xyz(filename,geometries,title=" ", units="Angstrom", mode='w'):
    """
    write geometries to xyz-file

    Parameters:
    ===========
    geometries: list of geometries, each is a list of tuples
      of type (Zi, (xi,yi,zi))

    Optional:
    =========
    title: string, if a list of strings is provided, the number
           titles should match the number of geometries
    """
    fh = open(filename,mode)
    txt = xyz2txt(geometries, title, units)
    fh.write(txt)
    fh.close()
    return

def xyz2txt(geometries, title="", units="Angstrom"):
    """
    write nuclear geometry to a string in xyz-format.

    Parameters:
    ===========
    geometries: list of geometries, each is a list of tuples
      of type (Zi, (xi,yi,zi))
    
    Optional:
    =========
    title: string, if a list of strings is provided, the number
           titles should match the number of geometries

    Returns:
    ========
    string 
    """
    txt = ""
    for i,atoms in enumerate(geometries):
        if type(title) == list:
            if i < len(title):
                current_title = title[i]
            else:
                current_title = " "
        else:
            current_title = title
        txt += _append_xyz2txt(atoms, current_title, units)
    return txt

def _append_xyz2txt(atoms,title="", units="Angstrom"):
    txt = "%d\n%s\n" % (len(atoms),title)
    for atom in atoms:
        atno,pos = atom
        x,y,z = pos[0], pos[1], pos[2]
        if units == "Angstrom":
            x,y,z = map(lambda c: c*bohr_to_angs, [x,y,z])
        try:
            atname = atom_names[atno-1]
        except TypeError:
            # leave it as is
            atname = atno
        txt += "%4s  %+15.10f  %+15.10f  %+15.10f\n" \
                   % (atname.capitalize(),x,y,z)
    return txt

def atomlist2vector(atomlist):
    """
    convert a list of atom positions [(Z1,(x1,y1,z1)), (Z2,(x2,y2,z2)),...]
    to a numpy array that contains on the positions: [x1,y1,z1,x2,y2,z2,...]
    """
    vec = np.zeros(3*len(atomlist))
    for i,(Zi,posi) in enumerate(atomlist):
        vec[3*i+0] = posi[0]
        vec[3*i+1] = posi[1]
        vec[3*i+2] = posi[2]
    return vec

def vector2atomlist(vec, ref_atomlist):
    """
    convert a vector [x1,y1,z1,x2,y2,z2,...] to a list of atom positions 
x    with atom types [(Z1,(x1,y1,z1)), (Z2,(x2,y2,z2)),...].
    The atom types are assigned in the same order as in ref_atomlist.
    """
    atomlist = []
    for i,(Zi,posi) in enumerate(ref_atomlist):
        atomlist.append( (Zi, vec[3*i:3*i+3]) )
    return atomlist
    
def distance_matrix(atomlist):
    """

    compute distances between all atoms and store them in matrix-form

    Paramters
    ---------
    atomlist: list of tuples (Zi,[x,y,z])
      atomic numbers and molecular geometry

    Returns
    -------
    D      :  ndarray (shape (nat,nat))
      distance matrix, D[i,j] = |R(i)-R(j)|

    """
    nat = len(atomlist)
    # D[i,j] = |R(i)-R(j)|
    D = np.zeros((nat,nat))
    for i in range(0, nat):
        Zi,posi = atomlist[i]
        D[i,i] = 0.0
        for j in range(i+1,nat):
            Zj,posj = atomlist[j]
            D[i,j] = la.norm(np.array(posj)-np.array(posi))
            D[j,i] = D[i,j]
    return D
    
def connectivity_matrix(atomlist, search_neighbours=None, thresh=1.3, hydrogen_bonds=False, debug=0):
    """
    compute matrix that shows which atoms are connected by bonds

    C[i,j] = 0    means that atoms i and j are not connected
    C[i,j] = 1    means that they are connected, they are closer than the <thresh>*(bond length between i and j)

    Paramters:
    ==========
    atomlist: list of tuples (Zi,posi) for each atom

    Optional:
    =========
    search_neighbours: search for connected atoms among the 
         <search_neighbours> atoms that are listed right after the current atom. 
         If search_neighbours == None, all atoms are checked.
    thresh: bond lengths can be larger by this factor and are still recognized
    hydrogen_bonds: include hydrogen bonds, too. If hydrogen donor i and hydrogen acceptor
         j are connected through a hydrogen atom k, the elements C[i,k] and C[j,k] are set
         to 1.

    Returns:
    ========
    Con: 2d numpy array with adjacency matrix
    """
    Nat = len(atomlist)
    Con = np.zeros((Nat,Nat), dtype=int)
    if search_neighbours == None:
        search_neighbours = Nat
    for A in range(0, Nat):
        ZA,posA = atomlist[A]
        for B in range(A+1, min(A+search_neighbours, Nat)):
            ZB,posB = atomlist[B]
            RAB = la.norm(np.array(posB) - np.array(posA))
            # approximate bond length by the sum of the covalent radii (in bohr)
            bond_length = covalent_radii[atom_names[ZA-1]] + covalent_radii[atom_names[ZB-1]]
            if RAB < thresh * bond_length:
                Con[A,B] = 1
                Con[B,A] = 1
                
    if (hydrogen_bonds == True):
        # Hydrogen bonds should have
        # 1) donor-acceptor-distances <= 3.5 Angstrom
        # 2) hydrogen-donor-acceptor angles <= 30 degrees
        #
        max_donor_acceptor_distance = 3.5
        max_H_donor_acceptor_angle = 30.0
        # The following atoms can donate or accept a hydrogen bond: O, N
        donor_acceptor_atoms = [8,9] 
        for A in range(0, Nat):  # loop over possible donors or acceptors
            ZA,posA = atomlist[A]
            posA = np.array(posA)
            if not (ZA in donor_acceptor_atoms):  # cannot donate or accept a hydrogen bond
                continue
            for B in range(A+1, min(A+search_neighbours, Nat)): # loop over possible donors or acceptors
                ZB,posB = atomlist[B]
                posB = np.array(posB)
                if not (ZB in donor_acceptor_atoms):
                    continue
                # donor-acceptor-distance
                RAB = la.norm(posB - posA)
                if (RAB*bohr_to_angs > max_donor_acceptor_distance):
                    continue
                for C in range(0, Nat): # loop over hydrogens
                    ZC,posC = atomlist[C]
                    posC = np.array(posC)
                    if ZC != 1: # not a hydrogen atom
                        continue
                    # Which of the atoms A or B is the donor atom? The one that is closer to the hydrogen
                    RAC = la.norm(posC - posA)
                    RBC = la.norm(posC - posB)
                    if RAC < RBC:
                        # A atom is the hydrogen donor
                        r_donor_H = posC - posA
                        r_donor_acceptor = posB - posA
                    else:
                        # B atom is the hydrogen donor
                        r_donor_H = posC - posB
                        r_donor_acceptor = posA - posB
                    
                    # hydrogen-donor-acceptor angle
                    # angle angle(hydrogen --> donor --> acceptor)
                    angle = np.arccos( np.dot(r_donor_H,r_donor_acceptor)/(la.norm(r_donor_H)*la.norm(r_donor_acceptor)) )
                    if angle*180.0/np.pi < max_H_donor_acceptor_angle:
                        # hydrogen bond found
                        # donor/acceptor -- H
                        Con[A,C] = 1
                        Con[C,A] = 1
                        # H -- acceptor/donor
                        Con[B,C] = 1
                        Con[C,B] = 1

                        if debug > 0:
                            print("hydrogen bond %s(%2.d)--H(%2.d)--%s(%2.d)     distance= %8.4f Ang    angle= %8.4f degrees" \
                            % (atom_names[ZA-1].upper(), A+1, C+1, atom_names[ZB-1].upper(), B+1, RAB*bohr_to_angs, angle*180.0/np.pi))

        # 
    return Con

def read_pdb(filename):
    """
    
    Parameters
    ----------
    filename  :  str
       path to pdb file
    
    Returns
    -------
    atomlist  :  list
      list of atomic numbers and coordinates as tuples (Zat,[x,y,z])
    Con       :  ndarray (shape (nat,nat))
      connectivity matrix

    """
    def l2s(l):
        """convert a list of characters into a string"""
        return "".join(l).strip()
    
    # In the first pass only the coordinates are read
    atomlist = []
    with open(filename) as f:
        for line in f.readlines():
            columns = list(line)
            if l2s(columns[0:7]) in ["ATOM", "HETATM"]:
                # For format of ATOM record see
                #     https://zhanglab.ccmb.med.umich.edu/SSIPe/pdb_atom_format.html
                #  X coordinate is stored in columns 31-38 in format Real(8.3)
                #  Y coordinate is stored in columns 39-46 in format Real(8.3)
                #  Z coordinate is stored in columns 47-54 in format Real(8.3)
                pos = map(lambda l: float(l2s(l)),
                          [columns[30:38], columns[38:46], columns[46:54]])
                element = l2s(columns[76:78]).lower()
                Z = atomic_number(element)
                atomlist.append( (Z,pos) )

    # In the second pass only the connectivity matrix is built
    Nat = len(atomlist)
    ConMat = np.zeros((Nat,Nat), dtype=int)
    with open(filename) as f:
        for line in f.readlines():
            words = line.split()
            key = words[0].strip()
            if key == "CONECT":
                i = int(words[1])
                connected_atoms = map(int, words[2:])
                for j in connected_atoms:
                    ConMat[i-1,j-1] = 1

    if (np.sum(ConMat) == 0):
        raise RuntimeError("PDB file does not contain connectivity information (CONECT lines)!")
                    
    return atomlist, ConMat


__all__ = ["read_xyz", "read_pdb", "write_xyz"]
