"""
write a molecule with bond orders and formal charges
in the Tripos Mol2 format, which can be used as input for
AMBER's Antechamber.
"""

from lewis_structures import AtomicData

def write_mol2(filename,
               atomlist, bonds, bond_orders, formal_charges):
    """
    save a Lewis structure to a mol2 file.
    
    The mol2 format is described in http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf
    """
    # MOLECULE
    mol2 = f"""
# Lewis structure

@<TRIPOS>MOLECULE

{len(atomlist)} {len(bonds)} 1
SMALL
USER_CHARGES
"""
    subst_name = 'lewis'

    # ATOM
    mol2 += "@<TRIPOS>ATOM\n"
    # Count number of occurances of each element.
    element_counter = {}
    for i in range(0, len(atomlist)):
        atomic_number, position = atomlist[i]
        element = AtomicData.element_name(atomic_number)

        # atoms with the same name are enumerated, e.g. C1, C2, C3, ...
        element_counter[element] = element_counter.get(element, 0)+1

        # write row for atom i
        atom_id = i+1
        atom_name = element.upper()+str(element_counter[element])
        x,y,z = position
        atom_type = element.upper()
        subst_id = 1
        charge = formal_charges[i]
        
        mol2 += f"{atom_id:4d} {atom_name:6s}  {x:12.8f} {y:12.8f} {z:12.8f}  {atom_type:6s}  {subst_id}  {subst_name}  {charge:6.4f}\n"

    # BONDS
    mol2 += "@<TRIPOS>BOND\n"
    for i in range(0, len(bonds)):
        # write row for bond i
        bond_id = i+1
        origin_atom_id = bonds[i][0]+1
        target_atom_id = bonds[i][1]+1

        bo = bond_orders[i]
        if bo in [1.0, 2.0, 3.0]:
            bond_type = int(bo)
        elif 1.0 < bo < 2.0:
            bond_type = 'ar'
        else:
            print(f"WARNING: strange bond order {bo} for bond between atoms {origin_atom_id} and {target_atom_id}")
            bond_type = 'un'

        mol2 += f"{bond_id:4d}   {origin_atom_id:4d}   {target_atom_id:4d}      {bond_type}\n"

    # SUBSTRUCTURE
    mol2 += "@<TRIPOS>SUBSTRUCTURE\n"
    mol2 += f"1     {subst_name}   1\n"

    # write it to file.
    with open(filename, "w") as f:
        f.write(mol2)
        
        
        
