from rdkit import Chem

from parser import Reaction

def validate_smiles(smiles) -> bool:

    molecule = Chem.MolFromSmiles(smiles)

    return molecule is not None

def validate_reaction(reaction: Reaction) -> Reaction:

    errors = []

    for smiles in reaction.reactants_raw.split("."):
        if not validate_smiles(smiles):
            errors.append(f"invalid smiles in reactants: {smiles}")

    if reaction.reagants_raw is not None:
      for smiles in reaction.reagants_raw.split("."):
        if not validate_smiles(smiles):
            errors.append(f"invalid smiles in reagants: {smiles}")
    if reaction.products_raw is not None:
      for smiles in reaction.products_raw.split("."):
        if not validate_smiles(smiles):
            errors.append(f"invalid smiles in products: {smiles}")

    reaction.is_valid = (len(errors) == 0)
    reaction.validation_errors = errors   

    return reaction            



def get_map_nums(smiles: str) -> set[int]:
  map_nums = set()

  for molecule in smiles.split("."):
    mol = Chem.MolFromSmiles(molecule)

    if mol is None:
      continue

    for atom in mol.GetAtoms():
      num = atom.GetAtomMapNum()

      if num != 0:
        map_nums.add(num) 

  return map_nums      


def get_map_nums_reaction(reaction: Reaction) -> tuple[set[int], set[int], set[int]]:
  reactant_map_nums = get_map_nums(reaction.reactants_raw)


  reagants_map_nums = set()

  if reaction.reagants_raw is not None:
    reagants_map_nums = get_map_nums(reaction.reagants_raw)


  products_map_nums = set()

  if reaction.products_raw is not None:
    products_map_nums = get_map_nums(reaction.products_raw)



  return reactant_map_nums, reagants_map_nums, products_map_nums




def reaction_map_differences(reaction: Reaction) -> dict[str, set[int]]:
  
  reactant_maps, reagant_maps, product_maps = get_map_nums_reaction(reaction)

  return {
    "reactant_only": reactant_maps - product_maps,
    "product_only": product_maps - reactant_maps,
    "shared_product_reactant": reactant_maps & product_maps,
  }


def smiles_molecule_bonds(smiles: str) -> dict[tuple[int, int], str]:
  mol = Chem.MolFromSmiles(smiles)

  bonds = {}

  for bond in mol.GetBonds():
    begin = bond.GetBeginAtom().GetAtomMapNum()
    end = bond.GetEndAtom().GetAtomMapNum()
    bond_type = str(bond.GetBondType())

    if bond == 0 or begin == 0:
      continue

    key = tuple(sorted((begin, end)))

    bonds[key] = bond_type

  return bonds

def smiles_reaction_bonds(reaction: Reaction) -> tuple[
  dict[tuple[int, int], str],
  dict[tuple[int, int], str],
  dict[tuple[int, int], str]
  ]:
  
  reactant_mols = reaction.reactants_raw.split(".")
  if reaction.reagants_raw is not None:
    reagant_mols = reaction.reagants_raw.split(".")
  if reaction.products_raw is not None:
    product_mols = reaction.products_raw.split(".")


  reactant_bonds: dict[tuple(int, int), str] = {}
  reagant_bonds: dict[tuple(int, int), str] = {}
  product_bonds: dict[tuple(int, int), str] = {}

  for mol in reactant_mols:
    reactant_bonds = reactant_bonds | smiles_molecule_bonds(mol)

  if reaction.reagants_raw is not None:
    for mol in reagant_mols:
      reagant_bonds = reagant_bonds | smiles_molecule_bonds(mol)  
 
  if reaction.products_raw is not None:
    for mol in product_mols:
      product_bonds = product_bonds | smiles_molecule_bonds(mol)  

  return reactant_bonds, reagant_bonds, product_bonds


def bond_compare(
  bonds_a: dict[tuple[int, int], str],
  bonds_b: dict[tuple[int, int], str]
  ) -> dict:

  broken = {}
  formed = {}
  changed = {}

  bonds_union = set(bonds_a) | set(bonds_b)

  for bond in bonds_union:
    bond_a = bonds_a.get(bond)
    bond_b = bonds_b.get(bond)


    if bond_a is not None and bond_b is None:
      broken[bond] = bond_a
    elif bond_a is None and bond_b is not None:
      formed[bond] = bond_b
    elif bond_a != bond_b:
      changed[bond] = (bond_a, bond_b)
      print((bond_a, bond_b))
  
  return {
    "broken": broken,
    "formed": formed,
    "changed": changed
  }


def reaction_bonds_changes(reaction: Reaction) -> tuple[
  dict[str, dict],
  dict[str, dict],
  dict[str, dict]
  ]:

  reactant_bonds, reagant_bonds, product_bonds = smiles_reaction_bonds(reaction)
  if reagant_bonds != {}:

    changes_reactant_reagant: dict[str, dict] = bond_compare(reactant_bonds, reagant_bonds)
    changes_reactant_product: dict[str, dict] = bond_compare(reactant_bonds, product_bonds)
    changes_reagant_product: dict[str, dict] = bond_compare(reagant_bonds, product_bonds)

    return changes_reactant_reagant, changes_reagant_product, changes_reactant_product

  else:

    changes_reactant_reagant: dict[str, dict] = {}
    changes_reactant_product: dict[str, dict] = bond_compare(reactant_bonds, product_bonds)
    changes_reagant_product: dict[str, dict] = {}

    return changes_reactant_reagant, changes_reagant_product, changes_reactant_product


def reaction_atom_consistency(reaction: Reaction) -> float:
  reactant_atoms = []
  product_atoms = []


  reactant_mols = reaction.reactants_raw.split(".")
  product_mols =  reaction.products_raw.split(".")

  for mol in reactant_mols:
    reactant_mol = Chem.MolFromSmiles(mol)
    for atom in reactant_mol.GetAtoms():
      reactant_atoms.append(atom.GetSymbol())

  
  for mol in product_mols:
    product_mol = Chem.MolFromSmiles(mol)
    for atom in product_mol.GetAtoms():
      product_atoms.append(atom.GetSymbol())

  found_atoms = 0

  for atom in product_atoms:
    if atom in reactant_atoms:
      reactant_atoms.remove(atom)
      found_atoms += 1

  consistency = (found_atoms * 2) / (len(product_atoms) + len(reactant_atoms) + found_atoms)    

  return consistency