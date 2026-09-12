from rdkit import Chem
from parser import Reaction

def canonicalise_mapped(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    return Chem.MolToSmiles(molecule)    

def canonicalise_unmapped(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)    

    return Chem.MolToSmiles(molecule)    

def canonicalise_mapped_polymolecules(smiles: str) -> str | None:
    smiles_list = smiles.split(".")

    molecules = []

    for molecule in smiles_list:
        canonical = canonicalise_mapped(molecule)

        if canonical is None:
            return None

        molecules.append(canonical)

    return ".".join(molecules)        
            

def canonicalise_unmapped_polymolecules(smiles: str) -> str | None:
    smiles_list = smiles.split(".")

    molecules = []

    for molecule in smiles_list:
        canonical = canonicalise_unmapped(molecule)

        if canonical is None:
            return None

        molecules.append(canonical)

    return ".".join(molecules)        
            





def canonicalise_reaction_mapped(reaction: Reaction) -> Reaction:
    reaction.reactants_canonical = canonicalise_mapped_polymolecules(reaction.reactants_raw)

    if reaction.reagants_raw is not None:
        reaction.reagants_canonical = canonicalise_mapped_polymolecules(reaction.reagants_raw)

    if reaction.products_raw is not None:
        reaction.products_canonical = canonicalise_mapped_polymolecules(reaction.products_raw)

    return reaction
    

def canonicalise_reaction_unmapped(reaction: Reaction) -> Reaction:
    reaction.reactants_canonical = canonicalise_unmapped_polymolecules(reaction.reactants_raw)

    if reaction.reagants_raw is not None:
        reaction.reagants_canonical = canonicalise_unmapped_polymolecules(reaction.reagants_raw)

    if reaction.products_raw is not None:
        reaction.products_canonical = canonicalise_unmapped_polymolecules(reaction.products_raw)

    return reaction    
