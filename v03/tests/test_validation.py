import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import validation
import parser


def test_valid_reaction_single_reactant():
    reaction = parser.Reaction.parse_reaction(
        smiles="CCO>>CC=O",
        id=1,
    )

    reaction = validation.validate_reaction(reaction)

    assert reaction.id == 1
    assert reaction.validation_errors == []
    assert reaction.is_valid == True


def test_valid_reaction_multi_reactant():
    reaction = parser.Reaction.parse_reaction(
        smiles="CCO.CN>>CCN",
        id=2,
    )

    reaction = validation.validate_reaction(reaction)

    assert reaction.id == 2
    assert reaction.validation_errors == []
    assert reaction.is_valid == True    




def test_valid_reaction_single_product():
    reaction = parser.Reaction.parse_reaction(
        smiles="CCO>>CC=O",
        id=3,
    )

    reaction = validation.validate_reaction(reaction)

    assert reaction.id == 3
    assert reaction.validation_errors == []
    assert reaction.is_valid == True


def test_valid_reaction_multi_product():
    reaction = parser.Reaction.parse_reaction(
        smiles="CCO>>CCN.CN",
        id=4,
    )

    reaction = validation.validate_reaction(reaction)


    assert reaction.id == 4
    assert reaction.validation_errors == []
    assert reaction.is_valid == True    








def test_valid_reaction_single_reagant():
    reaction = parser.Reaction.parse_reaction(
        smiles="CCO>>CC=O",
        id=5,
    )

    reaction = validation.validate_reaction(reaction)

    assert reaction.id == 5
    assert reaction.validation_errors == []
    assert reaction.is_valid == True


def test_valid_reaction_multi_reagant():
    reaction = parser.Reaction.parse_reaction(
        smiles="CCO>>CCN.CN",
        id=6,
    )

    reaction = validation.validate_reaction(reaction)

    assert reaction.id == 6
    assert reaction.validation_errors == []
    assert reaction.is_valid == True    










def test_invalid_reactant():
    reaction = parser.Reaction.parse_reaction(
        smiles="CiasaijdsaCO>>CC=O",
        id=7,
    )

    reaction = validation.validate_reaction(reaction)

    assert reaction.id == 7
    assert len(reaction.validation_errors) > 0
    assert reaction.is_valid == False

def test_invalid_product():
    reaction = parser.Reaction.parse_reaction(
        smiles="CCO>>CCiasaijdsa=O",
        id=8,
    )

    reaction = validation.validate_reaction(reaction)

    assert reaction.id == 8
    assert len(reaction.validation_errors) > 0
    assert reaction.is_valid == False

def test_invalid_reagant():
    reaction = parser.Reaction.parse_reaction(
        smiles="CCO>iasaijdsa>CC=O",
        id=9,
    )

    reaction = validation.validate_reaction(reaction)

    assert reaction.id == 9
    assert len(reaction.validation_errors) > 0
    assert reaction.is_valid == False




def test_reaction_map_nums():
    reaction = parser.Reaction(
        id=1,
        reactants_raw="[C:1]=[O:2]",
        reagants_raw="[C:1]=[O:2]",
        products_raw="[CH2:1][OH:2]",
    )

    reactant_maps, reagant_maps, product_maps = validation.get_map_nums_reaction(reaction)

    assert reactant_maps == {1, 2}
    assert reagant_maps  == {1, 2}
    assert product_maps  == {1, 2}


def test_reaction_differenc_map_nums_symmetry():
    reaction = parser.Reaction(
        id=1,
        reactants_raw="[C:1]=[O:2]",
        reagants_raw="[C:1]=[O:2]",
        products_raw="[CH2:1][OH:2]",
    )

    result = validation.reaction_map_differences(reaction)

    assert result["reactant_only"] == set()
    assert result["product_only"] == set()
    assert result["shared_product_reactant"] == {1, 2}



def test_reaction_differenc_map_nums_asymmetry():
    reaction = parser.Reaction(
        id=1,
        reactants_raw="[C:1]=[O:2]",
        reagants_raw="[C:1]=[O:2]",
        products_raw="[CH2:1][OH:3]",
    )

    result = validation.reaction_map_differences(reaction)

    assert result["reactant_only"] == {2}
    assert result["product_only"] == {3}
    assert result["shared_product_reactant"] == {1}


def test_smiles_molecule_bonds():
    reaction = parser.Reaction(
        id=1,
        reactants_raw="[C:1]=[O:2]",
        reagants_raw="[C:1][O:2].",
        products_raw="[CH2:1][OH:3]",
    )

    changes = validation.reaction_bonds_changes(reaction)
    print(changes[0])

    assert changes[0]["changed"] == {(1, 2): ("DOUBLE", "SINGLE")}
    assert changes[1]["changed"] == {}
    assert changes[2]["changed"] == {}

    assert changes[0]["formed"] == {}
    assert changes[1]["formed"] == {(1, 3): ("SINGLE")}
    assert changes[2]["formed"] == {(1, 3): ("SINGLE")}

    assert changes[0]["broken"] == {}
    assert changes[1]["broken"] == {(1, 2): ("SINGLE")}
    assert changes[2]["broken"] == {(1, 2): ("DOUBLE")}

    
def test_smiles_molecule_bonds():
   bonds = validation.smiles_molecule_bonds("Br[c:1]1[n:2][cH:3][cH:4][cH:5][n:6]1")

   assert (0,1) not in bonds
   assert all(0 not in bond for bond in bonds)