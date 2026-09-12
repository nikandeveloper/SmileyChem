import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser import Reaction
from canonicalisation import canonicalise_mapped, canonicalise_unmapped, canonicalise_mapped_polymolecules, canonicalise_unmapped_polymolecules, canonicalise_reaction_mapped, canonicalise_reaction_unmapped

def test_valid_canonicalise_mapped():
    smiles = "CCO"

    canonical1 = canonicalise_mapped(smiles)
    canonical2 = canonicalise_mapped("".join(reversed(smiles)))

    assert canonical1 == canonical2

def test_invalid_canonicalise_mapped():
    smiles = "CCO"

    canonical1 = canonicalise_mapped(smiles)
    canonical2 = canonicalise_mapped("".join(reversed(smiles)) + "s")

    assert canonical1 != canonical2


def test_canonicalise_mapped_invalid_smiles():
    smiles = "ChekadCO"

    canonical1 = canonicalise_mapped(smiles)

    assert canonical1 is None


def test_sanity_canonicalise_mapped():
    smiles = "CCO"

    canonical1 = canonicalise_mapped(smiles)

    assert canonical1 is not None


def test_canonicalise_mapped():
    smiles = "[CH3:1][CH2:2][OH:3]"

    canonical = canonicalise_mapped(smiles)
    print(canonical)

    assert canonical is not None
    assert ":1" in canonical
    assert ":2" in canonical
    assert ":3" in canonical






def test_valid_canonicalise_unmapped():
    smiles = "[CH3:1][CH2:2][OH:3]"
    smiles_reverse = "[OH:3][CH2:2][CH3:1]"

    canonical1 = canonicalise_unmapped(smiles)
    canonical2 = canonicalise_unmapped(smiles_reverse)

    assert canonical1 == canonical2

def test_invalid_canonicalise_unmapped():
    smiles = "[CH3:1][CH2:2][OH:3]"
    smiles_reverse = "[OH:3][CH2:2][CH3:1]"

    canonical1 = canonicalise_unmapped(smiles)
    canonical2 = canonicalise_unmapped(smiles_reverse + "s")

    assert canonical1 != canonical2


def test_canonicalise_unmapped_invalid_smiles():
    smiles = "[CH3:1hello][CH2:2][OH:3]"

    canonical1 = canonicalise_unmapped(smiles)

    assert canonical1 is None


def test_sanity_canonicalise_unmapped():
    smiles = "[CH3:1][CH2:2][OH:3]"

    canonical1 = canonicalise_unmapped(smiles)

    assert canonical1 is not None


def test_canonicalise_unmapped():
    smiles = "[CH3:1][CH2:2][OH:3]"

    canonical = canonicalise_unmapped(smiles)
    print(canonical)

    assert canonical is not None
    assert ":1" not in canonical
    assert ":2" not in canonical
    assert ":3" not in canonical

def test_canonicalize_unmapped_ignores_mapping_numbers():
    smiles_a = "[CH3:1][CH2:2][OH:3]"
    smiles_b = "[CH3:7][CH2:8][OH:9]"

    canonical_a = canonicalise_unmapped(smiles_a)
    canonical_b = canonicalise_unmapped(smiles_b)

    assert canonical_a == canonical_b





def test_canonicalize_multiple_unmapped_molecules():
    result = canonicalise_unmapped_polymolecules("OCC.CN")

    assert result == "CCO.CN"


def test_canonicalize_multiple_mapped_molecules():
    result = canonicalise_mapped_polymolecules("[CH3:1][CH2:2][OH:3].[NH2:4][CH3:5]")

    assert result == "[CH3:1][CH2:2][OH:3].[NH2:4][CH3:5]"



def test_canonicalise_mapped_reaction():
    reaction = Reaction(
        id=12,
        reactants_raw= "[CH3:1][CH2:2][OH:3].CO",
        reagants_raw=  "O",
        products_raw=  "[CH3:7][CH2:8][OH:9]"
    )

    result = canonicalise_reaction_mapped(reaction)

    assert result is reaction
    assert result.reactants_canonical == "[CH3:1][CH2:2][OH:3].CO"
    assert "." in result.reactants_canonical
    assert result.reagants_canonical  == "O"
    assert result.products_canonical  == "[CH3:7][CH2:8][OH:9]"



def test_canonicalise_unmapped_reaction():
    reaction = Reaction(
        id=13,
        reactants_raw= "[CH3:1][CH2:2][OH:3].CO",
        reagants_raw=  "O",
        products_raw=  "[CH3:7][CH2:8][OH:9]"
    )

    result = canonicalise_reaction_unmapped(reaction)

    assert result is reaction
    assert ":" not in result.reactants_canonical
    assert "." in result.reactants_canonical
    assert ":" not in result.reagants_canonical
    assert ":" not in result.products_canonical

