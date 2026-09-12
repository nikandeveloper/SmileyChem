import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import parser

def test_two_component_reaction():
    reaction = parser.Reaction.parse_reaction(
        smiles="CCO>>CC=O",
        id=1,
    )

    assert reaction.id == 1
    assert reaction.reactants_raw == "CCO"
    assert reaction.reagants_raw == ""
    assert reaction.products_raw == "CC=O"

def test_three_component_reaction():
    reaction = parser.Reaction.parse_reaction(
        smiles="CCO>[NaOH]>CC=O",
        id=2,
    )

    assert reaction.id == 2
    assert reaction.reactants_raw == "CCO"
    assert reaction.reagants_raw == "[NaOH]"
    assert reaction.products_raw == "CC=O"

def test_invalid_component_reaction():
    reaction = parser.Reaction.parse_reaction(
        smiles="CCO>[NaOH]>hello>CC=O",
        id=3,
    )

    assert reaction.id == 3
    assert reaction.reactants_raw == "CCO>[NaOH]>hello>CC=O"
    assert reaction.reagants_raw ==  None
    assert reaction.products_raw ==  None
    assert reaction.is_valid == False
    assert len(reaction.validation_errors) == 1
    