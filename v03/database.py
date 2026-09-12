import tokeniser
import validation
from canonicalisation import canonicalise_reaction_mapped, canonicalise_reaction_unmapped
from parser import Reaction
from dataclasses import dataclass, field

ignore = "id,class,reactants>reagents>production\n"

@dataclass
class Database:
  file_name: str
  ignore_line: str | None = None

  reactions: list[Reaction] = field(default_factory=list)

  mapping_nums: list[tuple[set[int], set[int], set[int]]] = field(default_factory=list)

  def load_data(self):
    with open(self.file_name, "r") as file:
      id_num = 0
      for line_raw in file:
       if line_raw != self.ignore_line:
        line = line_raw.split(",")[2]
        if line.strip():
            id_num += 1
        else:
            continue    

        reaction = Reaction.parse_reaction(id=line_raw.split(",")[0], smiles=line.rstrip("\n"))

        reaction = validation.validate_reaction(reaction)

        self.reactions.append(reaction)


  def canonicalise_mapped(self):
    self.reactions = [canonicalise_reaction_mapped(reaction) for reaction in self.reactions]


  def canonicalise_unmapped(self):
    self.reactions = [canonicalise_reaction_unmapped(reaction) for reaction in self.reactions]


  def tokenise(self):
    self.reactions = [reaction.tokenise() for reaction in self.reactions]


  def nums_mapping(self):
    self.mapping_nums = [validation.get_map_nums_reaction(reaction) for reaction in self.reactions]

