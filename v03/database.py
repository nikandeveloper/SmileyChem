import tokeniser
import validation
from canonicalisation import canonicalise_reaction_mapped, canonicalise_reaction_unmapped
from parser import Reaction
from dataclasses import dataclass, field
import torch

ignore = "id,class,reactants>reagents>production\n"

@dataclass
class Database:
  file_name: str
  size_bucket: int
  size_batch: int
  ignore_line: str | None = None


  reactions: list[Reaction] = field(default_factory=list)

  bucket_dict_reactant: dict[int ,list[torch.Tensor]] = field(default_factory=dict)
  bucket_dict_product: dict[int ,list[torch.Tensor]] = field(default_factory=dict)

  len_bucket_dict: dict[int ,list[int]] = field(default_factory=dict)
  batch_avai_bucket_dict: dict[int ,list[int]] = field(default_factory=dict)

  mapping_nums: list[tuple[set[int], set[int], set[int]]] = field(default_factory=list)


  def load_data(self):
    with open(self.file_name, "r") as file:
      id_num = 0
      for line_raw in file:
       if id_num != 0:
        line = line_raw.split(",")[2]
        if line.strip():
            id_num += 1
        else:
            continue    

        reaction = Reaction.parse_reaction(id=line_raw.split(",")[0], smiles=line.rstrip("\n"))

        reaction = validation.validate_reaction(reaction)

        self.reactions.append(reaction)
       else:
        id_num += 1 
        continue


  def add_marking_codes(self, EOS, SOS):
    for key in self.bucket_dict_reactant.keys():
      for i, element in enumerate(self.bucket_dict_product[key]):
        self.bucket_dict_product[key][i] = torch.cat([
          element.new_full((element.size(0), 1), SOS),
          element,    
          element.new_full((element.size(0), 1), EOS)
        ], dim=1)
   

  def bucket_batch(self, padding: int):
    for reaction in self.reactions:
      key = self.index_bucket(reaction.max_len)
      reaction.key = key
      self.len_bucket_dict.setdefault(key, []).append(reaction.max_len)

    for reaction in self.reactions:
      reaction.padded_tokens = reaction.pad(padding, max(self.len_bucket_dict[reaction.key]))

      self.add_padded_tokens(reaction.key, reaction.padded_tokens)

      
  def add_padded_tokens(self, key, padded_tokens):
    if key in self.batch_avai_bucket_dict and self.batch_avai_bucket_dict[key]:
      if self.batch_avai_bucket_dict[key][-1] < self.size_batch:
        self.batch_avai_bucket_dict[key][-1] += 1
      else:
        self.batch_avai_bucket_dict[key].append(1)
    else:
      self.batch_avai_bucket_dict[key] = [1]

    if self.bucket_dict_reactant.get(key) is None:
      self.bucket_dict_reactant[key] = [padded_tokens[0].unsqueeze(0)] 

    elif len(self.bucket_dict_reactant[key]) < len(self.batch_avai_bucket_dict[key]):
      self.bucket_dict_reactant[key].append(padded_tokens[0].unsqueeze(0)) 

    else:  
      self.bucket_dict_reactant[key][(len(self.batch_avai_bucket_dict[key]) - 1)]  =  torch.cat([self.bucket_dict_reactant[key][(len(self.batch_avai_bucket_dict[key]) - 1)], padded_tokens[0].unsqueeze(0)], dim=0)


    if self.bucket_dict_product.get(key) is None:
      self.bucket_dict_product[key] = [padded_tokens[1].unsqueeze(0)]

    elif len(self.bucket_dict_product[key]) < len(self.batch_avai_bucket_dict[key]):
      self.bucket_dict_product[key].append(padded_tokens[1].unsqueeze(0))

    else:
      self.bucket_dict_product[key][(len(self.batch_avai_bucket_dict[key]) - 1)]   =  torch.cat([self.bucket_dict_product[key][(len(self.batch_avai_bucket_dict[key]) - 1)], padded_tokens[1].unsqueeze(0)], dim=0)



  def index_bucket(self, max_len: int) -> int:
    for bucket in self.len_bucket_dict:
      if abs(bucket - max_len) < self.size_bucket / 2:
        return bucket

    bucket = (max_len // self.size_bucket) * self.size_bucket
    if max_len % self.size_bucket >= self.size_bucket / 2:
      bucket += self.size_bucket

    return bucket  


  def canonicalise_mapped(self):
    self.reactions = [canonicalise_reaction_mapped(reaction) for reaction in self.reactions]


  def canonicalise_unmapped(self):
    self.reactions = [canonicalise_reaction_unmapped(reaction) for reaction in self.reactions]


  def tokenise(self):
    self.reactions = [reaction.tokenise() for reaction in self.reactions]


  def nums_mapping(self):
    self.mapping_nums = [validation.get_map_nums_reaction(reaction) for reaction in self.reactions]
