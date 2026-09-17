from dataclasses import dataclass, field
import torch
import tokeniser
import torch.nn.functional as f

@dataclass
class Reaction:
    id: int

    reactants_raw: str
    reagants_raw: str
    products_raw: str

    reactants_canonical: str | None = None
    reagants_canonical: str | None = None
    products_canonical: str | None = None

    mapped_reaction: str | None = None

    is_valid: bool = False
    validation_errors: list = field(default_factory=list)

    reactants_tokens: torch.Tensor = field(default_factory=torch.Tensor)
    reagants_tokens: torch.Tensor = field(default_factory=torch.Tensor)
    products_tokens: torch.Tensor = field(default_factory=torch.Tensor)

    padded_tokens: torch.Tensor = field(default_factory=torch.Tensor)

    max_len: int | None = None

    key: int | None = None


    def tokenise(self):
        reader_reactant = tokeniser.SmilesReader(self.reactants_canonical)
        reader_reagant = tokeniser.SmilesReader(self.reagants_canonical)
        reader_product = tokeniser.SmilesReader(self.products_canonical)

        self.reactants_tokens = torch.tensor(reader_reactant.tokenise())
        self.reagants_tokens = torch.tensor(reader_reagant.tokenise())
        self.products_tokens = torch.tensor(reader_product.tokenise())

        self.max_len = max(len(self.reactants_tokens), len(self.products_tokens))

        return self


    def pad(self, padding: int, size: int) -> torch.Tensor:
        self.reactants_tokens = f.pad(self.reactants_tokens, (0, size - len(self.reactants_tokens)), value=0)
        self.products_tokens = f.pad(self.products_tokens, (0, size - len(self.products_tokens)), value=0)

        return torch.stack([self.reactants_tokens, self.products_tokens])


    def CanonicaltoString(self) -> None | str:
        if self.reactants_canonical is not None:
            reac_can = self.reactants_canonical
        else:
            reac_can = ""    
        if (self.reagants_canonical is not None):
            reaga_can = self.reagants_canonical
        else:
            reaga_can = ""   

        if (self.products_canonical is not None):
            produ_can = self.products_canonical
        
        else:
            produ_can = ""   
        
        return (reac_can + ">" + reaga_can + ">" + produ_can)


    @classmethod
    def parse_reaction(cls, id, smiles) -> "Reaction":
        parts = smiles.split(">")

        if len(parts) == 3:
            
            reactants = parts[0]
            reagants = parts[1]
            products = parts[2]

        elif len(parts) == 2:
            
            reactants = parts[0]
            reagants = None
            products = parts[1]    

        else:
            error = f"Expected 2 or 3 components got {len(parts)}"
            reactants = smiles

            return cls(
                id        =  id,
                reactants_raw      =    reactants,
                reagants_raw       =         None,
                products_raw       =         None,
                is_valid           =        False,
                validation_errors  =      [error]
            )


        return cls (
                id        =  id,
                reactants_raw      =    reactants,
                reagants_raw       =     reagants,
                products_raw       =     products,
                validation_errors  =           []
        )

