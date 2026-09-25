ALIPHATIC = [ #first
    "B", "C", "N", "O", "S",
    "P", "F", "Cl", "Br", "I"
]

AROMATIC = [ # second 
    "b", "c", "n", "o",
    "p", "s", "se", "as"
]

SYMBOLS = [ # forth
    "[", "]",
    "(", ")",
    "@", "@@",
    "+", "-",
    "=", "#", "$",
    "/", "\\",
    ":",
    "%",
    ".",
    "*"
]

NUMBERS = [ # fifth
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9"
]

ELEMENTS = [ # third
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
    "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
    "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
    "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg",
    "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]


# Remove elements that are already represented
# by ALIPHATIC

ELEMENTS = [
    element
    for element in ELEMENTS
    if element not in ALIPHATIC
]

WHOLE = ALIPHATIC + AROMATIC + SYMBOLS + NUMBERS + ELEMENTS 

class SmilesReader:

    def __init__(self, smiles):
        super().__init__()

        self.s = smiles
        self.i = 0
        self.tokens = []

    def peek(self, n=1):
        return self.s[self.i:self.i + n]

    def advance(self, n=1):
        thing = self.s[self.i:self.i + n]
        self.i += n
        return thing

    def read_atom(self):
        if self.peek().isupper():

            if (self.peek(2)) in ELEMENTS:
                self.tokens.append(
                    len(ALIPHATIC) + 
                    len(AROMATIC) +
                    ELEMENTS.index(self.peek(2)))
                self.advance(2)
                return

            elif (self.peek()) in ELEMENTS:
                self.tokens.append(
                    len(ALIPHATIC) + 
                    len(AROMATIC) +
                    ELEMENTS.index(self.peek()))
                self.advance()
                return


            if (self.peek(2)) in ALIPHATIC:
                self.tokens.append(ALIPHATIC.index(self.peek(2)))
                self.advance(2)
                return
            elif (self.peek()) in ALIPHATIC:
                self.tokens.append(ALIPHATIC.index(self.peek()))
                self.advance()
                return

            

                    
        else:
            if (self.peek(2)) in AROMATIC:
                self.tokens.append(
                    len(ALIPHATIC) + 
                    AROMATIC.index(self.peek(2)))
                self.advance(2)
                return


            if self.peek() in AROMATIC:
                self.tokens.append(
                    len(ALIPHATIC) + 
                    AROMATIC.index(self.peek()))
                self.advance()
                return

            
            if (self.peek(2)) in SYMBOLS:
                self.tokens.append(
                    len(ALIPHATIC) + 
                    len(AROMATIC) + 
                    len(ELEMENTS) +
                    SYMBOLS.index(self.peek(2)))
                self.advance(2)
                return

            if self.peek() in SYMBOLS:
                self.tokens.append(
                    len(ALIPHATIC) + 
                    len(AROMATIC) + 
                    len(ELEMENTS) +
                    SYMBOLS.index(self.peek()))
                self.advance()
                return


            if self.peek() in NUMBERS:
                self.tokens.append(
                    len(ALIPHATIC) + 
                    len(AROMATIC) + 
                    len(ELEMENTS) +
                    len(SYMBOLS) +
                    NUMBERS.index(self.peek()))
                self.advance()
                return

        raise ValueError(
        f"Unknown token at position "
        f"{self.i}: {self.peek()}"
        )


    def tokenise(self):

        self.tokens = []
        self.i = 0

        while self.i < len(self.s):
            self.read_atom()

        return self.tokens


    @staticmethod
    def detokenise(smiles_list):
        smiles = ""
        for i in smiles_list:
           smiles += WHOLE[i]

        return smiles   



    @staticmethod
    def vocab_size():

        return (
            len(ALIPHATIC)
            + len(AROMATIC)
            + len(ELEMENTS)
            + len(SYMBOLS)
            + len(NUMBERS)
        )


