ALIPHATIC = [
    "B", "C", "N", "O", "S",
    "P", "F", "Cl", "Br", "I"
]

AROMATIC = [
    "b", "c", "n", "o",
    "p", "s", "se", "as"
]

SYMBOLS = [
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

NUMBERS = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9"
]

ELEMENTS = [
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

        # --------------------------------
        # Check for '['
        # --------------------------------

        if self.peek() == "[":

            self.advance(1)

            # Check two-character atoms first
            two = self.peek(2)

            if two in ALIPHATIC:
                self.tokens.append(
                    ALIPHATIC.index(two)
                )
                self.advance(2)
                return

            if two in ELEMENTS:
                self.tokens.append(
                    len(ALIPHATIC)
                    + len(AROMATIC)
                    + ELEMENTS.index(two)
                )
                self.advance(2)
                return

            # Check one-character atoms
            one = self.peek()

            if one in ALIPHATIC:
                self.tokens.append(
                    ALIPHATIC.index(one)
                )
                self.advance(1)
                return

            if one in ELEMENTS:
                self.tokens.append(
                    len(ALIPHATIC)
                    + len(AROMATIC)
                    + ELEMENTS.index(one)
                )
                self.advance(1)
                return

            raise ValueError(
                f"Unknown atom inside '[' "
                f"at position {self.i}: {self.peek()}"
            )

        # --------------------------------
        # Check symbols
        # --------------------------------

        two = self.peek(2)

        if two in SYMBOLS:
            self.tokens.append(
                len(ALIPHATIC)
                + len(AROMATIC)
                + len(ELEMENTS)
                + SYMBOLS.index(two)
            )
            self.advance(2)
            return

        one = self.peek()

        if one in SYMBOLS:
            self.tokens.append(
                len(ALIPHATIC)
                + len(AROMATIC)
                + len(ELEMENTS)
                + SYMBOLS.index(one)
            )
            self.advance(1)
            return

        # --------------------------------
        # Check numbers
        # --------------------------------

        if one in NUMBERS:
            self.tokens.append(
                len(ALIPHATIC)
                + len(AROMATIC)
                + len(ELEMENTS)
                + len(SYMBOLS)
                + NUMBERS.index(one)
            )
            self.advance(1)
            return

        # --------------------------------
        # Check normal atoms
        # --------------------------------

        two = self.peek(2)

        if two in ALIPHATIC:
            self.tokens.append(
                ALIPHATIC.index(two)
            )
            self.advance(2)
            return

        if two in AROMATIC:
            self.tokens.append(
                len(ALIPHATIC)
                + AROMATIC.index(two)
            )
            self.advance(2)
            return

        one = self.peek()

        if one in ALIPHATIC:
            self.tokens.append(
                ALIPHATIC.index(one)
            )
            self.advance(1)
            return

        if one in AROMATIC:
            self.tokens.append(
                len(ALIPHATIC)
                + AROMATIC.index(one)
            )
            self.advance(1)
            return

        # --------------------------------
        # Nothing matched
        # --------------------------------

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

    def vocab_size(self):

        return (
            len(ALIPHATIC)
            + len(AROMATIC)
            + len(ELEMENTS)
            + len(SYMBOLS)
            + len(NUMBERS)
        )


