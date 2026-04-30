"""Data loading, batching, and tokenizers for SMILES and IUPAC sequences."""
import re 

# Taken from Schwaller et al. 2017 regex— the standard SMILES tokenizer
SMILES_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|"
    r"\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

# Tokenizer 
def tokenize_smiles(s: str) -> list[str]:
    # Split a SMILES string into tokens that I can use
    return SMILES_PATTERN.findall(s) 

# This removes the multi-character atoms 

print(tokenize_smiles("CC(=O)Oc1ccccc1C(=O)O"))
# Expected: ['C', 'C', '(', '=', 'O', ')', 'O', 'c', '1', 'c', 'c', 'c', 'c', 'c', '1', 'C', '(', '=', 'O', ')', 'O']

print(tokenize_smiles("[C@@H]1CC[C@H](Cl)CC1"))
# Expected: ['[C@@H]', '1', 'C', 'C', '[C@H]', '(', 'Cl', ')', 'C', 'C', '1']
