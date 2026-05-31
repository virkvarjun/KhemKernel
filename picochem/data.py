"""Data loading, batching, and tokenizers for SMILES and IUPAC sequences."""
import re 
import json 
import numpy as np 

# Taken from Schwaller et al. 2017 regex— the standard SMILES tokenizer
SMILES_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|"
    r"\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

# Tokenizer 
def tokenize_smiles(s: str) -> list[str]:
    """Split a SMILES string into tokens, keeping multi-character atoms intact.

    >>> tokenize_smiles("CC(=O)Oc1ccccc1C(=O)O")
    ['C', 'C', '(', '=', 'O', ')', 'O', 'c', '1', 'c', 'c', 'c', 'c', 'c', '1', 'C', '(', '=', 'O', ')', 'O']
    >>> tokenize_smiles("[C@@H]1CC[C@H](Cl)CC1")
    ['[C@@H]', '1', 'C', 'C', '[C@H]', '(', 'Cl', ')', 'C', 'C', '1']
    """
    return SMILES_PATTERN.findall(s)

# IUPAC tokenization.
# The pattern also recognises trace XML tags (<parent>, </groups>, etc.) as
# single tokens, and the semicolon used as group separator in traces.
IUPAC_PATTERN = re.compile(
    r"</?(?:parent|groups|atoms|rings|name)>"  # trace structure tags
    r"|\d+"                                     # numbers
    r"|[a-zA-Z_]+"                             # words + underscores (group names like carboxylic_acid)
    r"|[;\(\)\[\],\-\'\.]"                     # punctuation including trace separator ;
)

def tokenize_iupac(s: str) -> list[str]:
    """Split an IUPAC name (or trace) into tokens.

    >>> tokenize_iupac("2-acetyloxybenzoic acid")
    ['2', '-', 'acetyloxybenzoic', 'acid']
    >>> tokenize_iupac("(2R,3S)-3-amino-2-hydroxypropanoic acid")
    ['(', '2', 'R', ',', '3', 'S', ')', '-', '3', '-', 'amino', '-', '2', '-', 'hydroxypropanoic', 'acid']
    """
    return IUPAC_PATTERN.findall(s)

def load_vocab(path: str) -> tuple[dict[str, int], dict[int, str]]: 
    # Load a vocab from JSON and returns a tuple of (token_to_idx, idx_to_token) dictionaries
    with open(path) as f: 
        stoi = json.load(f) 
    itos = {idx: token for token, idx in stoi.items()}
    return stoi, itos
def encode_smiles(s: str, vocab: dict[str, int]) -> np.ndarray: 
    # Convert a SMILES string to a list of token indices using the provided vocabulary
    unk = vocab["<unk>"]
    ids = [vocab.get(tok, unk) for tok in tokenize_smiles(s)] 
    return np.array(ids, dtype=np.int32)
def encode_iupac(s: str, vocab: dict[str, int]) -> np.ndarray:
    # Convert an IUPAC name to an array of integer IDs, with start/end 
    unk = vocab["<unk>"]
    ids = [vocab["<start>"]] 
    ids.extend(vocab.get(tok, unk) for tok in tokenize_iupac(s)) 
    ids.append(vocab["<end>"])
    return np.array(ids, dtype=np.int32) 

def decode_smiles(ids: np.ndarray, vocab: dict[int, str]) -> str:
    # Convert integer IDs back to a SMILES string 
    return "".join(vocab.get(idx, "<unk>") for idx in ids)

def decode_iupac(ids: np.ndarray, vocab: dict[int, str]) -> str:
    # Convert integer IDs back to an IUPAC name, removing special tokens 
    tokens = [vocab.get(idx, "<unk>") for idx in ids if idx in vocab]
    tokens = [t for t in tokens if t not in ("<start>", "<end>", "<pad>")]
    return "".join(tokens)
