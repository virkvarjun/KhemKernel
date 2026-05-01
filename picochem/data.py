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
    # Split a SMILES string into tokens that I can use
    return SMILES_PATTERN.findall(s) 

# This removes the multi-character atoms 

print(tokenize_smiles("CC(=O)Oc1ccccc1C(=O)O"))
# Expected: ['C', 'C', '(', '=', 'O', ')', 'O', 'c', '1', 'c', 'c', 'c', 'c', 'c', '1', 'C', '(', '=', 'O', ')', 'O']

print(tokenize_smiles("[C@@H]1CC[C@H](Cl)CC1"))
# Expected: ['[C@@H]', '1', 'C', 'C', '[C@H]', '(', 'Cl', ')', 'C', 'C', '1']

# IUPAC tokenization 
IUPAC_PATTERN = re.compile(r"\d+|[a-zA-Z]+|[\(\)\[\],\-\'\.]") 

def tokenize_iupac(s: str) -> list[str]:
    # Split an IUPAC string into tokens that I can use
    return IUPAC_PATTERN.findall(s)

print(tokenize_iupac("2-acetyloxybenzoic acid"))
# Expected: ['2', '-', 'acetyloxybenzoic', 'acid']

print(tokenize_iupac("(2R,3S)-3-amino-2-hydroxypropanoic acid"))
# Expected: ['(', '2', 'R', ',', '3', 'S', ')', '-', '3', '-', 'amino', '-', '2', '-', 'hydroxypropanoic', 'acid']

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

def get_batch(
    pairs: list[tuple[np.ndarray, np.ndarray]], 
    batch_size: int, 
    smiles_pad_id: int, 
    iupac_pad_id: int, 
    rng: np.random.Generator, ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Sample a batch (random) of (SMILES, IUPAC) paris and then p[ad them 
    indices = rng.choicce(len(pairs), size=batch_size, replace=False)
    batch = [pairs[i] for i in indices]
    smiles_max = max(len(s) for s, _ in batch) 
    iupac_max = max(len(t) for _, t in batch)
    
    smiles_ids = np.full((batch_size, smiles_max), smiles_pad_id, dtype=np.int32) 
    iupac_ids = np.full((batch_size, iupac_max), iupac_pad_id, dtype=np.int32)
    smiles_mask = np.zeros((batch_size, smiles_max), dtype=np.float32) 
    iupac_mask = np.zeros((batch_size, iupac_max), dtype=np.float32)

    for i, (s, t) in enumerate(batch):
        smiles_ids[i, :len(s)] = s 
        iupac_ids[i, :len(t)] = t 
        smiles_mask[i, :len(s)] = 1.0 
        iupac_mask[i, :len(t)] = 1.0
    
    return smiles_ids, smiles_mask, iupac_ids, iupac_mask
