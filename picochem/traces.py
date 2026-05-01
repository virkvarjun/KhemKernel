"""Generate structured chemistry reasoning traces from (SMILES, IUPAC) pairs."""

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # suppress RDKit warnings


# --- Parent structure detection ---

PARENT_PATTERNS = [
    # (name, SMARTS pattern, atom count for tie-breaking)
    ("naphthalene", "c1ccc2ccccc2c1", 10),
    ("indole",      "c1ccc2[nH]ccc2c1", 9),
    ("quinoline",   "c1ccc2ncccc2c1", 10),
    ("pyrimidine",  "c1cnccn1", 6),
    ("pyridine",    "c1ccncc1", 6),
    ("imidazole",   "c1cnc[nH]1", 5),
    ("pyrrole",     "c1cc[nH]c1", 5),
    ("furan",       "c1ccoc1", 5),
    ("thiophene",   "c1ccsc1", 5),
    ("benzene",     "c1ccccc1", 6),
    ("cyclohexane", "C1CCCCC1", 6),
    ("cyclopentane","C1CCCC1", 5),
]

_PARENT_MOLS = [(name, Chem.MolFromSmarts(smarts), size) for name, smarts, size in PARENT_PATTERNS]


def detect_parent(mol: Chem.Mol) -> str:
    """Return the name of the largest matched parent structure, or 'chain'."""
    best_name = None
    best_size = 0
    for name, pattern, size in _PARENT_MOLS:
        if mol.HasSubstructMatch(pattern) and size > best_size:
            best_name = name
            best_size = size
    if best_name is None:
        n_carbons = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
        return f"chain_C{n_carbons}"
    return best_name


# --- Functional group detection ---

FUNC_GROUP_PATTERNS = [
    ("carboxylic_acid", "C(=O)[OH]"),
    ("ester",           "C(=O)O[C,c]"),
    ("amide",           "C(=O)[NX3]"),
    ("nitrile",         "C#N"),
    ("aldehyde",        "[CX3H1](=O)"),
    ("ketone",          "[C,c][CX3](=O)[C,c]"),
    ("alcohol",         "[CX4][OH]"),
    ("phenol",          "[c][OH]"),
    ("ether",           "[C,c]O[C,c]"),
    ("amine",           "[NX3;H2,H1;!$(NC=O)]"),
    ("nitro",           "[N+](=O)[O-]"),
    ("halide",          "[F,Cl,Br,I]"),
    ("sulfonamide",     "S(=O)(=O)N"),
]

_FUNC_GROUP_MOLS = [(name, Chem.MolFromSmarts(smarts)) for name, smarts in FUNC_GROUP_PATTERNS]


def detect_functional_groups(mol: Chem.Mol) -> list[str]:
    """Return list of functional group names present in the molecule."""
    found = []
    for name, pattern in _FUNC_GROUP_MOLS:
        if mol.HasSubstructMatch(pattern):
            found.append(name)
    return found


# --- Trace assembly ---

def build_trace(smiles: str, iupac: str) -> str | None:
    """Build a structured chemistry trace for a molecule.
    
    Returns None if the SMILES is invalid (caller should skip).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    parent = detect_parent(mol)
    groups = detect_functional_groups(mol)
    n_atoms = mol.GetNumHeavyAtoms()
    n_rings = mol.GetRingInfo().NumRings()
    
    groups_str = ";".join(groups) if groups else "none"
    
    trace = (
        f"<parent>{parent}</parent>"
        f"<groups>{groups_str}</groups>"
        f"<atoms>{n_atoms}</atoms>"
        f"<rings>{n_rings}</rings>"
        f"<name>{iupac}</name>"
    )
    return trace