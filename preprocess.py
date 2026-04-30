from datasets import load_dataset

# Load the entire dataset (warning: large)
ds = load_dataset("hheiden/PubChem-124M-Canonicalized-SELFIES-InChI-IUPAC")

# Streaming (Recommended for training)
ds_stream = load_dataset("hheiden/PubChem-124M-Canonicalized-SELFIES-InChI-IUPAC", streaming=True)
for sample in ds_stream['train']:
    print(sample['SMILES'], sample['SELFIES'])
