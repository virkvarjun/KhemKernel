"""Stream a SMILES↔IUPAC dataset from Hugging Face, filter, and save as parquet."""
from datasets import load_dataset
import pandas as pd
import os

# Configuration
TARGET_SIZE = 1_000_000 # Number of samples from the dataset to save
MAX_SMILES_LEN = 100 # Dataset contains some very long SMILES - containing to len of 100
MAX_IUPAC_LEN = 100
OUTPUT_PATH = "data/raw_pairs.parquet"

def main():
    os.makedirs("data", exist_ok=True) # Create the data folder
    ds = load_dataset(
        "hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC",
        split="train",
        streaming=True,
    ) # Avoid downloading the entire dataset at once
    records = []
    seen = 0

    for example in ds: # Iterate one row at a time
        seen += 1
        smiles = example.get("SMILES_Canonical") # Gives every molecule one unique string representation (i.e CCO = OCC)
        iupac = example.get("iupac")
        if not smiles or not iupac:
            continue # Skip if either is missing
        if "." in smiles:
            continue # Skip mixtures (multiple molecules)
        if len(smiles) > MAX_SMILES_LEN or len(iupac) > MAX_IUPAC_LEN:
            continue # Skip very long examples form the max that I've defined above

        records.append({"SMILES": smiles, "IUPAC": iupac})

        if len(records) >= TARGET_SIZE:
            break # Stop once we have enough samples

        if seen % 100_000 == 0:
            print(f"Streamed {seen:,} | Kept: {len(records):,}")
    df = pd.DataFrame(records) # Convert to DataFrame for easier manipulation
    df = df.drop_duplicates(subset=["SMILES"]) # Remove duplicates
    df.to_parquet(OUTPUT_PATH, index=False) # Save as parquet for efficient storage and loading

    print(f"\nFinal: {len(df):,} pairs saved to {OUTPUT_PATH}")# Print the final count of pairs saved and the output path
    print(f"Disk size: {os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB") # Print the size of the saved file in megabytes


if __name__ == "__main__": # Run the main function when this script is executed
    main()
