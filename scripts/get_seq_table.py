import os
import pandas as pd
from Bio import SeqIO

def get_pdb_id(pdb_id: str) -> str:
    """
    Get the pdb id from a fasta file name.

    Parameters
    ----------
    pdb_id : str
        Name of the fasta file

    Returns
    -------
    str
        PDB id of the protein or complex
    """
    return pdb_id.split('_')[0]

ab_prefixes = ["IMMUNOGLOBULIN", "scFv", "Fab", "HEAVY CHAIN", "LIGHT CHAIN",
               "heavy chain", "light chain","ANTIBODY", "SCFV", "FAB"]

if __name__ == "__main__":

    all_pdb_ids = pd.read_table('data/SAbDab/All_PDB_files.txt', header=1)
    print(all_pdb_ids.head())

    with open('./data/SAbDab/sequences.csv', 'w') as f:

        for pdb_id in all_pdb_ids:
            fasta_sequences = SeqIO.parse(open(f"data/SAbDab/fasta/all_samples/{get_pdb_id(pdb_id)}.fasta"),'fasta')
            for fasta in fasta_sequences:
                description, name, sequence = fasta.description, fasta.id, str(fasta.seq)

                name, chains, molecule, specie = description.split('|')
                for prefix in ab_prefixes:
                    if prefix in molecule:
                        f.write(f"{name.split('_')[0]}|ab;{chains};{sequence}\n")
                    
                    else:
                        f.write(f"{name.split('_')[0]}|ag;{chains};{sequence}\n")