import os
import itertools
import pandas as pd
import numpy
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


if __name__ == "__main__":

    positive_couples = []
    all_pdb_ids = pd.read_table('data/SAbDab/All_PDB_files.txt', header=None)


    with open('data/SAbDab/positive_samples.txt', 'r') as f:
        for line in f:
            ab, ag = line.split('\t')
            positive_couples.append((get_pdb_id(ab), get_pdb_id(ag)))


    with open('data/SAbDab/data.csv', 'w') as f_out:
        f_out.write("ab;ag;interaction\n")
        all_couples = itertools.combinations_with_replacement(all_pdb_ids[0].tolist(), 2)
        for ab, ag in all_couples:
            ab, ag = get_pdb_id(ab), get_pdb_id(ag)

            if (ab, ag) in positive_couples:
                f_out.write(f"{ab}|ab;{ag}|ag;1\n")
            else:
                f_out.write(f"{ab}|ab;{ag}|ag;0\n")

    