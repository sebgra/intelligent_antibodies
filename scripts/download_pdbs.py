import os
import urllib.request
import pandas as pd

"""
Fetch the fasta files for the proteins and complexes in the SAbDab database.
Data is fetched from the RCSB PDB database and stored in the data/SAbDab/fasta folder.
"""

root_url = "https://www.rcsb.org/fasta/entry"

def fetch_fasta(pdb_id : str) -> None:
    """
    Fetch the fasta file for a given pdb_id and save it.

    Parameters
    ----------
    pdb_id : str
        PDB id of the protein or complex to fetch

    """
    unfetched = open('data/SAbDab/unfetched.txt', 'a')

    try:
        with urllib.request.urlopen(f"{root_url}/{pdb_id.upper()}") as response:
            html = response.read()
            content = html.decode('utf-8')
        
            with open(f'data/SAbDab/fasta/all_samples/{pdb_id}.fasta', 'w') as f:
                f.write(content)
    except:
        print(f"{pdb_id} not found")
        unfetched.write(pdb_id)

    unfetched.close()
        

    return None

def control_fetch(pdb_id : str, folder: str) -> None:
    """
    Check if the fasta file for a given pdb_id is already fetched.

    Parameters
    ----------
    pdb_id : str
        PDB id of the protein or complex to fetch
    folder : str
        Path to the folder where the fasta files are stored
    """    
    if not os.path.exists(f"{folder}/{pdb_id}.fasta"):
        print(f"{folder}/{pdb_id}.fasta")
        print(f"{pdb_id} file not found")
    


def main() -> None:

    all_pdb_ids = pd.read_table('data/SAbDab/All_PDB_files.txt', header=None)
    positive_pdb_ids = pd.read_table('data/SAbDab/positive_samples.txt', header=None)

    test_id = all_pdb_ids[0][0].split('_')[0]
    print(f"Number of structures to fetch : {len(all_pdb_ids)}")

    for i in range(len(all_pdb_ids)):
        test_id = all_pdb_ids[0][i].split('_')[0]
        fetch_fasta(test_id)
        


if __name__ == "__main__":
    main()