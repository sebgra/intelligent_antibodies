import os
from time import sleep
import urllib.request
import pandas as pd
# import bs4


test_url = "http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/summary.php?pdb="
root_url = "https://www.rcsb.org/fasta/entry"

def fetch_fasta(pdb_id):
    unfetched = open('data/SAbDab/unfetched.txt', 'w')

    try:
        with urllib.request.urlopen(f"{root_url}/{pdb_id.upper()}") as response:
            html = response.read()
            content = html.decode('utf-8')

            with open(f'data/SAbDab/fasta/all_samples/{pdb_id}.fasta', 'w') as f:
                f.write(content)

    except:
        print(f'Failed to fetch {pdb_id}')
        unfetched.write(pdb_id)

    unfetched.close()
        

    return None


def main():

    all_pdb_ids = pd.read_table('data/SAbDab/All_PDB_files.txt', header=None)
    positive_pdb_ids = pd.read_table('data/SAbDab/positive_samples.txt', header=None)
    print(all_pdb_ids.head())

    test_id = all_pdb_ids[0][0].split('_')[0]
    print(len(all_pdb_ids))

    for i in range(len(all_pdb_ids)):
        test_id = all_pdb_ids[0][i].split('_')[0]
        print(f'Fetching {test_id}')
        fetch_fasta(test_id)
        # sleep(1)





if __name__ == "__main__":
    main()