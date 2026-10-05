"""
Fetch the FASTA file for every PDB structure/complex referenced in
data/SAbDab/All_PDB_files.txt from the RCSB PDB database, saving them under
data/SAbDab/fasta/all_samples/.

Safe to re-run: ids that already have a .fasta file on disk are skipped, so
an interrupted run can simply be restarted (use --force to re-fetch anyway).

Usage
-----
    uv run python scripts/download_pdbs.py                  # fetch everything
    uv run python scripts/download_pdbs.py --limit 20        # quick smoke test
"""
import argparse
import time
import urllib.error
import urllib.request

import pandas as pd
from tqdm import tqdm

from intelligent_antibodies.modules.utils.paths import SABDAB_DIR
from intelligent_antibodies.modules.utils.sabdab import unique_pdb_ids

ROOT_URL = "https://www.rcsb.org/fasta/entry"
FASTA_DIR = SABDAB_DIR / "fasta" / "all_samples"
UNFETCHED_PATH = SABDAB_DIR / "unfetched.txt"


def fetch_fasta(pdb_id: str, timeout: float = 15.0) -> bool:
    """
    Download one PDB id's FASTA file and save it to FASTA_DIR.

    Parameters
    ----------
    pdb_id : str
        4-character PDB id of the structure to fetch.
    timeout : float
        Seconds to wait for RCSB to respond before giving up.

    Returns
    -------
    bool
        True if the file was fetched and saved, False on any error.
    """
    try:
        with urllib.request.urlopen(f"{ROOT_URL}/{pdb_id.upper()}", timeout=timeout) as response:
            content = response.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"  {pdb_id}: fetch failed ({exc})")
        return False

    (FASTA_DIR / f"{pdb_id}.fasta").write_text(content)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only fetch the first N ids (handy for a quick test run).")
    parser.add_argument("--delay", type=float, default=0.2,
                         help="Seconds to wait between requests, to be polite to RCSB.")
    parser.add_argument("--force", action="store_true",
                         help="Re-fetch ids even if a .fasta file already exists for them.")
    args = parser.parse_args()

    FASTA_DIR.mkdir(parents=True, exist_ok=True)

    all_entries = pd.read_table(SABDAB_DIR / "All_PDB_files.txt", header=None)[0]
    pdb_ids = unique_pdb_ids(all_entries)
    if args.limit:
        pdb_ids = pdb_ids[: args.limit]

    todo = pdb_ids if args.force else [
        pdb_id for pdb_id in pdb_ids if not (FASTA_DIR / f"{pdb_id}.fasta").exists()
    ]
    print(f"{len(pdb_ids)} unique structure(s) referenced, {len(todo)} to fetch "
          f"({len(pdb_ids) - len(todo)} already present in {FASTA_DIR}).")

    unfetched = []
    for pdb_id in tqdm(todo, desc="Fetching FASTA files", unit="structure"):
        if not fetch_fasta(pdb_id):
            unfetched.append(pdb_id)
        time.sleep(args.delay)

    if unfetched:
        UNFETCHED_PATH.write_text("\n".join(unfetched) + "\n")
        print(f"{len(unfetched)} id(s) could not be fetched; see {UNFETCHED_PATH}")
    else:
        print("All requested structures fetched successfully.")


if __name__ == "__main__":
    main()
