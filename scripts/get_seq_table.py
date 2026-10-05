"""
Build the sequence table from the downloaded FASTA files.

Reads every data/SAbDab/fasta/all_samples/<pdb_id>.fasta (see
download_pdbs.py) and writes data/SAbDab/sequences.csv with one row per
chain: its id (suffixed |ab or |ag depending on whether the chain reads as
an antibody or an antigen), the source organism, and the residue sequence.

Usage
-----
    uv run python scripts/get_seq_table.py
"""
from Bio import SeqIO
from tqdm import tqdm

from intelligent_antibodies.modules.utils.paths import SABDAB_DIR
from intelligent_antibodies.modules.utils.sabdab import get_pdb_id, is_antibody_molecule

FASTA_DIR = SABDAB_DIR / "fasta" / "all_samples"


def iter_chain_rows(pdb_id: str):
    """Yield (seq_id, specie, sequence) rows for every chain in one structure's FASTA file."""
    fasta_path = FASTA_DIR / f"{pdb_id}.fasta"
    if not fasta_path.exists():
        return
    for record in SeqIO.parse(fasta_path, "fasta"):
        name, _chains, molecule, specie = record.description.split("|")
        suffix = "ab" if is_antibody_molecule(molecule) else "ag"
        seq_id = f"{name.split('_')[0].lower()}|{suffix}"
        yield seq_id, specie, str(record.seq)


def main() -> None:
    all_entries = (SABDAB_DIR / "All_PDB_files.txt").read_text().splitlines()
    pdb_ids = sorted({get_pdb_id(entry) for entry in all_entries if entry.strip()})

    out_path = SABDAB_DIR / "sequences.csv"
    n_rows, n_missing = 0, 0
    with open(out_path, "w") as f_out:
        f_out.write("seq_id;specie;sequence\n")
        for pdb_id in tqdm(pdb_ids, desc="Extracting sequences", unit="structure"):
            if not (FASTA_DIR / f"{pdb_id}.fasta").exists():
                n_missing += 1
                continue
            for seq_id, specie, sequence in iter_chain_rows(pdb_id):
                f_out.write(f"{seq_id};{specie};{sequence}\n")
                n_rows += 1

    print(f"Wrote {n_rows} sequence row(s) to {out_path}.")
    if n_missing:
        print(f"{n_missing} structure(s) had no FASTA file on disk and were skipped "
              f"(run scripts/download_pdbs.py first to fetch them).")


if __name__ == "__main__":
    main()
