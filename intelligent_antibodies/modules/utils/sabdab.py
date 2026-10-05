"""
Small helpers shared by the scripts that parse raw SAbDab files
(scripts/download_pdbs.py, get_interaction_table.py, filter_interaction_table.py,
get_seq_table.py). Centralized here instead of being redefined (sometimes
slightly differently) in each script.
"""
from typing import Iterable

# Substrings in a FASTA record's "molecule" field that mark it as an
# antibody chain (heavy/light chain, Fab, scFv, ...) rather than an antigen.
AB_MOLECULE_MARKERS = (
    "IMMUNOGLOBULIN", "scFv", "Fab", "HEAVY CHAIN", "LIGHT CHAIN",
    "heavy chain", "light chain", "ANTIBODY", "SCFV", "FAB",
)


def get_pdb_id(entry: str) -> str:
    """
    The 4-character RCSB PDB code from a chain-qualified SAbDab entry, e.g.
    ``get_pdb_id("4gms_J_N_E") == "4gms"``.
    """
    return entry.split("_")[0]


def is_antibody_molecule(molecule_description: str) -> bool:
    """True if a FASTA record's molecule description reads as an antibody chain."""
    return any(marker in molecule_description for marker in AB_MOLECULE_MARKERS)


def unique_pdb_ids(entries: Iterable[str]) -> list:
    """Deduplicated, order-preserving list of 4-character PDB ids from chain-qualified entries."""
    seen = set()
    ordered = []
    for entry in entries:
        pdb_id = get_pdb_id(entry)
        if pdb_id not in seen:
            seen.add(pdb_id)
            ordered.append(pdb_id)
    return ordered
