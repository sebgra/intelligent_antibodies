"""
Build the antibody-antigen interaction label table.

Reads data/SAbDab/All_PDB_files.txt (every structure/complex) and
data/SAbDab/positive_samples.txt (pairs of structures known to form immune
complexes), and writes data/SAbDab/data.csv: every pair of structures,
labeled 1 if they're a positive pair, 0 otherwise.

As the README describes, a positive pair is reciprocal -- "4gms<TAB>2vir"
means 4gms's antigen chains can complex with 2vir's antibody chains *and*
vice versa -- so both orderings are checked against the positive set.

Usage
-----
    uv run python scripts/get_interaction_table.py
"""
import itertools

import pandas as pd
from tqdm import tqdm

from intelligent_antibodies.modules.utils.paths import SABDAB_DIR
from intelligent_antibodies.modules.utils.sabdab import get_pdb_id


def load_positive_pairs(path) -> set:
    """
    Read positive_samples.txt into a set of (pdb_id, pdb_id) tuples,
    containing both orderings of each pair since the relation is reciprocal.
    """
    pairs = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ab, ag = line.split("\t")
            a, b = get_pdb_id(ab), get_pdb_id(ag)
            pairs.add((a, b))
            pairs.add((b, a))
    return pairs


def main() -> None:
    all_entries = pd.read_table(SABDAB_DIR / "All_PDB_files.txt", header=None)[0].tolist()
    positive_pairs = load_positive_pairs(SABDAB_DIR / "positive_samples.txt")
    print(f"{len(all_entries)} structure entries, {len(positive_pairs) // 2} positive pair(s).")

    out_path = SABDAB_DIR / "data.csv"
    all_couples = list(itertools.combinations_with_replacement(all_entries, 2))

    with open(out_path, "w") as f_out:
        f_out.write("ab;ag;interaction\n")
        for ab_entry, ag_entry in tqdm(all_couples, desc="Labeling pairs", unit="pair"):
            ab, ag = get_pdb_id(ab_entry), get_pdb_id(ag_entry)
            interaction = int((ab, ag) in positive_pairs)
            f_out.write(f"{ab}|ab;{ag}|ag;{interaction}\n")

    print(f"Wrote {len(all_couples)} labeled pairs to {out_path}")


if __name__ == "__main__":
    main()
