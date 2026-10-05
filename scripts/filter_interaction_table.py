"""
Filter data.csv down to pairs whose antibody *and* antigen chain both
actually have a sequence in sequences.csv (get_interaction_table.py labels
every combination of referenced structures, including ones that failed to
download or had no antibody/antigen chain parsed out by get_seq_table.py).

Writes data/SAbDab/data_filtered.csv, which is what `dataset.py`'s
`EquilibratedDataset` (and therefore model training) actually reads.

Previously this step only existed as a notebook (notebooks/filtering.ipynb);
this script is its scripted equivalent, run as part of the data pipeline:

    uv run python scripts/download_pdbs.py
    uv run python scripts/get_seq_table.py
    uv run python scripts/get_interaction_table.py
    uv run python scripts/filter_interaction_table.py
"""
import pandas as pd

from intelligent_antibodies.modules.utils.paths import SABDAB_DIR


def main() -> None:
    df_match = pd.read_csv(SABDAB_DIR / "data.csv", sep=";")
    df_seq = pd.read_csv(SABDAB_DIR / "sequences.csv", sep=";")

    known_ids = set(df_seq["seq_id"])
    df_filtered = df_match[df_match["ab"].isin(known_ids) & df_match["ag"].isin(known_ids)]

    out_path = SABDAB_DIR / "data_filtered.csv"
    df_filtered.to_csv(out_path, sep=";", index=False)

    print(f"{len(df_match)} labeled pairs -> {len(df_filtered)} with a known sequence on both "
          f"sides ({100 * len(df_filtered) / max(len(df_match), 1):.1f}%).")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
