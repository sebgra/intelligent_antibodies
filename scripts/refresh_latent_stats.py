"""
Recompute the VAE's latent statistics for an already-trained model.

`train_vae.py` writes these at the end of a run, but the per-chain-type
centres were added later, and retraining a working model purely to obtain
them would change its weights (GPU kernels are not bit-deterministic) and
invalidate every figure measured from it. This reads the saved encoder
instead and rewrites the stats file in place.

Usage
-----
    uv run python scripts/refresh_latent_stats.py --vector-size 250
"""
import argparse
import json

import numpy as np
import pandas as pd

from intelligent_antibodies.modules.dataset import AntibodyOneHotProtDataset
from intelligent_antibodies.modules.main_pipeline import load_models
from intelligent_antibodies.modules.utils.chains import classify_chains
from intelligent_antibodies.modules.utils.gpu import configure_compute
from intelligent_antibodies.modules.utils.paths import SABDAB_DIR, vae_latent_stats_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-size", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    configure_compute()
    models = load_models(args.vector_size)

    x_data, _y, vector_size, _alphabet = AntibodyOneHotProtDataset.get_data(args.vector_size)
    x_data = x_data.reshape(*x_data.shape, 1).astype("float32")

    # The chain labels must line up with the rows the dataset returns, so read
    # the same table through the same filter rather than re-deriving it.
    table = pd.read_csv(SABDAB_DIR / "sequences.csv", sep=";")
    table[["seq_rcpb", "seq_type"]] = table["seq_id"].str.split("|", n=1, expand=True)
    sequences = table[table["seq_type"] == "ab"]["sequence"].tolist()
    assert len(sequences) == len(x_data), (
        f"{len(sequences)} sequences against {len(x_data)} encoded rows"
    )
    labels = classify_chains(sequences)
    counted = {name: labels.count(name) for name in ("heavy", "light", None)}
    print(f"Chain types: {counted}")

    stats = models["vae_full"].vae.latent_statistics(
        x_data, batch_size=args.batch_size, labels=labels
    )
    stats["vector_size"] = int(vector_size)

    path = vae_latent_stats_path(vector_size)
    previous = json.loads(path.read_text()) if path.exists() else {}
    stats = {**previous, **stats}          # keep beta and anything else recorded
    path.write_text(json.dumps(stats, indent=2))

    print(f"Wrote {path}")
    for name, group in stats.get("by_chain_type", {}).items():
        separation = np.linalg.norm(
            np.asarray(group["mean"]) - np.asarray(stats["mean"])
        )
        print(f"  {name:<6} n={group['n_samples']:<5} "
              f"centre {separation:.2f} away from the global centre")


if __name__ == "__main__":
    main()
