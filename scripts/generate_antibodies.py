"""
Generate antibody candidates for a target antigen using the trained models.

Thin CLI wrapper around `intelligent_antibodies.modules.main_pipeline`: loads
the trained VAE + Siamese models from run/models/, samples candidate
antibody sequences for the given antigen, and writes the ones that clear the
interaction-score threshold to a FASTA file.

Usage
-----
    uv run python scripts/generate_antibodies.py --antigen-id 6xe1
    uv run python scripts/generate_antibodies.py --antigen-id 6xe1 \\
        --n-candidates 30 --threshold 0.85 --temperature 1.2

Requires trained weights under run/models/vae/ and run/models/siamese/ --
see README.md, "Training the models".
"""
import argparse

from intelligent_antibodies.modules.main_pipeline import load_antigen_sequence, load_models, run_pipeline
from intelligent_antibodies.modules.utils.paths import RUN_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antigen-id", default="6xe1",
                         help="SAbDab seq_id (without |ag) of the target antigen.")
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=20,
                         help="Number of 10-candidate generation batches to attempt.")
    parser.add_argument("--vector-size", type=int, default=200)
    parser.add_argument("--out", default=None,
                         help="Output FASTA path (default: run/antibody-sequences-<antigen-id>.fasta)")
    args = parser.parse_args()

    antigen_sequence = load_antigen_sequence(args.antigen_id)
    print(f"Loaded antigen {args.antigen_id!r} ({len(antigen_sequence)} aa).")

    print("Loading trained models from run/models/...")
    models = load_models(args.vector_size)

    candidates = run_pipeline(
        antigen_sequence, models=models, vector_size=args.vector_size,
        n_candidates=args.n_candidates,
        threshold=args.threshold, temperature=args.temperature, limit=args.limit,
    )
    print(f"Generated {len(candidates)} candidate(s) above threshold {args.threshold}.")

    out_path = args.out or (RUN_DIR / f"antibody-sequences-{args.antigen_id}.fasta")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, c in enumerate(candidates):
            f.write(f">{args.antigen_id}_{i} score={c['score']:.3f} z=({c['z1']:.2f},{c['z2']:.2f})\n")
            f.write(f"{c['sequence']}\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
