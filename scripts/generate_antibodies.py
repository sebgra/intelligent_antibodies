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

import json

from intelligent_antibodies.modules.main_pipeline import load_antigen_sequence, load_models, run_pipeline
from intelligent_antibodies.modules.utils.paths import RUN_DIR, siamese_calibration_path


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
    parser.add_argument("--chain-type", choices=("heavy", "light"), default=None,
                         help="Generate only heavy or only light chains (default: a mixture).")
    parser.add_argument("--residue-temperature", type=float, default=0.0,
                         help="Above 0, sample each residue from the decoder's distribution "
                              "instead of taking the argmax -- more distinct candidates.")
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
        residue_temperature=args.residue_temperature,
        chain_type=args.chain_type,
    )
    print(f"Generated {len(candidates)} candidate(s) above threshold {args.threshold}.")

    if not candidates:
        # Say WHY nothing passed. The usual reason is a threshold the trained
        # classifier cannot reach at all, which no amount of sampling fixes.
        calibration_path = siamese_calibration_path(args.vector_size)
        if calibration_path.exists():
            try:
                calibration = json.loads(calibration_path.read_text())
            except (json.JSONDecodeError, OSError):
                calibration = None
            if calibration:
                print(
                    f"  This model scores held-out pairs in "
                    f"{calibration['score_min']:.2f}-{calibration['score_max']:.2f} "
                    f"(AUC {calibration['roc_auc']:.2f}); its calibrated operating point is "
                    f"--threshold {calibration['recommended_threshold']:.2f}."
                )
                if args.threshold > float(calibration["score_max"]):
                    print("  The requested threshold is above anything this model can produce.")
        else:
            print(f"  No calibration file at {calibration_path} -- retrain with "
                  "scripts/train_siamese.py to find out which thresholds are reachable.")

    out_path = args.out or (RUN_DIR / f"antibody-sequences-{args.antigen_id}.fasta")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, c in enumerate(candidates):
            f.write(f">{args.antigen_id}_{i} score={c['score']:.3f} z=({c['z1']:.2f},{c['z2']:.2f})\n")
            f.write(f"{c['sequence']}\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
