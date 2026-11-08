# Analysis figures

Scripts that inspect an **already-trained** model and write the figures used in
the write-up. Nothing here retrains anything: every figure is reproducible from
the committed 250 aa weights.

```bash
uv run python scripts/figures/latent_maps.py          # what the latent space organises by
uv run python scripts/figures/biophysics.py           # disulfides, domain extent, plausibility
uv run python scripts/figures/cdr_organisation.py     # framework / CDR agreement
uv run python scripts/figures/sampling_temperature.py # what the temperature knobs cost
```

Output goes to the article's image directory; override with `IA_FIGURE_DIR`.
Sampling is memoised under `.figure-cache/` (gitignored) so the scripts can be
re-run cheaply while iterating on a plot — delete it to resample.

`_common.py` holds the shared palette, the chain-anatomy helpers (germline
family assignment, cysteine anchors, framework-4 location) and the plausibility
profile.
