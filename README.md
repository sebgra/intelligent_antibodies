<div align="center">

# 🧬 Intelligent Antibodies

**Generate antibody sequences that never existed, then score them against a target antigen.**

A convolutional VAE learns what antibodies look like; a Siamese network predicts whether a candidate binds a given antigen. Use it from the command line or through an interactive dashboard.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/badge/managed%20with-uv-6340ac.svg)](https://docs.astral.sh/uv/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/status-research%20project-orange.svg)](#limitations)

[Quickstart](#quickstart) • [Dashboard](#using-the-dashboard) • [CLI](#using-the-cli) • [Training](#training-your-own-models) • [How it works](#how-it-works) • [Limitations](#limitations)

</div>

---

## What this is

Antibody design is slow and expensive, and still leans on in-vitro screening. This project explores the in-silico alternative at small scale: candidate discovery as **rejection sampling in a learned latent space** rather than a physics-based energy calculation.

Two models, trained independently:

| Model | Job | Sees |
|---|---|---|
| **VAE** (generator) | invent a plausible antibody chain | antibody chains only — never an antigen |
| **Siamese network** (judge) | predict whether a pair interacts | (antibody, antigen) pairs labelled 0/1 |

Sample from the generator, score with the judge, keep what clears a calibrated threshold.

> **This is a research/learning project, not a design tool.** The classifier is a useful filter (AUC 0.76), not an oracle, and nothing here is validated experimentally. See [Limitations](#limitations) before trusting any output.

---

## Quickstart

```bash
git clone https://github.com/sebgra/intelligent_antibodies.git
cd intelligent_antibodies

uv sync                           # Python 3.12 + every dependency
uv run streamlit run app/main.py  # opens the dashboard
```

No `uv`? `curl -LsSf https://astral.sh/uv/install.sh | sh` ([docs](https://docs.astral.sh/uv/getting-started/installation/)).

The dashboard runs **without trained weights** — it detects their absence and shows clearly-labelled *simulated* results so you can explore the interface. To generate for real, [train the models](#training-your-own-models) first.

---

## Using the dashboard

```bash
uv run streamlit run app/main.py
```

### 1. Generate tab

| Control | What it does |
|---|---|
| **Target antigen** | Paste a sequence, tick the bundled lysozyme example, or upload a FASTA file and pick a record |
| **Desired number of candidates** | A *target*, not a cap — sampling continues until this many distinct candidates clear the threshold |
| **Trained model** | Which encoding window to use. Only windows with both a VAE and a classifier are listed; the best-scoring one is pre-selected |
| **Interaction score threshold** | Bounded by what the model can actually produce, and pre-set to its calibrated operating point |
| **Sampling temperature** | Scales the latent draw. Low = conservative, high = exploratory |
| **Chain type** | `Either`, `Heavy only` or `Light only` — see [chain types](#heavy-or-light-chains) |
| **Residue sampling diversity** | `0` decodes greedily; above 0 samples each position from the decoder's own distribution |

Click **🚀 Launch generation**. If the budget runs out, the warning tells you the best score actually reached and which threshold would have worked — it never fails silently.

### 2. Results tab

- **Overview** — score distribution, tier breakdown, key statistics
- **Latent space** — where each candidate was sampled from, with box/lasso selection
- **Sequence analysis** — composition, hydrophobicity, net charge, colourised sequences
- **Candidate explorer** — per-candidate detail, a schematic secondary-structure cartoon, and **🔬 Fold with ESMFold** for a real predicted 3-D structure coloured by per-residue confidence (pLDDT)

### 3. Dataset & Model tab

The real SAbDab tables, the measured quality of each trained model, and the training curves (tucked into an expander).

---

## Using the CLI

Everything the dashboard does is scriptable.

### Generate candidates

```bash
uv run python scripts/generate_antibodies.py --antigen-id 6xe1 \
    --n-candidates 20 --threshold 0.91 --vector-size 250
```

Writes a FASTA file of passing candidates to `run/`. Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--antigen-id` | `6xe1` | SAbDab id (without `\|ag`), looked up in `sequences.csv` |
| `--n-candidates` | 20 | How many candidates to return |
| `--threshold` | 0.8 | Minimum interaction score. Use the model's calibrated value (printed at the end of training) |
| `--vector-size` | 200 | Encoding window — must match a trained pair |
| `--temperature` | 1.0 | Latent sampling spread |
| `--chain-type` | *(mixture)* | `heavy` or `light` |
| `--residue-temperature` | 0.0 | Above 0, sample residues instead of taking the argmax |
| `--out` | `run/…fasta` | Output path |

### The pipeline as a module

```bash
uv run python -m intelligent_antibodies.modules.main_pipeline \
    --antigen-id 6xe1 --n-candidates 20 --threshold 0.91 --chain-type heavy
```

### Or drive it yourself

```python
import pandas as pd
from intelligent_antibodies.modules.main_pipeline import load_models
from intelligent_antibodies.modules.utils.encoding import ProteinOneHotEncoder
from intelligent_antibodies.modules.utils.inference import (
    generate_antibody_sequence, test_interaction,
)

VECTOR_SIZE = 250
models = load_models(VECTOR_SIZE)          # VAE + classifier from run/models/
encoder = ProteinOneHotEncoder()

antigen = "KVFGRCELAAAMKRHGLANYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWC..."
onehot_antigen = encoder.encode(pd.Series([antigen]), VECTOR_SIZE)

for sequence, onehot, z, confidence in generate_antibody_sequence(
    n=10, vae_model=models["vae_full"].vae, encoder=encoder,
    vector_size=VECTOR_SIZE, temperature=1.0, chain_type="heavy",
):
    passed, score = test_interaction(onehot, onehot_antigen, models["siamese"],
                                     threshold=0.91)
    if passed:
        print(f"{score:.3f}  {sequence}")
```

### Heavy or light chains

An antibody pairs one heavy chain with one light chain. The VAE is trained on both without being told which is which — yet it separates them in its latent space anyway, so generation can be restricted to one family:

```bash
uv run python scripts/generate_antibodies.py --antigen-id 6xe1 --chain-type heavy
```

Measured over 120 candidates per mode on the 250 aa model:

| Asked for | Heavy | Light | No clear signature |
|---|---|---|---|
| *(unrestricted)* | 48% | 22% | 30% |
| `--chain-type heavy` | **86%** | 4% | 10% |
| `--chain-type light` | 3% | **69%** | 28% |

The family centres live in the VAE's latent-stats file. For a model trained before this existed, `scripts/refresh_latent_stats.py --vector-size 250` adds them **without retraining**.

---

## Training your own models

### 1. Build the dataset

Four scripts, in order, from the repo root:

```bash
uv run python scripts/download_pdbs.py            # fetch FASTA files from RCSB
uv run python scripts/get_seq_table.py            # -> data/SAbDab/sequences.csv
uv run python scripts/get_interaction_table.py    # -> data/SAbDab/data.csv
uv run python scripts/filter_interaction_table.py # -> data/SAbDab/data_filtered.csv
```

| Script | What it does |
|---|---|
| `download_pdbs.py` | Fetches a FASTA per structure listed in `All_PDB_files.txt`. Resumable; `--limit 20` for a smoke test |
| `get_seq_table.py` | One row per chain, tagged `\|ab` or `\|ag` |
| `get_interaction_table.py` | Labels every pair 1 (interacting) or 0, from `positive_samples.txt` |
| `filter_interaction_table.py` | Drops pairs whose sequence is missing — this is what training reads |

### 2. Train

The two models are independent and can be trained in either order.

```bash
uv run python scripts/train_vae.py     --vector-size 250 --latent-dim 16 --epochs 200
uv run python scripts/train_siamese.py --vector-size 250 --filters 48 --epochs 80 \
    --negative-ratio 1 --split-by antibody --seed 42 --monitor val_mcc
```

Both print a per-epoch progress bar with the metric that matters — held-out `val_mcc`, `val_f1` and `val_auc` for the classifier, exact-residue accuracy for the VAE — and stop early on the best epoch.

**Why these settings.** They were chosen by experiment, not taste: a 250 aa window keeps 97% of antibody chains intact while a 400 aa one is 47% padding, and balanced classes beat 3:1 on every metric. Reproduce the sweep by looping over `--vector-size` and `--negative-ratio`.

**Compute budget.** Every entry point caps itself at **50% of the machine** — half the GPU's VRAM, half the CPU threads. Override per run with `--compute-fraction 0.25`, or globally with `IA_COMPUTE_FRACTION=0.25`.

### 3. What training writes

```
run/models/vae/vae-one-hot-250-encoder.keras
run/models/vae/vae-one-hot-250-decoder.keras
run/models/vae/vae-one-hot-250-latent-stats.json     # where sampling should draw from
run/models/siamese/one-hot-250-model.h5
run/models/siamese/one-hot-250-calibration.json      # the score range the model can reach
plots/…                                               # training curves + raw history as JSON
```

`run/` is gitignored — these are local artefacts.

The two JSON side-files are what keep generation honest:

- **`-calibration.json`** — held-out AUC, the score range the classifier actually produces, and precision at a series of operating points. The dashboard reads it to bound its threshold slider. Without it you can ask for a threshold the model is mathematically unable to reach, and a run then burns its whole budget and returns nothing.
- **`-latent-stats.json`** — the VAE's aggregate posterior, plus a centre per chain family. Sampling assumes the latent space resembles N(0, 1); nothing guarantees it does.

---

## How it works

<div align="center">

| | VAE (generator) | Siamese (judge) |
|---|---|---|
| **Input** | 250 × 22 one-hot | two 250 × 22 one-hot |
| **Body** | 3× Conv2D ↓ / Conv2DTranspose ↑ | shared tower: embedding → 4× (Conv1D + BatchNorm + pool) → BiGRU |
| **Latent / merge** | 16 dimensions | concat[ a⊙b , \|a−b\| , a+b ] |
| **Output** | softmax per position | sigmoid interaction score |

</div>

The decoder ends on a **softmax across the alphabet axis**, because a position is one residue among 22 — not 22 independent coin flips. The judge's head sees the product, the absolute difference *and* the sum of the two embeddings: a product alone can only express agreement, and is symmetric, so it cannot tell a pair from its mirror image.

### Measured quality (250 aa, balanced classes)

| | |
|---|---|
| Classifier ROC AUC | **0.76** on 1,586 held-out pairs, **no antibody shared with training** |
| MCC / F1 at the 0.5 cut | **+0.38** / **0.65** |
| Precision at the suggested threshold | **78%**, against a 45% base rate |
| VAE reconstruction | **81%** of residues exact, on held-out chains |
| VAE latent space | per-dimension std ≈ 1.0, matching the prior sampling draws from |

Read a score as a **ranking, not a probability of binding**. The limiting factor is data: 3,989 interacting pairs over 1,406 distinct antibodies.

### Structure prediction

Candidates can be folded with ESMFold through whichever provider is configured, tried in order:

| Provider | Setup | Limit |
|---|---|---|
| [NVIDIA NIM](https://build.nvidia.com/nvidia/esmfold) | free key in `NVIDIA_API_KEY` | 1024 aa |
| [ESM Atlas](https://esmatlas.com/) | none — the keyless default | 400 aa, and a hard ~29 s gateway timeout |
| Local ESMFold | `uv add transformers accelerate` + `IA_LOCAL_ESMFOLD=1` | 1024 aa, works offline |

```bash
NVIDIA_API_KEY=nvapi-... uv run streamlit run app/main.py
```

A full-length antibody chain often cannot beat ESM Atlas's 29 s limit (measured: 55 aa folds in 0.4 s, 215 aa times out), so the dashboard offers to fold just the **variable domain** — where the CDR loops are — when the whole chain fails.

---

## Project structure

```
intelligent_antibodies/modules/
├── models/        VAE, VAEFull, SiameseInteractionClassifier (+ stateful MCC/F1 metrics)
├── layers/        SamplingLayer (the VAE reparameterisation trick)
├── utils/         encoding, inference, chains, paths, gpu, plotting
├── dataset.py     SAbDab loading, de-duplication, class balancing
└── main_pipeline.py   load models → generate → score

scripts/           data pipeline, training, generation, latent-stats refresh
app/               Streamlit dashboard (main.py), real/mock pipelines, ESMFold client
data/              SAbDab + CoV-AbDab reference files
run/               trained weights, calibration, generated FASTA  (gitignored)
plots/             training curves and their raw history
```

Each model module runs its own self-check:

```bash
uv run python -m intelligent_antibodies.modules.models.VAEFull
uv run python -m intelligent_antibodies.modules.models.SiameseInteractionClassifier
```

---

## Data

- **[SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab)** — antibody-antigen complexes solved by X-ray crystallography. What the pipeline trains on.
- **[CoV-AbDab](https://opig.stats.ox.ac.uk/webapps/covabdab/)** — SARS-CoV pairs (`data/CoV-AbDab/`). Included for future work, not yet wired into training.

Two reference files drive the SAbDab pipeline:

- **`All_PDB_files.txt`** — one id per structure; the first four characters are the [RCSB PDB](https://www.rcsb.org/) id.
- **`positive_samples.txt`** — pairs known to form immune complexes, e.g. `4gms_J_N_E\t2vir_B_A_C`. The relation is **reciprocal and unordered**, so `filter_interaction_table.py` checks both orderings. Any pair not listed is treated as a negative.

FASTA records look like this (PDB 1A2Y — the lysozyme complex used as the bundled example):

```
>1A2Y_1|Chain A|IGG1-KAPPA D1.3 FV (LIGHT CHAIN)|Mus musculus (10090)
DIVLTQSPASLSASVGETVTITCRASGNIHNYLAWYQQKQGKSPQLLVYYTTTLADGVPSRFSGSGSGTQYSLKINSLQPEDFGSYYCQHFWSTPRTFGGGTKLEIK
>1A2Y_3|Chain C|LYSOZYME|Gallus gallus (9031)
KVFGRCELAAAMKRHGLANYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL
```

More structures can be pulled from [SAbDab's search tool](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/search/).

---

## Limitations

Worth reading before trusting any output.

- **The judge is weak.** AUC 0.76 makes it a filter, not an oracle. A score of 0.99 means the model ranks that candidate highly, not that it binds.
- **Sequence only, no structure.** Binding is three-dimensional — shape complementarity, buried surface, electrostatics. None of that is modelled.
- **One chain at a time.** A real binding site is formed by a heavy *and* a light chain contributing three CDR loops each; the classifier scores a single chain against an antigen.
- **Nothing about developability.** Real candidates must be stable, soluble, expressible and non-immunogenic. A sequence scoring well here could be impossible to manufacture.
- **Small data.** 3,989 interacting pairs over 1,406 antibodies. No architecture change manufactures more.
- **The field is well past this.** Antibody-specific language models (AbLang, AntiBERTy, IgLM), general protein LMs (ESM), and structure-based design (RFdiffusion + ProteinMPNN, AlphaFold-Multimer) are what the real work uses.

No pretrained weights are shipped — train your own. No licence has been chosen yet; open an issue if that matters to you.

---

## References

- [Deep learning benchmark for antibody-antigen binding](https://www.sciencedirect.com/science/article/pii/S1093326322002431) — the motivating article.
- [emersON106/AbAgIntPre](https://github.com/emersON106/AbAgIntPre) ([paper](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2022.1053617/full)) — a closely related Siamese approach.
- [piercelab/antibody_benchmark](https://github.com/piercelab/antibody_benchmark) — standard antibody-antigen docking benchmark.
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (Kingma & Welling, 2013) — the original VAE.
- [ESM Atlas](https://esmatlas.com/) — the public ESMFold API.

Issues and pull requests welcome.
