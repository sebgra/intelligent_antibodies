# Graph Report - intelligent_antibodies  (2026-10-06)

## Corpus Check
- 98 files · ~74,680 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 902 file(s) not represented in the graph (top: .fasta 866, .ipynb 13, .csv 10)

## Summary
- 565 nodes · 937 edges · 36 communities (22 shown, 14 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 57 edges (avg confidence: 0.88)
- Token cost: 395,597 input · 1,300 output

## Community Hubs (Navigation)
- Streamlit Dashboard App
- VAE Generator & Datasets
- Custom Keras Layers
- Vue Frontend Views
- SAbDab Data Pipeline
- Architecture & Design Rationale
- Vue View Dependencies
- Package Sequence Encoders
- Notebook Sequence Encoders
- Notebook Datasets
- ESMFold Structure Prediction
- Web Interface Dependencies
- Siamese Interaction Classifier
- FASTA Input Component
- Flask API Prototype
- Package Data Generator
- Notebook Data Generator
- Siamese Overfitting Findings
- Siamese F1/MCC Metrics
- App Logo (IgG)
- Siamese Training Curves
- VAE Training Curves
- JS Config
- Web Logo (Y-shape)
- Loading Spinner
- Package Namespace

## God Nodes (most connected - your core abstractions)
1. `load_models()` - 17 edges
2. `ProteinOneHotEncoder` - 17 edges
3. `generate_candidates_real()` - 13 edges
4. `SamplingLayer` - 13 edges
5. `VAE` - 12 edges
6. `generate_candidates()` - 11 edges
7. `VAEFull` - 9 edges
8. `generate_antibody_sequence()` - 9 edges
9. `VAE` - 9 edges
10. `fold_sequence()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `bio-pv in-browser protein structure viewer` --semantically_similar_to--> `ESMFold structure prediction`  [INFERRED] [semantically similar]
  web_interface/README.md → README.md
- `Siamese CNN+GRU interaction classifier` --implements--> `SiameseInteractionClassifier`  [EXTRACTED]
  README.md → intelligent_antibodies/modules/models/SiameseInteractionClassifier.py
- `Convolutional VAE generator (2-D latent space)` --implements--> `VAEFull`  [EXTRACTED]
  README.md → intelligent_antibodies/modules/models/VAEFull.py
- `unfetched.txt - failed fetch log` --semantically_similar_to--> `SAbDab unfetched.txt - concatenated unretrieved PDB ids`  [INFERRED] [semantically similar]
  unfetched.txt → data/SAbDab/unfetched.txt
- `_cached_models()` --calls--> `load_models()`  [EXTRACTED]
  app/real_pipeline.py → intelligent_antibodies/modules/main_pipeline.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **VAE sampling -> Siamese scoring -> folding pipeline** — readme_convolutional_vae_generator, readme_siamese_interaction_classifier, readme_rejection_sampling, readme_one_hot_encoding, readme_esmfold_structure_prediction [EXTRACTED 1.00]
- **SAbDab reference files to filtered training table** — data_sabdab_all_pdb_files_structure_id_list, data_sabdab_positive_samples_interacting_pairs, scripts_download_pdbs, scripts_get_seq_table, scripts_get_interaction_table, scripts_filter_interaction_table, fetched_fetched_pdb_ids, unfetched_failed_fetch_log [EXTRACTED 1.00]
- **CoV-AbDab positive / negative / independent-test split (not yet wired in)** — data_cov_abdab_positive_dataset_positive_pairs, data_cov_abdab_negative_dataset_negative_pairs, data_cov_abdab_independent_test_holdout_set, readme_cov_abdab_dataset [EXTRACTED 1.00]

## Communities (36 total, 14 thin omitted)

### Community 0 - "Streamlit Dashboard App"
Cohesion: 0.05
Nodes (22): load_sabdab_stats(), _plotly_base_layout(), render_3dmol_viewer(), _selected_candidate_ids(), amino_acid_composition(), _amino_acid_composition(), colorize_sequence_html(), generate_candidates() (+14 more)

### Community 1 - "VAE Generator & Datasets"
Cohesion: 0.06
Nodes (30): models_available(), trained_vector_sizes(), AntibodyNLFProtDataset, AntibodyOneHotProtDataset, DatasetFactory, EquilibratedDataset, OneHotProtDataset, _limit_gpu_memory() (+22 more)

### Community 2 - "Custom Keras Layers"
Cohesion: 0.06
Nodes (12): SamplingLayer, VariationalLossLayer, VAE, decoder(), encoder(), SamplingLayer, VariationalLossLayer, VAE (+4 more)

### Community 3 - "Vue Frontend Views"
Cohesion: 0.05
Nodes (12): vue, vue-json-to-csv, vue-router, instance, app, router, getId(), mounted() (+4 more)

### Community 4 - "SAbDab Data Pipeline"
Cohesion: 0.07
Nodes (19): All_PDB_files.txt - antigen-antibody structure id list, positive_samples.txt - known interacting structure pairs, SAbDab unfetched.txt - concatenated unretrieved PDB ids, fetched.txt - successfully fetched PDB ids, alphabet_kmer(), get_pdb_id(), is_antibody_molecule(), unique_pdb_ids() (+11 more)

### Community 5 - "Architecture & Design Rationale"
Cohesion: 0.07
Nodes (27): Serena project configuration (vue language server), app/main.py Streamlit entrypoint, CoV-AbDab independent test set (held out), CoV-AbDab negative dataset (non-interacting pairs), CoV-AbDab positive dataset (interacting heavy/light pairs), modules/dataset.py dataset loading and balancing, modules/utils/inference.py inference loop, AbAgIntPre - Siamese antibody-antigen interaction predictor (+19 more)

### Community 6 - "Vue View Dependencies"
Cohesion: 0.07
Nodes (27): ngl, vite, @vitejs/plugin-vue, dependencies, axios, bio-pv, ngl, vue (+19 more)

### Community 7 - "Package Sequence Encoders"
Cohesion: 0.10
Nodes (4): alphabet_kmer(), BLOSUMEncoder, NLFEncoder, ProteinOneHotEncoder

### Community 8 - "Notebook Sequence Encoders"
Cohesion: 0.10
Nodes (3): BLOSUMEncoder, NLFEncoder, ProteinOneHotEncoder

### Community 9 - "Notebook Datasets"
Cohesion: 0.09
Nodes (5): AntibodyNLFProtDataset, AntibodyOneHotProtDataset, DatasetFactory, EquilibratedDataset, OneHotProtDataset

### Community 10 - "ESMFold Structure Prediction"
Cohesion: 0.17
Nodes (4): fold_sequence(), FoldingError, FoldResult, _parse_ca_bfactors()

### Community 11 - "Web Interface Dependencies"
Cohesion: 0.15
Nodes (12): dependencies, axios, bio-pv, vuepython, vuex, devDependencies, sass, axios (+4 more)

### Community 12 - "Siamese Interaction Classifier"
Cohesion: 0.36
Nodes (6): accuracy(), binary_crossentropy(), f1(), forward(), mcc(), SiameseInteractionClassifier

### Community 13 - "FASTA Input Component"
Cohesion: 0.29
Nodes (7): emitError(), emitText(), getFileContent(), input_text(), loadedContent(), parseFasta(), rules()

### Community 14 - "Flask API Prototype"
Cohesion: 0.31
Nodes (4): generate_antibodies(), get_id(), get_pdb(), hello()

### Community 18 - "Siamese Overfitting Findings"
Cohesion: 0.40
Nodes (4): Siamese Training Curve, Train/Validation Accuracy, Train/Validation Loss, Siamese Network Model

### Community 19 - "Siamese F1/MCC Metrics"
Cohesion: 0.40
Nodes (4): Siamese Training Metrics Plot, F1-score, Matthews Correlation Coefficient (MCC), Siamese Network

### Community 20 - "App Logo (IgG)"
Cohesion: 0.50
Nodes (4): Antibody (IgG Y-shaped structure), IgG Antibody (Y-shaped immunoglobulin), Intelligent Antibodies App Branding, Intelligent Antibodies Logo (Y-shaped antibody icon)

### Community 21 - "Siamese Training Curves"
Cohesion: 0.50
Nodes (4): Siamese Network Training Curves, Siamese Accuracy Curve (train vs validation), Siamese Loss Curve (train vs validation), Siamese Network Model

### Community 22 - "VAE Training Curves"
Cohesion: 0.83
Nodes (4): VAE Training History Chart, VAE KL Divergence Loss (mislabeled), VAE Reconstruction Loss, VAE Total Loss

### Community 23 - "JS Config"
Cohesion: 0.50
Nodes (3): compilerOptions, paths, exclude

### Community 24 - "Web Logo (Y-shape)"
Cohesion: 0.50
Nodes (4): Antibody Y-shaped Icon (teal), Antibody Y-shape Glyph, intelligent-antibodies-view Web App, Intelligent Antibodies Logo

## Ambiguous Edges - Review These
- `All_PDB_files.txt - antigen-antibody structure id list` → `SAbDab unfetched.txt - concatenated unretrieved PDB ids`  [AMBIGUOUS]
  data/SAbDab/unfetched.txt · relation: shares_data_with
- `VAE Training History Chart` → `VAE KL Divergence Loss (mislabeled)`  [AMBIGUOUS]
  plots/VAE_training_curve.png · relation: references

## Knowledge Gaps
- **63 isolated node(s):** `sass`, `sass`, `axios`, `axios`, `bio-pv` (+58 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 258 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `All_PDB_files.txt - antigen-antibody structure id list` and `SAbDab unfetched.txt - concatenated unretrieved PDB ids`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **Why does `VAEFull` connect `VAE Generator & Datasets` to `Custom Keras Layers`, `Architecture & Design Rationale`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `load_models()` (e.g. with `accuracy()` and `binary_crossentropy()`) actually correct?**
  _`load_models()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `sass`, `sass`, `axios` to the rest of the system?**
  _63 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Streamlit Dashboard App` be split into smaller, more focused modules?**
  _Cohesion score 0.05442428730099963 - nodes in this community are weakly interconnected._
- **What is the exact relationship between `VAE Training History Chart` and `VAE KL Divergence Loss (mislabeled)`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `ProteinOneHotEncoder` connect `Notebook Sequence Encoders` to `Streamlit Dashboard App`, `VAE Generator & Datasets`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._