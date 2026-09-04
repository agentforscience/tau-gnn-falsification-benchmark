# Resources Catalog

**Project:** Graph Neural Networks for Predicting Tau Propagation Patterns in Alzheimer's Disease
from Structural Connectome Data
**Phase:** `resource_finder` — complete
**Date:** 2026-09-04

## Summary

| Resource | Count | Location |
|---|---:|---|
| Papers curated | 36 | `papers/` |
| — with full text locally | 24 | 12 PDFs + 12 markdown full texts |
| — abstract-only (paywalled) | 12 | abstracts in `search_results/` |
| Datasets staged | 4 groups, 39 CSVs (~85 MB) | `datasets/` |
| Repositories cloned | 7 (~385 MB) | `code/` |
| Preparation scripts written | 9 | `code/prep/` |

**Bottom line:** the project has everything it needs to run. Both clauses of the hypothesis are
testable with data obtained without any access application — including a **truly longitudinal**
propagation benchmark with 14 experiments, and *MAPT*/*APOE*/*TREM2* expression in both human and
mouse region spaces. The one genuine limitation is that the *human* trajectory is stage-stratified
rather than within-subject longitudinal; the upgrade path is documented.

---

## 1. Papers

36 curated; 24 with complete full text. See `papers/README.md` for the annotated list.

| Category | n | Most important |
|---|---:|---|
| Network diffusion baselines | 13 | **Schäfer, Mormino & Kuhl 2020** — the only multi-timepoint human tau PET fit; supplies both the model equations and the null-model results that constrain our evaluation |
| Gene expression / selective vulnerability | 9 | **Cornblath 2021**, **Henderson 2019**, **Zheng 2019** — the H2 evidence base and the evaluation protocol we copy |
| Tau biology & staging | 5 | Franzmeier 2020 (tau *rate*), Vogel 2024 (tau follows principal axes) |
| GNN / ML methodology | 4 | BrainGNN 2021; Dai 2020 and TauFlowNet 2024 (closest prior art to H1) |
| Tooling | 5 | abagen, neuromaps, ENIGMA Toolbox |

**Acquisition method.** The paper-finder service was not running (`localhost:8000` refused), so
manual multi-source search was used: Europe PMC (primary — this is a biomedical topic), Semantic
Scholar, arXiv, OpenAlex, Crossref, Unpaywall, bioRxiv. Europe PMC's JATS `fullTextXML` endpoint
proved the most reliable route where publisher PDFs were bot-blocked, and was rendered to readable
markdown.

**A correctness note worth flagging.** Three early downloads were the *wrong paper* because a DOI
was guessed rather than resolved (one was a coral-microbiome paper, one a water-hammer slurry
pipeline paper, one an ADNI florbetapir methods memo). `code/prep/verify_titles.py` was written to
check first-page text against the expected title; all 12 retained PDFs now pass at ≥0.75 token
overlap, and all 12 markdown full texts were title-checked by hand.

---

## 2. Datasets

Full documentation, loading examples and caveats: `datasets/README.md`.

| Dataset | Shape | Task role | Location |
|---|---|---|---|
| **Mouse longitudinal tauopathy** | 14 experiments × 3–4 timepoints × 426 regions | **Primary H1 benchmark** — truly longitudinal, known seed | `datasets/mouse_tau_longitudinal/` |
| Mouse connectomes | 4 × (426×426): retrograde, anterograde, non-directional, spatial | Graph + null network | same |
| Mouse gene expression | 426 × 3,855 (incl. `Mapt`, `Apoe`, `Trem2`) | H2 node features | same |
| **ADNI group regional tau/amyloid/atrophy** | 86 × 4 stages (NC/EMCI/LMCI/AD) | **Primary human target** | `datasets/human_tau_adni/` |
| Independent replication cohort (Lyoo/Cho) | 4 groups × 86 regions | Cross-cohort generalisation | same |
| Individual ADNI subjects | 27 × 86 regions + APOE, DX, CDR, MMSE | Subject-level checks | same |
| Human structural connectomes | 3 × (86×86), canonical order | Graph + distance null | same |
| **AHBA gene expression** | 86 × 15,633 (incl. `MAPT`, `APOE`, `TREM2`) | H2 node features | `datasets/ahba_gene_expression/` |
| ENIGMA HCP connectomes | DK-68/82, Schaefer, Glasser | Independent connectome robustness check | `datasets/hcp_connectome/` |

All licensed for research use; no access application was required for any of it.

### The one thing that will silently break the experiment

The upstream connectome files are **not** in the same region order as the tau vectors. The
original MATLAB applies permutations (`permHCP`, `permEve`) before use, and skipping them does not
error — it scrambles the anatomy (left entorhinal ↔ left hippocampus reads 0.00 raw vs 6.25
permuted). `code/prep/build_human_dataset.py` reproduces the permutation and **asserts** three
anatomical invariants. Un-permuted originals are quarantined under
`raw_unpermuted_DO_NOT_USE/`. **Use only the `*_canonical.csv` files.**

### Validation performed (all passing)

- Connectomes: symmetric; homotopic edges enriched 1.70× (ACS) and 2.64× (HCP) over mean edge;
  entorhinal–hippocampus edge non-zero.
- ADNI tau: AD-minus-NC peaks in inferior temporal / inferior parietal / banks-STS / middle
  temporal / fusiform — the expected Braak IV–V pattern.
- Gene alignment: verified region-by-region (`LT_ENTO → L_entorhinal`, `RT_ENTO → R_entorhinal`);
  MAPT highest in entorhinal/temporal cortex, lowest in thalamus/hippocampus/accumbens.
- Mouse data: DS1 pathology grows monotonically 0.0026 → 0.0168 → 0.1804 across 4/8/12 weeks; at
  t=8 the top regions are lateral and medial entorhinal cortex, consistent with dentate-gyrus seeding.
- Missingness in mouse data (54% NaN) is **structural, not random** — each experiment measures a
  fixed region subset at all its timepoints (194 regions complete across all 8 Kaufman strains).

---

## 3. Code repositories

See `code/README.md` for detail.

| Name | URL | Purpose |
|---|---|---|
| `Aggregation-Network-Diffusion` | github.com/Raj-Lab-UCSF/Aggregation-Network-Diffusion | **Human tau + connectome data**; AND/eNDM MATLAB baselines |
| `Nexis` | github.com/Raj-Lab-UCSF/Nexis | **Mouse longitudinal tau + connectome + genes**; NexIS gene-modulated NDM |
| `tau-spread` | github.com/ejcorn/tau-spread | Cornblath 2021 individual-mouse tau; **out-of-sample evaluation protocol** |
| `connectome_diffusion` | github.com/ejcorn/connectome_diffusion | Henderson 2019 α-synuclein; null-model templates |
| `ENIGMA` | github.com/MICA-MNI/ENIGMA | HCP connectomes; AHBA expression source |
| `tau-PET-prediction` | github.com/Raj-Lab-UCSF/tau-PET-prediction | Python NDM/linear/**constant** baselines on ADNI Δtau |
| `NTM-Human` | github.com/Raj-Lab-UCSF/NTM-Human | Axonal transport (advection+diffusion) model |

---

## 4. Environment

Isolated venv at `.venv/` (Python 3.12.8), `uv`-managed, `pyproject.toml` scoped to this workspace
(`package = false`). Installed and smoke-tested: `torch 2.14.0+cpu`, `torch-geometric 2.8.0`,
`pandas 3.0.5`, `numpy 2.5.2`, `scipy 1.18.1`, `scikit-learn 1.9.0`, `networkx`, `matplotlib`,
`pypdf`, `httpx`, `openpyxl`. A `GCNConv` forward pass runs.

---

## 5. Search strategy and selection criteria

**Strategy.** Keyword families across five sources: (a) network diffusion / epidemic spreading +
tau; (b) GNN / geometric deep learning + connectome + tau propagation; (c) AHBA / imaging
transcriptomics + regional vulnerability; (d) longitudinal tau PET prediction; (e) tooling.
Europe PMC did the heavy lifting; Semantic Scholar's `tau_spreading_model` query surfaced the
richest single result set.

Dataset discovery followed a different and more productive route: rather than searching data
portals (which yielded nothing usable — OpenNeuro has 36 PET datasets and only one tau study, of
33 *young healthy* adults), I read the **data-availability statements of the papers themselves**.
That is what surfaced the Raj Lab and Cornblath repositories, which is where all the usable data
turned out to live.

**Selection criteria.** Papers: direct bearing on H1/H2, methodological reusability, released
data/code, and coverage of the classical baselines we must beat. Datasets: openly obtainable
without an access application, region-space compatible with both a connectome and gene expression,
and — critically — longitudinal where possible.

---

## 6. Challenges, and how they were handled

| Challenge | Resolution |
|---|---|
| paper-finder service down | Manual multi-source search (Europe PMC / S2 / arXiv / OpenAlex / Crossref / Unpaywall) |
| Publisher PDFs bot-blocked (Cell 403, PMC 403, eLife 406) | Europe PMC JATS `fullTextXML` → markdown; recovered 12 papers this way |
| bioRxiv rate-limiting (429) | Retried with delay; partially successful |
| Three wrong papers downloaded from guessed DOIs | Wrote `verify_titles.py`; re-resolved via Crossref; all retained files verified |
| **No open longitudinal human tau PET** | Verified exhaustively (OpenNeuro API, ADNI/OASIS gated). Pivoted to (a) mouse longitudinal data and (b) stage-stratified human data — both literature-standard designs. Documented the ADNI upgrade path. |
| Connectome/tau region-order mismatch | Traced the permutation in the upstream MATLAB, reproduced it in Python **with assertions**, quarantined the raw files |
| Baseline implementations are MATLAB/R; neither runtime available | Confirmed absent; read and documented the algorithms instead. NDM equations are fully specified in the PDFs we hold, so Python reimplementation is unblocked |
| `uv add torch` failed to resolve | Added an explicit `[[tool.uv.index]]` CPU index to `pyproject.toml` |

---

## 7. Gaps and honest limitations

1. **Human data is stage-stratified, not within-subject longitudinal.** This is the most important
   caveat. It is a field-wide constraint (the 2025 *Brain* paper skipped longitudinal analysis for
   the same reason), and the design we use matches Raj et al. 2015 — but H1 as literally worded
   ("longitudinal tau PET") is fully testable only on the mouse arm, or on ADNI after an access
   application.
2. **12 papers are abstract-only.** No baseline we must implement is blocked by this.
3. **AHBA has 6 donors, 2 with right-hemisphere tissue**; 6 of 86 regions lack expression.
4. **Mouse ≠ human.** The mouse arm is seeded tauopathy in transgenic models, not sporadic AD.
   Cross-species transfer was explicitly pruned as a direction (`planning.md`, R6).
5. **Statistical power is limited by design.** 86 regions × 4 stages (human) and 194–346 regions ×
   3–4 timepoints × 14 experiments (mouse). This favours constrained models over large ones and
   makes overfitting the primary risk to H1.

---

## 8. Recommendations for experiment design

1. **Primary dataset:** `datasets/mouse_tau_longitudinal/` — the only true longitudinal test of H1.
   Hold out whole experiments, not just regions.
   **Secondary:** `datasets/human_tau_adni/` with the independent replication cohort for
   cross-cohort generalisation.
2. **Baselines (in priority order):** persistence/constant → NDM heat kernel → Fisher–Kolmogorov →
   eNDM/NexIS (gene-modulated — the true H2 comparator) → bidirectional antero/retrograde → ESM.
3. **Null networks:** spatial-distance, degree-preserving rewired, anterograde-only,
   retrograde-only, shuffled seed.
4. **Metrics:** spatial Pearson R on **Δtau** (log10 for mouse), R² on held-out
   regions/experiments/cohorts, out-of-sample distributions over ≥100 splits, spin tests for any
   gene–pathology correlation, Fisher R-to-z for model comparison.
5. **Model choice:** prefer a **diffusion-constrained GNN** (graph neural ODE, or message passing
   initialised at the graph Laplacian) over a generic GCN — the data is small and overfitting is
   the main threat to a valid comparison.
6. **H2 ablation:** {none} → {MAPT, APOE, TREM2} → {AD-risk panel} → {size-matched random gene
   sets} → {all genes, PCA}. The random-gene-set control is not optional.
7. **Always report the persistence baseline.** Per Schäfer et al. 2020, a no-spreading model scores
   R = 0.974 on absolute tau. Without this column, results are not interpretable — and a null
   result for H1 is a legitimate, publishable outcome.

---

## 9. Reproducing everything

```bash
cd /workspaces/graph_neural_networks_for_pred_20260904_071227_2a08199c
source .venv/bin/activate

python code/prep/build_human_dataset.py      # human DK-86 (asserts anatomical validity)
python code/prep/align_gene_expression.py    # AHBA -> DK-86
python code/prep/export_mouse_data.py        # mouse 426-region longitudinal

curl -L -o datasets/ahba_gene_expression/allgenes_stable_rAll_dk82.csv \
  https://raw.githubusercontent.com/saratheriver/enigma-extra/master/ahba/allgenes_stable_r-1.csv
```
Repository clone commands are listed in `code/README.md`.
