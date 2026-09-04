# Research Planning — Direction Budget

**Topic:** Graph Neural Networks for Predicting Tau Propagation Patterns in Alzheimer's Disease
from Structural Connectome Data

**Hypothesis (two clauses):**
- **H1** — GNNs trained on structural connectome data outperform standard network diffusion
  models at predicting longitudinal tau PET spreading patterns.
- **H2** — Adding regional gene expression features (particularly *MAPT*, *APOE*, *TREM2*) as
  node-level features improves prediction of region-specific vulnerability to tau accumulation.

Phase: `resource_finder`. Directions enumerated below were scored on (a) literature evidence,
(b) relevance to H1/H2, (c) expected information gain, (d) implementation feasibility with
resources actually obtainable in this workspace. **Top 3 retained**; the rest are pruned with
reasons, and must not be re-expanded unless new evidence invalidates the ranking.

---

## Scoring summary

| # | Direction | Evidence | Relevance | Info gain | Feasibility | **Total** | Verdict |
|---|-----------|---------:|----------:|----------:|------------:|----------:|---------|
| D1 | GNN vs. network-diffusion on **mouse longitudinal tauopathy** (Nexis / Kaufman–Diamond) with gene node features | 5 | 4 | 5 | 5 | **19** | **KEEP** |
| D2 | GNN vs. NDM/ESM on **human DK-86 stage-stratified tau** (ADNI + independent replication cohort) with AHBA gene features | 5 | 5 | 4 | 4 | **18** | **KEEP** |
| D3 | **Gene-feature ablation + regional-vulnerability attribution** with rigorous nulls (spin/rewire/random-gene) | 5 | 5 | 5 | 5 | **20** | **KEEP** |
| R1 | Individual-subject longitudinal human tau PET (full ADNI tau tables) | 5 | 5 | 5 | 1 | 16 | prune (access) |
| R2 | Voxel-level GNN on raw tau PET images | 2 | 3 | 3 | 1 | 9 | prune |
| R3 | Joint amyloid–tau interaction modelling | 4 | 2 | 3 | 4 | 13 | prune (scope) |
| R4 | Functional-connectome propagation | 4 | 2 | 3 | 5 | 14 | prune (hypothesis says *structural*) |
| R5 | Build a new connectome from raw dMRI | 3 | 1 | 1 | 1 | 6 | prune |
| R6 | Cross-species transfer (mouse → human) | 2 | 3 | 4 | 2 | 11 | prune (risk) |
| R7 | Continuum Fisher–Kolmogorov PDE on brain meshes | 4 | 2 | 2 | 2 | 10 | prune |

*(1 = poor, 5 = excellent.)*

---

## Retained directions

### D1 — GNN vs. network diffusion on mouse longitudinal tauopathy
**Why it wins:** this is the only *truly longitudinal, per-experiment* propagation dataset that is
openly downloadable. `datasets/mouse_tau_longitudinal/` provides **14 independent tau-seeding
experiments**, each with 3–4 timepoints over 426 regions, a **known ground-truth seed region**,
**directed** anterograde/retrograde connectomes, a spatial-distance null network, and a
426 × 3,855 regional gene expression matrix that contains *Mapt*, *Apoe* and *Trem2*.

It supports the strongest possible test of H1: a forward-prediction task with an unambiguous
initial condition, held-out timepoints, and held-out *experiments*. Schäfer et al. (2020) showed
the field has never validated these models on multi-timepoint data — this closes that gap.

**Concrete next steps:** predict pathology at t+1 from t (and from seed alone) with GCN/GAT/GraphSAGE
and a graph neural-ODE; baselines = NDM (heat-kernel), Fisher–Kolmogorov, eNDM/Nexis,
anterograde-only, retrograde-only, Euclidean-distance network, degree-preserving rewired
connectome, and the persistence ("no diffusion") null.

### D2 — GNN vs. NDM/ESM on human DK-86 stage-stratified tau
**Why it wins:** H1 and H2 are stated about *human* tau PET, so a human arm is required for
relevance even though the openly available human trajectory is stage-stratified
(NC → EMCI → LMCI → AD) rather than within-subject longitudinal. This is exactly the design used
by Raj et al. (2015) and the Franchi–Raj AND model, so it is a *literature-standard* evaluation,
not an improvisation. Crucially, `datasets/human_tau_adni/` supplies an **independent replication
cohort** (Lyoo/Cho: AD / aMCI / naMCI / NL) in the same 86-region space, so generalisation can be
tested across cohorts rather than only across regions.

The connectome, tau, and AHBA gene expression are all aligned to one verified canonical region
order (see `datasets/README.md` — this alignment is *not* trivial and is the single most likely
source of silent error).

**Concrete next steps:** fit stage-to-stage transitions; evaluate with region-held-out and
cohort-held-out splits; compare against NDM/eNDM and the distance-weighted null.

### D3 — Gene-feature ablation and regional-vulnerability attribution
**Why it wins:** this *is* H2, and it is cheap to run inside both D1 and D2. It is also where the
literature is most contested — Zheng et al. (2019), Cornblath et al. (2021), Henderson et al.
(2019) and Anand et al. (2022) all report that connectivity alone leaves structured residual
variance that regional molecular features explain, but effect sizes are modest and null models
matter enormously.

**Concrete next steps:** ablate {no genes} vs {MAPT, APOE, TREM2} vs {AD-risk panel} vs {matched
random gene sets} vs {all genes, PCA}; test whether gene features specifically reduce *residual*
error after connectivity is accounted for; attribute per-region vulnerability and check it recovers
the known entorhinal/temporal gradient.

---

## Pruned directions and reasons

- **R1 — individual-subject longitudinal human tau PET.** This is scientifically the ideal
  dataset and is *deliberately* pruned only on access, not on merit. ADNI's
  `UCBERKELEY_TAU_*` tables give per-subject longitudinal DK-atlas SUVR, but require an
  application to LONI/IDA that cannot complete within this session. The exact request procedure
  is documented in `datasets/README.md` as the primary upgrade path. If access is granted,
  D2 should be re-scored and promoted.
- **R2 — voxel-level GNN on raw PET.** No openly available longitudinal AD tau PET exists on
  OpenNeuro (verified: 36 PET datasets, only one tau study — 33 *young healthy* adults,
  cross-sectional). Would require raw imaging plus a full preprocessing pipeline.
- **R3 — joint amyloid–tau modelling.** Data *is* present (regional amyloid for both cohorts) and
  Lee et al. (2022) show real Aβ–tau interaction effects, but this widens the hypothesis beyond
  what was asked. Retained only as an optional covariate, not a direction.
- **R4 — functional connectome.** ENIGMA ships HCP functional matrices and Franzmeier et al.
  (2020) show functional connectivity predicts tau accumulation — but the hypothesis explicitly
  names the *structural* connectome. Usable as a secondary comparison, not a direction.
- **R5 — build a connectome from raw dMRI.** No raw diffusion data; group connectomes from HCP
  are already available from three independent sources.
- **R6 — cross-species mouse → human transfer.** Attractive, but region homology mapping between
  the 426-region Allen space and DK-86 is itself an unsolved research problem; high risk of the
  result being an artefact of the mapping.
- **R7 — continuum Fisher–Kolmogorov PDE.** Requires brain meshes; the graph-discretised form
  (Schäfer et al. 2020, Eq. 2) is already the standard baseline and is what we implement.

---

## Non-negotiable evaluation constraint (drives all three directions)

Schäfer, Mormino & Kuhl (2020) is the only study that fit network diffusion to genuinely
multi-timepoint human tau PET, and its results contain a warning that must shape our design:

| Model | residual err | R |
|---|---:|---:|
| Connectivity-weighted (full) | 2.4618 | 0.9752 |
| Distance-weighted (full) | 2.3861 | 0.9756 |
| **No diffusion (κ = 0)** | **2.7269** | **0.9740** |
| Baseline PET vs. final PET (raw data autocorrelation) | — | **0.9496** |

A model that predicts *"tau does not change"* scores R = 0.974. Absolute-tau spatial correlation
is therefore very close to uninformative, and the distance-weighted null slightly **beat** the
connectome model. Consequently every experiment in D1–D3 must:

1. evaluate on **Δtau** (change from baseline), not absolute tau;
2. always report the **persistence baseline** (predict no change) alongside;
3. include the **Euclidean-distance network** and a **degree-preserving rewired connectome** as
   null networks — following Henderson et al. (2019) and Cornblath et al. (2021);
4. use **out-of-sample** splits (Cornblath used 500 train/test splits; Henderson 100), not in-sample fit;
5. report spatial **Pearson R on log10 pathology** for the mouse arm, matching the literature
   convention so numbers are comparable to published baselines.

Any claim that "the GNN beats network diffusion" that is not accompanied by (1)–(4) should be
treated as unsupported.

---
---

# PART II — Experiment Runner Phase: Motivation, Novelty, and Experimental Protocol

*Added by the `experiment_runner` phase, 2026-09-04. Part I above (direction budget) is
retained unchanged and is the ranking this part executes against.*

## Motivation & Novelty Assessment

### Why This Research Matters

Tau neurofibrillary pathology, unlike amyloid, tracks tightly with cognitive decline and regional
atrophy in Alzheimer's disease, and it spreads through the brain in a stereotyped sequence (Braak
staging) that is widely believed to reflect trans-synaptic, connectome-constrained propagation of
misfolded tau seeds. If the spatial trajectory of tau can be *predicted* from an individual's
baseline scan plus a connectome, that enables (i) prognostic staging, (ii) enrichment of clinical
trials with patients whose pathology is about to reach cognitively eloquent cortex, and (iii)
identification of the molecular determinants of why some regions resist tau while their
directly-connected neighbours succumb. The classical tool for this is the **network diffusion
model (NDM)** — an analytically elegant heat-kernel on the connectome graph. Graph neural networks
are the obvious modern generalisation, but the field has almost no controlled comparison of the
two on genuinely longitudinal data.

### Gap in Existing Work

From `literature_review.md` and the 24 full-text papers in `papers/`:

1. **The longitudinal gap.** Nearly every published network-diffusion-vs-tau paper fits
   *cross-sectional* or *stage-stratified* data. Schäfer, Mormino & Kuhl (2020) is the single
   exception with multi-timepoint human tau PET — and its own null models (below) show the
   evaluation used across the entire field is close to uninformative.
2. **The evaluation gap (the critical one).** Schäfer et al. report that a **"no diffusion"
   model scores R = 0.9740** against the full connectivity model's R = 0.9752, and that the raw
   baseline→final autocorrelation of the data is already R = 0.9496. Almost every headline
   correlation in this literature is dominated by the fact that tau maps are similar to
   themselves. A distance-weighted null network also numerically *beat* the connectome.
3. **The GNN gap.** The two closest prior works (Dai et al. 2020; TauFlowNet 2024) apply deep
   models to tau imaging but do **not** benchmark against a properly-nulled NDM on Δtau with
   out-of-sample splits, so it is currently unknown whether GNN gains are real or are the
   persistence effect in disguise.
4. **The gene gap.** Zheng 2019, Henderson 2019, Cornblath 2021 and Anand 2022 all report that
   regional gene expression explains structured residual variance beyond connectivity — but with
   modest effect sizes, inconsistent gene sets, and (in several cases) **no size-matched random
   gene-set control**, which is exactly the control that a 15,000-gene search space demands.

### Our Novel Contribution

We contribute a **falsification-first benchmark**, not a new architecture claim:

- A **diffusion-constrained graph neural ODE** (`TauODE`) whose zero-parameter limit is *exactly*
  the NDM heat kernel, so the GNN and its classical baseline live in the same model family and
  differ only in the learned terms. This makes "does the GNN beat diffusion?" a nested-model
  question rather than an apples-to-oranges comparison.
- Evaluation on **Δtau with the persistence baseline always reported**, on **held-out whole
  experiments** (mouse) and a **held-out independent cohort** (human) — the protocol the field
  has been missing.
- A **null-network ladder** (spatial distance, degree-preserving rewire, shuffled seed,
  anterograde-only, retrograde-only) run identically for every model.
- For H2, a **size-matched random gene-set null** (100 draws) plus an AD-risk-panel and a
  whole-transcriptome PCA condition, so any *MAPT*/*APOE*/*TREM2* effect is tested against the
  correct null rather than against zero.

We consider a **negative result for H1 to be a fully publishable outcome** and have pre-committed
to reporting it as such.

### Experiment Justification

| Exp | Name | Why it is necessary |
|---|---|---|
| **E0** | Null-model calibration | Reproduces the Schäfer persistence trap on *our* data, establishing the numbers every later table must be read against. Without it nothing else is interpretable. |
| **E1** | Mouse seed→trajectory, LOEO | The only truly longitudinal, known-initial-condition test of H1 available openly. Leave-one-experiment-out is the strictest split (generalisation to a new mouse strain/injection, not a new region). |
| **E2** | Mouse one-step forecast | Tests the *forecasting* form of H1 where persistence is the dominant competitor; separates "can you predict tau" from "can you predict the *change* in tau". |
| **E3** | Null-network ladder | Distinguishes "the connectome matters" from "any smooth spatial operator matters". Schäfer's distance-null result makes this non-optional. |
| **E4** | Human stage-transition + cohort transfer | H1/H2 are stated about human tau PET. Fitting ADNI and testing on the independent Lyoo/Cho cohort is the strongest generalisation test the open data allows. |
| **E5** | H2 gene ablation ladder | Directly tests clause 2 of the hypothesis, with the random-gene-set null that makes the test valid. |
| **E6** | Regional vulnerability attribution | Asks whether gene features improve prediction *where the hypothesis says they should* (entorhinal/temporal), rather than only in aggregate. |
| **E7** | Robustness: seeds, capacity, connectome choice | Guards against the two failure modes that would invalidate E1–E6: seed-dependent variance in tiny-data training, and results driven by one particular connectome. |

## Experimental Protocol (pre-registered before running)

**Primary endpoint (H1).** Out-of-sample spatial Pearson *R* between predicted and observed
log10 pathology on held-out mouse experiments (LOEO, n = 8 Kaufman experiments), averaged over
timepoints. GNN vs. best network-diffusion baseline, compared with a **paired** test across
held-out experiments (Wilcoxon signed-rank; Fisher r-to-z for the correlation scale).
α = 0.05, two-sided.

**Primary endpoint (H2).** Change in the same metric when *MAPT/Apoe/Trem2* node features are
added, evaluated against a null distribution of 100 size-matched random gene triplets.
Reported as an empirical p-value = (1 + #{random ≥ observed}) / (1 + 100).

**Secondary endpoints.** Δtau Pearson R and R²; human ADNI region-held-out CV; human
ADNI→replication-cohort transfer; per-region residual analysis.

**Pre-committed decision rule.** H1 is supported only if the GNN beats *both* (a) the best
diffusion baseline and (b) the best null-network model, on held-out data, with p < 0.05 paired.
H2 is supported only if the AD-gene condition exceeds the 95th percentile of the random-gene
null. Anything else is reported as not supported.

**Seeds.** Every model trained with 5 random seeds (0–4); we report mean ± SD across seeds, and
seed variance is itself a reported result.
