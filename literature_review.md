# Literature Review

**Hypothesis under test**
- **H1** — GNNs trained on structural connectome data outperform standard network diffusion
  models at predicting longitudinal tau PET spreading patterns.
- **H2** — Adding regional gene expression (particularly *MAPT*, *APOE*, *TREM2*) as node-level
  features improves prediction of region-specific vulnerability to tau accumulation.

Based on 36 curated papers, 24 read in full. Sources: Europe PMC, Semantic Scholar, arXiv,
OpenAlex, Crossref (the paper-finder service was not running; manual multi-source search was used).

---

## 1. Research area overview

Tau in Alzheimer's disease spreads in a stereotyped spatial sequence (Braak staging: transentorhinal
→ limbic → temporal → association isocortex → primary sensory). Since Braak & Braak (1991) the
central mechanistic question has been *what determines that sequence*. Two families of answer
compete, and the modern literature is essentially an argument about their relative weight:

1. **Connectivity-mediated (prion-like) spread.** Misfolded tau templates and transfers along
   axonal connections, so the connectome constrains the spatial pattern. Formalised as diffusion
   on a graph.
2. **Selective regional vulnerability.** Regions differ intrinsically — in cell-type composition,
   molecular milieu, gene expression — and this modulates where pathology takes hold regardless
   of connectivity.

The field has largely converged on "both", but the *quantitative* balance is unresolved, and that
is exactly the gap H1/H2 addresses. Deep learning has arrived in this space almost entirely for
**diagnosis/classification**, not for **propagation forecasting** — which is the opening this
project targets.

---

## 2. The baseline family: network diffusion models

### 2.1 The canonical formulation

All variants discretise a reaction–diffusion PDE on the connectome graph. Following
Schäfer, Mormino & Kuhl (2020), Eq. 2:

```
dc_i/dt = −κ Σ_j L_ij c_j  +  α c_i (1 − c_i)
```

- `c_i` — misfolded tau concentration at region *i*, scaled to [0,1]
- `L = D − A` — weighted graph Laplacian; `D_ii = Σ_{j≠i} A_ij`
- `κ` — global diffusion coefficient; `α` — local production/clearance rate

Dropping the reaction term gives the original **Raj et al. (2012, 2015) NDM**, which has the
closed form `c(t) = exp(−βLt) c(0)` — a heat kernel on the connectome. Adjacency choices matter:

| Model | Adjacency |
|---|---|
| Connectivity-weighted (intracellular) | `A_ij = n_ij / l_ij` (fibre count / fibre length) |
| Distance-weighted (extracellular) | `A_ij = 1 / d_ij` (inverse Euclidean distance) |

### 2.2 Model variants in the literature

| Model | Paper | Addition over plain NDM |
|---|---|---|
| **NDM** | Raj 2012, 2015 | — (heat kernel) |
| **ESM** (epidemic spreading) | Iturria-Medina 2014; Vogel 2020 | Agent-based production/clearance/transmission; fits regional onset times |
| **Fisher–Kolmogorov** | Schäfer 2020 | Logistic growth term `α c(1−c)` |
| **eNDM / NexIS** | Anand 2022 | **Gene-modulated** source and sink terms; directional weight `s` |
| **AND** (aggregation + diffusion) | Raj 2021 | Explicit aggregation kinetics |
| **NTM** (network transport) | Raj Lab 2024 | Advection (active axonal transport) + diffusion |
| **Bidirectional** | Cornblath 2021; Henderson 2019 | Separate anterograde and retrograde terms with fitted weights |
| **Learned dynamical system** | Garbarino & Lorenzi 2021 | Learns the propagation operator instead of assuming diffusion |

### 2.3 Evidence for and against connectivity

**For:** Vogel et al. (2020) fit ESM to human tau PET and find connectome-constrained spread.
Franzmeier et al. (2020, *Sci. Adv.*) show epicentre connectivity predicts the tau spreading
sequence in ADNI (β = 0.72, R² = 0.52) and replicates in BioFINDER (β = 0.71, R² = 0.50), with
cross-cohort spatial correlation r = 0.78. Henderson et al. (2019) show a Euclidean-distance
network explains α-synuclein spread far worse (r = −0.07–0.1) than the real connectome, and that
degree-preserving rewiring destroys prediction (r = 0.14–0.18).

**Against / complicating:** Schäfer et al. (2020) — the only study fitting to genuinely
multi-timepoint human tau PET — found the **distance-weighted model marginally outperformed the
connectivity-weighted one** (err 2.386 vs 2.462; R 0.9756 vs 0.9752; paired t-test p = 0.07, i.e.
not significant). So in the one human longitudinal test that exists, the connectome hypothesis is
not clearly supported over simple spatial proximity.

---

## 3. ⚠️ The evaluation trap (this should drive the whole experimental design)

Schäfer et al. (2020) report null-model results that undermine most headline numbers in this field:

| Model | residual err | R |
|---|---:|---:|
| Connectivity-weighted (full) | 2.4618 | 0.9752 |
| Distance-weighted (full) | 2.3861 | 0.9756 |
| Intracellular, no production (α=0) | 4.3736 | 0.9528 |
| **No diffusion (κ=0) — pure local production** | **2.7269** | **0.9740** |
| No diffusion, no production (**constant**) | 4.5624 | 0.9514 |
| **Baseline PET vs. final PET (raw autocorrelation)** | — | **0.9496** |

Read that carefully: a model with **no spreading at all** achieves R = 0.9740 against a full model's
R = 0.9752. And the data's own baseline-to-final autocorrelation is already R = 0.95. Regional tau
is so temporally autocorrelated that *spatial correlation on absolute tau is nearly uninformative*.

**Consequences that are non-negotiable for this project:**

1. Evaluate on **Δtau** (change), not absolute tau.
2. Always report the **persistence baseline** ("tau does not change") next to every model.
3. Include a **Euclidean/spatial-distance network** and a **degree-preserving rewired connectome**
   as null networks (Henderson 2019; Cornblath 2021).
4. Use **out-of-sample** evaluation — Cornblath used 500 train/test splits, Henderson 100 — not
   in-sample fit.
5. For the mouse arm, report spatial Pearson R on **log10** pathology, matching convention so
   numbers are comparable to published baselines.

Any claim that a GNN "beats network diffusion" without (1)–(4) is unsupported. This is the single
most valuable thing the literature review produced.

---

## 4. Selective vulnerability and gene expression (H2)

### 4.1 The core evidence

- **Zheng et al. (2019, *PLoS Biol.*)** — "Local vulnerability and global connectivity jointly
  shape neurodegenerative disease propagation." The clearest statement of H2: neither connectivity
  nor regional molecular profile alone suffices.
- **Henderson et al. (2019, *Nat. Neurosci.*)** — regional *Snca* expression explains variance in
  α-synuclein spread that connectivity leaves unexplained. The direct structural analogue of H2,
  with the most rigorous null models in the literature.
- **Cornblath et al. (2021, *Sci. Adv.*)** — bidirectional connectome model on longitudinal mouse
  tau; residuals from the connectivity model map onto regional gene-expression gradients, and a
  genetic risk factor (*LRRK2* G2019S) alters fitted spread parameters.
- **Anand et al. (2022, *Sci. Rep.*)** — NexIS explicitly parameterises gene modulation and finds
  *Trem2* effects on tauopathy progression are **stage-dependent** (protective vs. detrimental at
  different disease stages). This is a direct, quantified precedent for the *TREM2* clause of H2 —
  and a warning that a single static gene coefficient may be the wrong model.
- **Sepulcre et al. (2018, *Nat. Med.*)** — neurogenetic contributions to human Aβ and tau
  spreading (abstract only locally).
- **Grothe et al. (2018, *Brain*)** — AHBA-derived molecular correlates of the AD vulnerability
  gradient.
- **Shafiei et al. (2023, *Brain*)** — same connectome + transcriptome design in FTD; a good
  methodological template.
- **Selective vulnerability and resilience... genes and the connectome (2025, *Brain*)** — the most
  recent human synthesis (ADNI tau PET + AHBA + eNDM), stratified by APOE ε4 and amyloid status.

### 4.2 Honest caveats on the gene side

- **AHBA is 6 donors, only 2 with right-hemisphere tissue.** The 2025 *Brain* paper concedes this
  directly and notes no comparable spatially-resolved alternative exists. Right-hemisphere
  expression is interpolated or missing (in our data: `RT_FRONT_POLE`, `RT_TEMP_POLE` are NaN).
- **Gene expression is from *healthy* donors**, not patients — it is a constitutional vulnerability
  proxy, not a disease-state measurement.
- **Spatial autocorrelation inflates significance.** Arnatkevičiūtė et al. (2019) is the standard
  reference for this; spatial-null ("spin") tests are mandatory when correlating a gene map with a
  pathology map. Random *gene-set* nulls matter too — many genes covary with the cortical
  hierarchy, so a "significant" MAPT effect may not be specific to MAPT.

---

## 5. Deep learning and GNNs in this space — where the gap is

**What exists.** GNNs are widely applied to brain graphs for **classification**: BrainGNN
(Li et al. 2021, *Med. Image Anal.*) for fMRI, and many AD diagnosis models found in our search
(MLC-GCN, contrastive graph pooling, brain-aware readout layers, graph transformers).

**What barely exists.** GNNs for **propagation forecasting**. Our searches surfaced only two
directly on-hypothesis works, and both are small:

- **Dai et al. (2020, MICCAI)** — "A physics-informed geometric learning model for pathological
  tau spread in Alzheimer's disease." Geometric deep learning with a diffusion prior. Closest
  prior art to H1.
- **TauFlowNet (2024, *Med. Image Anal.*)** — learns the tau transport/flow field with deep neural
  transport equations.

Neither is open-access; both are held as abstract-only. Also relevant: "Network
Diffusion-Constrained Variational Generative Models" (2025) and "Prediction of misfolded protein
spreading using machine learning and spreading models" (2023).

**Assessment:** H1 addresses a genuine and current gap — the intersection of "GNN" and
"longitudinal tau propagation forecasting" is nearly empty, whereas "GNN" and "AD classification"
is saturated. But note the corollary: the reason may partly be **data scarcity**, not lack of
interest. Longitudinal tau PET cohorts are small (Schäfer: n=46; the 2025 *Brain* paper skipped
longitudinal analysis entirely for lack of data), which limits how much a high-capacity model can
gain over a 2-parameter diffusion model. A GNN with thousands of parameters fit to ~86 regions
across 4 disease stages will overfit unless heavily regularised or structurally constrained.
**A physics-informed / diffusion-constrained GNN is the more defensible design than a generic GCN.**

---

## 6. Standard baselines, metrics and null models

**Baselines to implement** (in rough order of necessity):
1. **Persistence / constant** — predict no change. *Mandatory.*
2. **NDM** — heat kernel `exp(−βLt)`, one parameter.
3. **Fisher–Kolmogorov** — adds logistic growth, two parameters (Schäfer 2020 Eq. 2).
4. **ESM** — epidemic spreading (Iturria-Medina 2014; Vogel 2020).
5. **eNDM / NexIS** — gene-modulated NDM (Anand 2022). *The correct comparator for H2* — the GNN
   must beat this, not just plain NDM.
6. **Bidirectional antero/retrograde** — mouse arm (Cornblath 2021).
7. **Linear regression on connectivity features** — cheap, often surprisingly strong.

**Null networks:** Euclidean/spatial distance; degree-preserving rewired connectome;
anterograde-only; retrograde-only; alternative seed regions (specificity test).

**Metrics:**
- Spatial **Pearson R** between observed and predicted regional pathology (field convention;
  log10 for mouse). Report on **Δtau**.
- **R²** / variance explained on held-out regions and held-out subjects/experiments.
- Out-of-sample fit distributions over many train/test splits, compared with paired tests.
- **Spin tests** (spatial autocorrelation-preserving nulls) for any gene–pathology correlation.
- Fisher R-to-z for comparing correlation coefficients between models.

---

## 7. Datasets used in the literature vs. what we obtained

| Dataset | Used by | Our access |
|---|---|---|
| ADNI tau PET (AV1451), per-subject longitudinal | Schäfer 2020; Franzmeier 2020; *Brain* 2025 | **Gated** — requires LONI application (procedure documented in `datasets/README.md`) |
| ADNI group-level regional tau/amyloid/atrophy | Raj 2015, 2021 | ✅ obtained |
| Independent Korean tau cohort (Lyoo/Cho) | Raj 2021 | ✅ obtained |
| BioFINDER | Franzmeier 2020 | ✗ not public |
| Kaufman/Diamond mouse tauopathy, longitudinal | Anand 2022; Raj Lab | ✅ obtained (14 experiments) |
| Cornblath PS19 mouse tau | Cornblath 2021 | ✅ obtained |
| Allen Mouse Brain Connectivity Atlas | Anand 2022; Henderson 2019 | ✅ obtained |
| AHBA human gene expression | Grothe 2018; Sepulcre 2018; *Brain* 2025 | ✅ obtained (15,633 genes × DK-86) |
| HCP structural connectome | most human studies | ✅ obtained (3 independent versions) |

---

## 8. Gaps and opportunities

1. **No GNN has been benchmarked against network diffusion on multi-timepoint tau propagation.**
   This is H1, and it is genuinely open.
2. **Null models are inconsistently applied.** Several high-profile results would look weaker
   against the persistence baseline. A study that reports them rigorously has value even if the
   GNN *loses*.
3. **Gene effects are usually tested as post-hoc correlation with residuals**, rarely as node
   features inside the predictive model. H2's framing (gene expression as *node features*) is a
   real methodological contribution.
4. **Stage-dependence is under-modelled.** Anand 2022 found *Trem2* flips sign across disease
   stages; static coefficients cannot capture this, but a GNN with stage-conditioned features can.
5. **Directionality is underused in human work.** Mouse studies fit separate antero/retrograde
   terms; human studies almost always symmetrise. Our mouse data has both directed connectomes.

---

## 9. Recommendations for the experiment

**Data.** Run **both** arms. The mouse arm (`datasets/mouse_tau_longitudinal/`) is the only place
H1 can be tested as literally stated — *longitudinal* propagation with a known seed, 14
experiments, held-out-experiment evaluation. The human arm (`datasets/human_tau_adni/`) provides
species relevance and a genuine independent replication cohort, but is stage-stratified rather
than within-subject longitudinal, and this must be stated plainly in any write-up.

**Models.** Prefer a **diffusion-constrained / physics-informed GNN** (graph neural ODE, or a GNN
whose message passing is initialised at the graph Laplacian) over a generic GCN. With ~86–426
nodes and few timepoints, an unconstrained high-capacity model will overfit and the comparison
will be uninformative.

**H2 protocol.** Ablate: {no genes} → {MAPT, APOE, TREM2} → {AD-risk panel} → {size-matched random
gene sets} → {all genes via PCA}. The random-gene-set control is essential; without it a "gene
effect" may just be the cortical hierarchy. Compare against **eNDM/NexIS**, which already encodes
gene modulation classically.

**Reporting.** Report the persistence baseline in every table. Report where the GNN *fails* —
given Schäfer's null-model results, a negative or null result here is a legitimate and publishable
finding, and is a more likely outcome than a large win.

---

## 10. Key references

1. Raj A, Kuceyeski A, Weiner M (2012). A network diffusion model of disease progression in dementia. *Neuron* 73(6):1204–15.
2. Raj A, LoCastro E, Kuceyeski A, et al. (2015). Network diffusion model of progression predicts longitudinal patterns of atrophy and metabolism in Alzheimer's disease. *Cell Reports* 10(3):359–69.
3. Iturria-Medina Y, Sotero RC, Toussaint PJ, Evans AC (2014). Epidemic spreading model to characterize misfolded proteins propagation. *PLoS Comput Biol* 10(11):e1003956.
4. **Schäfer A, Mormino EC, Kuhl E (2020). Network diffusion modeling explains longitudinal tau PET data. *Front Neurosci* 14:566876.**
5. Vogel JW, Iturria-Medina Y, Strandberg OT, et al. (2020). Spread of pathological tau proteins through communicating neurons. *Nat Commun* 11:2612.
6. Franzmeier N, Dewenter A, Frontzkowski L, et al. (2020). Patient-centered connectivity-based prediction of tau pathology spread. *Sci Adv* 6:eabd1327.
7. **Cornblath EJ, Li HL, Changolkar L, et al. (2021). Computational modeling of tau pathology spread reveals patterns of regional vulnerability. *Sci Adv* 7:eabg6677.**
8. **Henderson MX, Cornblath EJ, Darwich A, et al. (2019). Spread of α-synuclein pathology through the brain connectome is modulated by selective vulnerability. *Nat Neurosci* 22:1248–57.**
9. **Zheng YQ, Zhang Y, Yau Y, et al. (2019). Local vulnerability and global connectivity jointly shape neurodegenerative disease propagation. *PLoS Biol* 17(11):e3000495.**
10. Anand C, Torok J, Abdelnour F, et al. (2022). The effects of microglia on tauopathy progression can be quantified using Nexopathy in silico (Nexis) models. *Sci Rep* 12:21170.
11. Li X, Zhou Y, Dvornek N, et al. (2021). BrainGNN: Interpretable brain graph neural network for fMRI analysis. *Med Image Anal* 74:102233.
12. Markello RD, Arnatkevičiūtė A, Poline JB, et al. (2021). Standardizing workflows in imaging transcriptomics with the abagen toolbox. *eLife* 10:e72129.
13. Larivière S, Paquola C, Park BY, et al. (2021). The ENIGMA Toolbox. *Nat Methods* 18:698–700.
14. Arnatkevičiūtė A, Fulcher BD, Fornito A (2019). A practical guide to linking brain-wide gene expression and neuroimaging data. *NeuroImage* 189:353–67.
15. Garbarino S, Lorenzi M (2021). Investigating hypotheses of neurodegeneration by learning dynamical systems of protein propagation in the brain. *NeuroImage* 235:117980.
