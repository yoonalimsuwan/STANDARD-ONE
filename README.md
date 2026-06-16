``
# STANDARD ONE 


Unified Differentiable Framework for Particle & Cosmos Physics

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20007526-blue)](https://doi.org/10.5281/zenodo.20007526)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19814975-blue)](https://doi.org/10.5281/zenodo.19814975)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19869633-blue)](https://doi.org/10.5281/zenodo.19869633)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20194882-blue)](https://doi.org/10.5281/zenodo.20194882)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20718920-blue)](https://doi.org/10.5281/zenodo.20718920)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20482078-blue)](https://doi.org/10.5281/zenodo.20482078)


Democratizing frontier physics through differentiable programming, from quarks to the cosmos.

STANDARD ONE is a comprehensive, fully differentiable, multi‑paradigm statistical engine for fundamental physics research. Built entirely on PyTorch, it unifies Bayesian, Frequentist, and Structural Deterministic Probability within a single lightweight infrastructure. The framework covers the Standard Model particles and all four fundamental forces, parton distribution functions (PDFs) with DGLAP evolution, hard‑process matrix elements, collider event simulation, full cosmological observations (CMB, Planck data), black‑hole thermodynamics, dark matter, vacuum energy, unification models, and now — via its companion module YANG MILLS MASS GAP (ONE) — the Yang–Mills mass gap problem. Every component is end‑to‑end differentiable, enabling gradient‑based optimisation, Bayesian inference, and model comparison across particle physics and cosmology.

---

Key Features

🔬 Particle & Nuclear Physics

· Complete Standard Model particle database (masses, charges, spins, colour, weak isospin, hypercharge).
· Differentiable electroweak parameters (α, α_s, G_F, M_Z, sin²θ_W) and running couplings.
· Parton Distribution Functions with multiple backends:
  · Differentiable DGLAP evolution in Mellin space (LO/NLO, exact singlet/non‑singlet anomalous dimensions).
  · Neural PDF surrogate trained on LHAPDF grids for fast, differentiable evaluation.
  · Direct LHAPDF interface for non‑differentiable checks.
· Matrix Elements for QED, QCD, and electroweak processes with NNLO K‑factors (Drell‑Yan, gluon‑fusion Higgs).
· Collider event generation:
  · Structural generators (crystal‑ball, exponential, etc.) and physical cross‑section generators.
  · Integration with Pythia8 and Herwig for full parton shower and hadronization (optional, GPL).
  · Differentiable detector simulation (fast, trainable resolution and efficiency model).
  · Support for CERN Open Data (ROOT via uproot/awkward) and pyhf workspace likelihoods.

🌌 Cosmology & Astrophysics

· Cosmology engine with trainable parameters (H₀, Ω_b, Ω_c, Ω_Λ, w, T_cmb, N_eff).
· Differentiable CMB power spectra via:
  · CosmoPower neural emulator (fast, differentiable).
  · CAMB and CLASS Boltzmann solvers (optional, GPL).
  · Built‑in neural emulator (trainable on CAMB outputs).
  · Analytic Hu–White approximation.
  · Automatic backend selection (auto chooses best available).
· Planck 2018 data loader (high‑ℓ TT spectrum, covariance) with full Gaussian likelihood.
· Cross‑correlation between collider and CMB observables.

🕳️ Gravity & Quantum Fields

· Black‑hole thermodynamics (Hawking, Page, primordial BH models) with differentiable spectra.
· Dark matter models (WIMP, axion, sterile, fuzzy) as differentiable generators.
· Vacuum energy & extraction models (zero‑point, Casimir, quintessence, holographic).
· Unification models (running SU(3), SU(2), U(1) couplings, Randall–Sundrum warp factor).

🧮 Statistical & Mathematical Methods

· Three statistical paradigms in one framework:
  · Frequentist: profile likelihood, significance, CLs, confidence intervals, bootstrap calibration.
  · Bayesian: Laplace approximation, adaptive MCMC, NUTS (via Pyro), marginal likelihood, Bayes factors.
  · Structural Deterministic Probability: CSOC kernel, semantic state contraction (SSC), renormalization group (RG), BV consistency.
· Model comparison with AIC, BIC, and Bayes factors.
· End‑to‑end automatic differentiation — all physical and statistical parameters are trainable via PyTorch.

🧬 YANG MILLS MASS GAP (ONE) 

A dedicated differentiable extension for the Yang–Mills mass gap problem:

· Parametric gluon propagator models (Gribov, massive, refined Gribov, decoupling/scaling).
· Differentiable complex‑pole analysis for mass gap extraction.
· Fitting to lattice QCD data (or synthetic) using gradient descent.
· Seamless integration with STANDARD ONE’s running coupling and CSOC regularisation.
· Does not require modification of STANDARD ONE — instant plug‑in.

🖥️ Hardware & Portability

· Multi‑backend: CPU, CUDA, Apple MPS, Ascend NPU.
· Lightweight: runs on 3 GB RAM, Google Colab T4, Apple Silicon, Chinese chips.
· Pure Python; no compilation required.

---

Installation

STANDARD ONE requires Python ≥ 3.8 and PyTorch ≥ 1.12. Install the core dependencies:

```bash
pip install torch numpy scipy matplotlib
# Optional but recommended
pip install uproot awkward astropy pyhf pywt pyro-ppl cosmopower
# For advanced features (GPL‑licensed, optional)
pip install lhapdf camb classy pythia8 herwig  # may need manual builds
```

Then clone or download the repository:

```bash
git clone https://github.com/yoonalimsuwan/STANDARD-ONE.git
cd standard-one
```

For the Yang–Mills extension, simply copy yang_mills_mass_gap_one.py into the same directory.

---

Quick Start

```python
from standard_one import StandardOneUnified

config = {
    'physics': 'collider',
    'mass_min': 50, 'mass_max': 200,
    'n_events': 2000,
    'use_physical_cross_section': True,
    'process': 'drell_yan', 'sqrts': 13000
}

# Initialize framework
framework = StandardOneUnified(config, device='cpu')

# Load simulated collider data and run a frequentist fit
framework.load_collider_data(source='simulate', n_samples=1500)
framework.train_soc_gradient(n_steps=100)
result = framework.run_full_frequentist(poi_name='signal_mass')
print(result)

# Bayesian analysis
bayes_result = framework.run_full_bayesian(n_samples=1000)
print(bayes_result['map'])

# Validate against real ATLAS Z→μμ data
framework.validate_against_atlas_zmumu(sqrts=13000)

# CMB fit
cmb_fit = framework.run_cmb_fit(cmb_backend='auto')
print(cmb_fit)
```

---

Yang–Mills Mass Gap (Plug‑in Example)

No modifications to STANDARD ONE needed. Just create a YangMillsMassGap object using the framework’s physics parameters and CSOC kernel:

```python
from yang_mills_mass_gap_one import YangMillsMassGap

# Use existing physics & CSOC from your STANDARD ONE instance
ym = YangMillsMassGap(
    physics_params=framework.physics,
    csoc=framework.csoc,
    propagator_type='refined',   # choose 'gribov', 'massive', or 'refined'
    device='cpu'
)

# Fit to lattice gluon propagator data (CSV or synthetic)
ym.fit_to_lattice("gluon_propagator.csv", epochs=500, lr=0.01)

# Extract mass gap in GeV
mass_gap = ym.extract_mass_gap(method='pole_scan')
print(f"Yang–Mills mass gap = {mass_gap*1000:.1f} MeV")

# Bayesian inference on mass gap
import torch
def log_prob_fn():
    return -ym.likelihood(p2_data, D_data)
from standard_one import BayesianAnalysis
params = list(ym.propagator.parameters())
bayes = BayesianAnalysis(log_prob_fn, params, ['log_M4','log_m2'], device='cpu')
map_est = bayes.laplace_approximation()
```

---

Examples & Tutorials

· Higgs mass fit demo
    python standard_one.py --higgs-demo
    Fits the Z boson mass peak using a crystal‑ball signal + exponential background, comparing frequentist and Bayesian results.
· Full chain simulation with Pythia8 and detector
    python standard_one.py --physics collider --use-physical-xsec --data-source full_simulation --shower pythia8 --frequentist
· CMB power spectrum fitting
    python standard_one.py --cmb-fit --cmb-backend cosmopower
· Yang–Mills mass gap standalone
    python yang_mills_mass_gap_one.py
    (requires STANDARD ONE in the path) – demonstrates a propagator fit and mass gap extraction.
· Run all validation tests
    python standard_one.py --test

---

### ⚡ AI‑Accelerated O(1) Speed

STANDARD ONE is designed from the ground up to leverage **neural surrogates** — including the built‑in neural PDF, CMB emulator, detector simulator, and the Yang‑Mills propagator models. Once trained, these differentiable networks replace iterative computations (DGLAP evolution, Boltzmann integrals, detector smearing, pole scanning) with a single feed‑forward pass. The result: **near‑O(1) evaluation time** per physics prediction, independent of the underlying complexity.

This unlocks:
- **Real‑time inference** for collider trigger decisions or cosmological parameter updates.
- **Massive parameter scans** (billions of points) that were previously intractable.
- **Active learning and Bayesian optimization** with negligible latency per proposal.
- **Edge deployment** — run on lightweight hardware while preserving full physics fidelity.

Simply train a surrogate once, then call it as a drop‑in replacement for the exact model — the framework automatically routes to the fastest differentiable path.

Documentation & Citation

Full API documentation and detailed physics notes are available in the docs/ directory.
If you use STANDARD ONE or YANG MILLS MASS GAP (ONE) in your research, please cite:

```
@software{standardone2026,
  author       = {Yoon A Limsuwan},
  title        = {STANDARD ONE: Unified Differentiable Framework for Particle \& Cosmos Physics},
  year         = 2026,
  doi          = {DOI: 10.5281/zenodo.20364171},
  url          = https://github.com/yoonalimsuwan/STANDARD-ONE
}

@software{yangmillsmassgap2026,
  author       = {Yoon A Limsuwan},
  title        = {YANG MILLS MASS GAP (ONE): Differentiable Explorer for the Yang-Mills Mass Gap},
  year         = 2026,
  doi          = https://doi.org/10.5281/zenodo.20718920 ,
  note         = {Extension to STANDARD ONE}
}
```

---

License

STANDARD ONE is released under the MIT License.
YANG MILLS MASS GAP (ONE) is also MIT‑licensed.
External optional libraries (LHAPDF, CAMB, CLASS, Pythia8, Herwig) retain their own licenses; if linked, the combined work must comply with those terms. To remain pure MIT, rely on the built‑in neural PDF, neural CMB emulator, and structural collider generator.

---

Contributing & Community

We welcome contributions from the worldwide physics and AI communities. Please see CONTRIBUTING.md for guidelines.
This software is intended exclusively for peaceful civilian applications.

---

STANDARD ONE is more than a toolbox — it is a new paradigm for differentiable, multi‑paradigm physics. Together with YANG MILLS MASS GAP (ONE), it opens a path to solving the deepest problems of nature using the full power of modern machine learning and statistical inference.
