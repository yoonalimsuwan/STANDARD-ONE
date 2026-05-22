``
# STANDARD ONE

**Unified Differentiable Framework for Particle & Cosmos Physics**

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20007526-blue)](https://doi.org/10.5281/zenodo.20007526)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19814975-blue)](https://doi.org/10.5281/zenodo.19814975)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20194882-blue)](https://doi.org/10.5281/zenodo.20194882)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20343293-blue)](https://doi.org/10.5281/zenodo.20343293)

STANDARD ONE is a comprehensive, fully differentiable, multi‑paradigm statistical engine for frontier research in fundamental physics. Built entirely on PyTorch, it integrates Bayesian, Frequentist, and Structural Deterministic Probability into a single lightweight infrastructure. It covers the Standard Model particles, all four fundamental forces, parton distribution functions, hard‑process matrix elements, collider event simulation, cosmological observations (CMB, Planck data), black‑hole thermodynamics, dark matter, vacuum energy, and unification models. The entire framework is end‑to‑end differentiable, enabling gradient‑based optimisation of every physical and statistical parameter.

---

## Key Features

- **Complete Standard Model** – quarks, leptons, gauge bosons, Higgs with full quantum numbers (charge, spin, colour, weak isospin, hypercharge).
- **Differentiable Forces** – 2‑loop running couplings with smooth flavour thresholds, plus electroweak and gravitational parameters.
- **Parton Distribution Functions** – MIT‑safe parametric form (differentiable) or optional LHAPDF grid interpolation.
- **Matrix Elements & K‑factors** – QED, QCD, electroweak, Drell‑Yan, gg→Higgs, all with differentiable higher‑order corrections.
- **Collider & Cosmology Data Loaders** – CERN ROOT files, pyhf HistFactory workspaces, NASA FITS/CSV, Planck CMB spectra.
- **Differentiable CMB** – Hu & White analytic TT spectrum (fully differentiable) ready for gradient‑based cosmological parameter inference.
- **Alternative Physics Models** – Black‑hole (Hawking, Page, PBH), dark matter (WIMP, axion, sterile, fuzzy), vacuum energy (Casimir, quintessence, holographic), vacuum extraction (dynamical Casimir, Schwinger), unification (Randall‑Sundrum, running couplings).
- **Structural Components** – Learnable Coupled Self‑Organised Criticality (CSOC) kernel, Semantic State Contraction (SSC), Differentiable RG refiner, Bias–Variance consistency check.
- **Full Statistical Toolbox**  
  *Frequentist*: profile likelihood ratio, asymptotic significance (Z), p‑values, confidence intervals, upper limits.  
  *Bayesian*: Metropolis‑Hastings, NUTS (via Pyro), Laplace approximation, Bayes factors.  
  *Structural Probability*: deterministic probability statements with an unresolved interface Γ.
- **Model Comparison** – AIC, BIC, posterior predictive checks, Bayes factors.
- **Cross‑Correlation** – simple neural connector between collider and cosmological observables.
- **O(1) Inference & Differentiable Emulators** – train constant‑time neural surrogates for any component of the pipeline.  
  The differentiable structure allows high‑fidelity emulators (e.g., for CMB spectra, cross‑sections, PDFs, or full generator outputs) to be trained with exact gradients. Once trained, these emulators evaluate in a fixed number of operations, independent of data size or parameter complexity. The built‑in analytic CMB, learnable CSOC kernel, and RG refiner already provide O(1) building blocks, and the framework is designed to replace expensive routines with lightweight neural surrogates for real‑time inference and large‑scale MCMC.
- **Multi‑backend** – runs on CPU, CUDA, Apple MPS, and Ascend NPU with automatic fallback.
- **Lightweight** – fits within 3 GB RAM, runs on a Colab T4 or Apple Silicon.

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ (with your preferred backend)

### Basic Installation
```bash
pip install torch numpy scipy matplotlib
```

Optional Dependencies

For full functionality install any combination of the following:

```bash
pip install uproot awkward          # CERN ROOT I/O
pip install astropy                 # NASA FITS & CSV tables
pip install pyhf                    # differentiable HistFactory models
pip install pywt                    # wavelet denoising
pip install lhapdf-management      # PDF grids (GPL – use parametric PDF to avoid copyleft)
pip install pyro-ppl                # advanced MCMC (NUTS)
# cosmo‑power is not required; analytic CMB is built‑in
```

Clone the repository:

```bash
git clone https://github.com/yoonalimsuwan/STANDARD-ONE.git
cd standard-one
```

---

Quick Start

Run the built‑in validation tests to verify the physics:

```bash
python standard_one.py --test
```

Generate a collider mass spectrum and fit its CSOC parameters:

```bash
python standard_one.py --physics collider --train-soc
```

Perform a full Frequentist analysis on the collider model:

```bash
python standard_one.py --physics collider --frequentist --poi log_mu
```

Fit the CMB TT power spectrum using gradient descent:

```bash
python standard_one.py --physics cmb --cmb-fit
```

---

Programmatic Usage

```python
from standard_one import StandardOneUnified

config = {
    'physics': 'dark_matter',
    'dm_model': 'wimp',
    'dm_mass': 100.0,
    'mass_min': 0.1,
    'mass_max': 200,
    'n_events': 500
}

fw = StandardOneUnified(config, device='cuda')
fw.load_collider_data(source='simulate')

# Frequentist analysis
results = fw.run_full_frequentist(poi_name='log_mu', null=0.0, cl=0.68)
print(results['significance'], results['conf_interval'])

# Bayesian analysis
bayes_res = fw.run_full_bayesian(use_nuts=True)
print(bayes_res['map'])

# Compare different DM models
from standard_one import DarkMatterGenerator, CSOCKernel, SemanticStateContraction, DiffRGRefiner

gen_wimp = DarkMatterGenerator('wimp', dm_mass=100, csoc=CSOCKernel(), ...)
gen_axion = DarkMatterGenerator('axion', dm_mass=100, csoc=CSOCKernel(), ...)
comparison = fw.model_comparison([gen_wimp, gen_axion])
print(comparison)
```

---

Core Components

Particle Database (ParticleDB)

Full SM particle masses, PDG IDs, and quantum numbers.

Force Parameters (ForceParameters)

Differentiable 2‑loop α_s(μ) with smooth n_f(μ), constant α_EM, G_F, G_N.

PDF Provider (PDFProvider)

Uses either LHAPDF grids or a fully differentiable parametric form with 8 trainable parameters per flavour.

Matrix Elements (MatrixElements)

LO squared amplitudes for key processes, each equipped with a differentiable K‑factor.

Differentiable CMB (DifferentiableCMB)

Implements the Hu & White (1997) analytic TT spectrum, fully embedded in PyTorch for direct gradient‑based fits to Planck data.

Structural Components

· CSOC Kernel: learnable Cs * r^{‑α} * exp(‑r/λ) with 5 trainable parameters.
· SSC: stabilising state contraction filter.
· RG Refiner: Fourier‑space low‑pass filter (RG‑inspired).
· BV Consistency: simple bias–variance diagnostic.

Generators (ColliderGenerator, BlackHoleGenerator, DarkMatterGenerator)

Each combines a physics signal/background model with the CSOC kernel, SSC, and RG refiner. All parameters are differentiable.

Statistical Engines

· FrequentistAnalysis: profile likelihood, q₀, asymptotic significance, confidence intervals, upper limits.
· BayesianAnalysis: MH, NUTS, Laplace approximation, marginal likelihood, Bayes factors.
· StructuralProbability: deterministic probability from the generator’s PDF.

---

O(1) Speed & Emulation

The entire framework is designed to enable constant‑time (O(1)) evaluation for complex physical calculations. Because every component is differentiable, you can train lightweight neural surrogates (emulators) to replace expensive numerical routines while retaining full physics fidelity.

· CSOC & RG as O(1) building blocks – The learnable CSOC kernel and Fourier RG refiner provide analytical forms that mimic power‑law tails and renormalisation group flows, delivering predictions in a fixed number of floating‑point operations.
· Differentiable CMB – The built‑in Hu & White spectrum already computes Cₗ in O(1) per multipole. The framework can also be used to train a neural CMB emulator (e.g., CosmoPower‑style) using exact gradients, reducing full‑sky likelihood evaluations to milliseconds.
· End‑to‑end emulation – All generator and likelihood classes expose differentiable forward passes. By wrapping any part of the pipeline (cross‑section calculations, PDF convolutions, detector response) with a small neural network and training it via gradient descent, you obtain an O(1) surrogate that can be deployed in real‑time analysis, embedded systems, or large‑scale MCMC chains.
· Edge & Colab readiness – Once trained, these emulators require minimal computation, making it feasible to run state‑of‑the‑art physics inference on a laptop, a Colab GPU, or even a mobile device.

This capability makes STANDARD ONE not only a research tool but a production‑ready engine for accelerated discovery.

---

Philosophy

STANDARD ONE treats probability as an emergent structural property, not a fundamental randomness. The CSOC kernel and associated components implement the idea that all apparent stochasticity arises from the unresolved interface Γ. Once Γ is fully specified, outcomes are deterministic. This perspective unifies frequentist and Bayesian views and offers a novel pathway for interpretable AI in fundamental physics.

---

License

This project is distributed under the MIT License (see LICENSE).
External libraries retain their own licences. To avoid copyleft restrictions from GPL‑licensed LHAPDF, the built‑in parametric PDF can be used freely.

---

Citation

If you use STANDARD ONE in your research, please cite:

```
Yoon A Limsuwan. (2026). STANDARD ONE: Unified Differentiable Framework for Particle & Cosmos Physics.
https://github.com/yoonalimsuwan/STANDARD-ONE

```

---

Disclaimer

This software is intended exclusively for peaceful civilian applications.

---

Contact

For questions, collaboration, or commercial licensing, please open an issue or contact the author.

---

Ready for frontier research.
STANDARD ONE – the differentiable backbone for 21st‑century physics.

```
Thank you.
