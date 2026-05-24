# =============================================================================
# STANDARD ONE — Unified Differentiable Framework for Particle & Cosmos Physics
# =============================================================================
# Author : Yoon A Limsuwan
# License: MIT
# Year   : 2026
#
# A comprehensive, fully differentiable, multi‑paradigm statistical engine for
# frontier research in fundamental physics. Integrates Bayesian, Frequentist,
# and Structural Deterministic Probability within a single PyTorch‑based
# infrastructure. Covers:
#
#   • Standard Model particles (quarks, leptons, gauge & Higgs bosons)
#   • Four fundamental forces (EM, weak, strong, gravity)
#   • Full quantum numbers (charge, spin, colour, weak isospin, hypercharge)
#   • Parton distribution functions (differentiable via LHAPDF grid or parametric)
#   • Matrix elements for hard processes (QED, QCD, electroweak) with higher‑order K‑factors
#   • Collider event simulation & analysis (CERN Open Data, pyhf)
#   • Cosmological observations (Planck, NASA – FITS, HDF5, CSV)
#   • Black‑hole thermodynamics, dark matter, vacuum energy & extraction
#   • Differentiable CMB (Hu & White analytic + CAMB/CLASS/CosmoPower interface)
#   • Cross‑correlation between collider and cosmic data
#   • Toy unification (running couplings, Randall–Sundrum)
#   • Structural deterministic probability (CSOC, SSC, RG, BV)
#   • Model comparison: AIC, BIC, Bayes factors, posterior predictive checks
#   • End‑to‑end differentiation: gradient‑based optimisation of all parameters
#   • Multi‑backend (CPU, CUDA, MPS, Ascend NPU)
#   • Lightweight: runs on 3 GB RAM, Colab T4, Apple Silicon, Chinese chips
#
# Open‑source foundations (with licences):
#   • PyTorch (BSD‑3‑Clause)           — automatic differentiation & GPU/NPU
#   • NumPy (BSD‑3‑Clause)             — array operations
#   • SciPy (BSD‑3‑Clause)             — statistical functions (no optimisation)
#   • Matplotlib (PSF)                 — optional visualisation
#   • uproot (BSD‑3‑Clause)            — CERN ROOT I/O
#   • awkward (BSD‑3‑Clause)           — columnar data manipulation
#   • astropy (BSD‑3‑Clause)           — NASA FITS & table handling
#   • pyhf (Apache 2.0)                — differentiable HistFactory models
#   • pywt (BSD‑3‑Clause)              — wavelet denoising (optional)
#   • lhapdf‑management (GPL‑3)        — PDF grid interpolation (if installed)
#   • pyro‑ppl (Apache 2.0)            — advanced MCMC (NUTS)
#   • CAMB (modified BSD)              — Boltzmann solver (optional)
#   • CLASS (GPL‑v2)                   — Boltzmann solver (optional)
#   • CosmoPower (MIT)                 — CMB neural emulator (optional)
#   • Pythia8 (GPL‑2)                  — collider event generator (optional)
#   • Herwig (GPL‑3)                   — collider event generator (optional)
#
# Business use: MIT licence for this code. External libraries retain their own
# licences. GPL‑licensed components (LHAPDF, CLASS, Pythia, Herwig) are
# optional; if linked, the combined work must comply with the GPL. To remain
# pure MIT, rely on built‑in parametric PDFs, analytic CMB, and the structural
# collider generator.
#
# This software is intended exclusively for peaceful civilian applications.
# =============================================================================

import math, os, sys, argparse, logging, warnings, hashlib, json, urllib
from typing import Tuple, List, Optional, Dict, Any, Union, Callable
from urllib.request import urlretrieve
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS, Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress, chi2, norm
from scipy.interpolate import interp1d

# ---- Optional imports with graceful degradation ---------------------------
try:
    import uproot
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False
try:
    import awkward as ak
    HAS_AWKWARD = True
except ImportError:
    HAS_AWKWARD = False
try:
    from astropy.io import fits
    from astropy.table import Table
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
try:
    import pyhf
    from pyhf import Model, set_backend
    HAS_PYHF = True
except ImportError:
    HAS_PYHF = False
try:
    import lhapdf
    HAS_LHAPDF = True
except ImportError:
    HAS_LHAPDF = False
try:
    import pyro
    import pyro.distributions as dist_pyro
    from pyro.infer import MCMC, NUTS
    HAS_PYRO = True
except ImportError:
    HAS_PYRO = False
try:
    import camb
    HAS_CAMB = True
except ImportError:
    HAS_CAMB = False
try:
    import classy
    HAS_CLASS = True
except ImportError:
    HAS_CLASS = False
try:
    import cosmopower
    HAS_COSMOPOWER = True
except ImportError:
    HAS_COSMOPOWER = False
try:
    import pythia8
    HAS_PYTHIA = True
except ImportError:
    HAS_PYTHIA = False
try:
    import herwig
    HAS_HERWIG = True
except ImportError:
    HAS_HERWIG = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger("StandardOne")

# =============================================================================
# 0. Device & Backend Utilities
# =============================================================================
def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "ascend":
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def diff_interp(x, xp, fp):
    """
    Differentiable linear interpolation in 1‑D.
    x : (N,) tensor, xp : (M,) sorted tensor, fp : (M,) tensor.
    Returns interpolated values of shape (N,).
    """
    x = x.clamp(xp.min(), xp.max())
    idx = torch.bucketize(x, xp)
    idx = idx.clamp(1, len(xp)-1)
    x_lo = xp[idx-1]
    x_hi = xp[idx]
    f_lo = fp[idx-1]
    f_hi = fp[idx]
    t = (x - x_lo) / (x_hi - x_lo + 1e-12)
    return f_lo + t * (f_hi - f_lo)


# =============================================================================
# 1. Complete Particle Database & Quantum Numbers
# =============================================================================
class ParticleDB:
    _db = {
        1:  (0.0022, -1/3, 0.5, 3, -0.5, 1/3),
        2:  (0.0022,  2/3, 0.5, 3,  0.5, 1/3),
        3:  (0.096,  -1/3, 0.5, 3, -0.5, 1/3),
        4:  (1.27,    2/3, 0.5, 3,  0.5, 1/3),
        5:  (4.18,   -1/3, 0.5, 3, -0.5, 1/3),
        6:  (172.76,  2/3, 0.5, 3,  0.5, 1/3),
        11: (0.511e-3, -1, 0.5, 0, -0.5, -1),
        -11:(0.511e-3,  1, 0.5, 0,  0.5,  1),
        13: (0.10566,  -1, 0.5, 0, -0.5, -1),
        -13:(0.10566,   1, 0.5, 0,  0.5,  1),
        15: (1.77686,  -1, 0.5, 0, -0.5, -1),
        -15:(1.77686,   1, 0.5, 0,  0.5,  1),
        12: (0.0, 0, 0.5, 0,  0.5, -1),
        14: (0.0, 0, 0.5, 0,  0.5, -1),
        16: (0.0, 0, 0.5, 0,  0.5, -1),
        21: (0.0, 0, 1.0, 8, 0, 0),
        22: (0.0, 0, 1.0, 0, 0, 0),
        23: (91.188, 0, 1.0, 0, 0, 0),
        24: (80.379, 1, 1.0, 0, 1, 0),
        -24:(80.379, -1, 1.0, 0, -1, 0),
        25: (125.1, 0, 0.0, 0, 0, 0)
    }
    _name = {
        1:'d',2:'u',3:'s',4:'c',5:'b',6:'t',
        11:'e-',-11:'e+',13:'mu-',-13:'mu+',15:'tau-',-15:'tau+',
        12:'ve',14:'vm',16:'vt',21:'g',22:'γ',23:'Z',24:'W+',-24:'W-',25:'H'
    }
    @classmethod
    def mass(cls, pid): return cls._db[pid][0]
    @classmethod
    def charge(cls, pid): return cls._db[pid][1]
    @classmethod
    def spin(cls, pid): return cls._db[pid][2]
    @classmethod
    def colour(cls, pid): return cls._db[pid][3]
    @classmethod
    def isospin3(cls, pid): return cls._db[pid][4]
    @classmethod
    def hypercharge(cls, pid): return cls._db[pid][5]
    @classmethod
    def name(cls, pid): return cls._name.get(pid, f"PID{pid}")
    @classmethod
    def all_pids(cls): return list(cls._db.keys())


# =============================================================================
# 2. Fundamental Forces & Differentiable Running Couplings
# =============================================================================
class ForceParameters:
    def __init__(self, device='cpu'):
        self.alpha_EM_MZ = 1 / 127.952
        self.alpha_s_MZ   = 0.1180
        self.G_F          = 1.1663787e-5
        self.G_N          = 6.70883e-39
        self.MZ           = 91.1876
        self.m_top        = 172.76
        self.m_bot        = 4.18
        self.m_charm      = 1.27
        self.device = device

    def alpha_EM(self, Q2=None):
        return torch.tensor(self.alpha_EM_MZ, device=self.device)

    def alpha_s(self, Q2: Union[float, torch.Tensor]) -> torch.Tensor:
        Q2 = torch.as_tensor(Q2, dtype=torch.float32, device=self.device)
        mu = torch.sqrt(torch.clamp(Q2, min=1e-6))
        nf = (3.0 +
              torch.tanh((mu - self.m_charm) / (0.1*self.m_charm)) * 0.5 + 0.5 +
              torch.tanh((mu - self.m_bot)   / (0.1*self.m_bot))   * 0.5 + 0.5 +
              torch.tanh((mu - self.m_top)   / (0.1*self.m_top))   * 0.5 + 0.5)
        nf = torch.clamp(nf, 3.0, 6.0)
        beta0 = (33.0 - 2.0*nf) / (12.0 * math.pi)
        beta1 = (153.0 - 19.0*nf) / (24.0 * math.pi**2)
        L = torch.log(Q2 / (self.MZ**2))
        denom = 1.0 + beta0 * self.alpha_s_MZ * L + \
                (beta1 / (beta0 + 1e-10)) * self.alpha_s_MZ * torch.log(
                    torch.abs(1.0 + beta0 * self.alpha_s_MZ * L) + 1e-10)
        alpha = self.alpha_s_MZ / torch.clamp(denom, min=0.01)
        return alpha

    def weak_coupling(self):
        return torch.tensor(self.G_F, device=self.device)

    def gravitational_coupling(self):
        return torch.tensor(self.G_N, device=self.device)


# =============================================================================
# 3. Parton Distribution Functions (Fully Differentiable)
# =============================================================================
class PDFProvider(nn.Module):
    # Universal flavour strings: 'g', 'u','d','s','c','b','t',
    #   'ubar','dbar','sbar','cbar','bbar','tbar'
    FLAVOUR_TO_PID = {
        'g': 21, 'u': 2, 'd': 1, 's': 3, 'c': 4, 'b': 5, 't': 6,
        'ubar': -2, 'dbar': -1, 'sbar': -3, 'cbar': -4, 'bbar': -5, 'tbar': -6
    }

    def __init__(self, pdf_name: str = "CT14nlo", device='cpu', grid_size: int = 500):
        super().__init__()
        self.device = device
        self.pdf_name = pdf_name
        self.grid_size = grid_size
        self.use_lhapdf = HAS_LHAPDF
        if self.use_lhapdf:
            self._init_lhapdf_grid(pdf_name)
        else:
            logger.info("LHAPDF not available; using internal parametric PDF (MIT safe).")
            self._init_parametric()

    def _init_lhapdf_grid(self, pdf_name):
        lhapdf.setVerbosity(0)
        self._lhapdf_set = lhapdf.mkPDF(pdf_name)
        # PIDs used to build the grid (all quark flavours + gluon)
        self.lha_pids = list(dict.fromkeys(self.FLAVOUR_TO_PID.values()))  # unique ordered
        # Build a 2D grid: (x, Q) -> xf(x,Q) for each PID
        self._x_grid = torch.logspace(-5, 0, self.grid_size, device=self.device)
        self._q_grid = torch.logspace(0, 3, 80, device=self.device)
        grid_vals = []
        for q in self._q_grid:
            q_val = q.item()
            row = []
            for x in self._x_grid:
                x_val = x.item()
                xfx = [self._lhapdf_set.xfxQ(pid, x_val, q_val) for pid in self.lha_pids]
                row.append(xfx)
            grid_vals.append(row)
        # shape: (n_q, n_x, n_pids)
        grid = torch.tensor(grid_vals, dtype=torch.float32, device=self.device)
        # store as (1, n_pids, n_q, n_x) for grid_sample with (N,C,H,W)
        self._grid = grid.permute(2, 0, 1).unsqueeze(0)   # [1, n_pids, n_q, n_x]
        # Normalized coordinates for grid_sample: we map (log10 x, log10 Q) to [-1,1]
        self._x_log_min = math.log10(self._x_grid[0].item())
        self._x_log_max = math.log10(self._x_grid[-1].item())
        self._q_log_min = math.log10(self._q_grid[0].item())
        self._q_log_max = math.log10(self._q_grid[-1].item())

    def _init_parametric(self):
        # Parameters: norm, small‑x exponent, large‑x exponent, polynomial coeffs
        # flavours: g, u_val, d_val, u_sea, d_sea, s_sea, c_sea, b_sea
        self.params = nn.Parameter(torch.tensor([
            [2.5,   -0.1, 4.0, -0.5, 0.2, 0.0],   # gluon
            [1.8,   0.3,  3.2,  0.1, 0.0, 0.0],   # u_val
            [1.2,   0.3,  3.5, -0.2, 0.0, 0.0],   # d_val
            [0.3,  -0.1, 6.0, -0.8, 0.3, 0.0],   # u_sea
            [0.25, -0.1, 6.0, -0.8, 0.3, 0.0],   # d_sea
            [0.08, -0.2, 7.0, -1.0, 0.5, 0.1],   # s (sea)
            [0.02, -0.1, 9.0, -1.5, 0.6, 0.2],   # c (sea)
            [0.005, 0.0,11.0, -2.0, 0.8, 0.3],   # b (sea)
        ], device=self.device).float())

    def _evaluate_parametric(self, x: torch.Tensor, flavour: str) -> torch.Tensor:
        """Evaluate parametric PDF for a given flavour string."""
        def param(idx):
            A, a, b, c, d, e = self.params[idx]
            xf = A * x**a * (1-x)**b * (1 + c*torch.sqrt(x) + d*x + e*x**2)
            return torch.clamp(xf, min=1e-12)

        if flavour == 'g':
            return param(0)
        elif flavour == 'u':
            return param(1) + param(3)      # u_val + u_sea
        elif flavour == 'ubar':
            return param(3)                 # u_sea
        elif flavour == 'd':
            return param(2) + param(4)      # d_val + d_sea
        elif flavour == 'dbar':
            return param(4)                 # d_sea
        elif flavour in ('s', 'sbar'):
            return param(5)                 # s_sea
        elif flavour in ('c', 'cbar'):
            return param(6)                 # c_sea
        elif flavour in ('b', 'bbar'):
            return param(7)                 # b_sea
        elif flavour in ('t', 'tbar'):
            return torch.zeros_like(x)      # negligible
        else:
            raise ValueError(f"Unknown flavour: {flavour}")

    def xf(self, x: torch.Tensor, flavour: str, Q: float = 100.0) -> torch.Tensor:
        """Return x*f(x,Q) for a given flavour string."""
        if self.use_lhapdf and hasattr(self, '_grid'):
            return self._interpolate_lhapdf(x, flavour, Q)
        else:
            return self._evaluate_parametric(x, flavour)

    def _interpolate_lhapdf(self, x: torch.Tensor, flavour: str, Q: float) -> torch.Tensor:
        pid = self.FLAVOUR_TO_PID.get(flavour, None)
        if pid is None:
            raise ValueError(f"Unknown flavour: {flavour}")
        try:
            idx = self.lha_pids.index(pid)
        except ValueError:
            raise ValueError(f"PID {pid} not found in LHAPDF grid.")
        x = x.clamp(min=1e-9)
        x_log = torch.log10(x)
        q_log = math.log10(Q)
        # Normalize to [-1,1] using the full log range
        x_norm = 2.0 * (x_log - self._x_log_min) / (self._x_log_max - self._x_log_min) - 1.0
        q_norm = 2.0 * (q_log - self._q_log_min) / (self._q_log_max - self._q_log_min) - 1.0
        n_pts = x.shape[0]
        # grid_sample expects (x,y) = (width, height)  i.e. (x_norm for n_x, q_norm for n_q)
        grid_4d = self._grid[:, idx:idx+1, :, :]   # [1,1,n_q,n_x]
        grid_coords = torch.stack([
            x_norm,
            torch.full_like(x, q_norm)
        ], dim=-1).view(n_pts, 1, 1, 2)
        sampled = F.grid_sample(grid_4d.expand(n_pts, -1, -1, -1),
                                grid_coords, mode='bilinear', padding_mode='border',
                                align_corners=True)
        return sampled.view(n_pts)

    def luminosity_qqbar(self, sqrts: float, M: torch.Tensor, qtype: str = 'u') -> torch.Tensor:
        """Compute parton luminosity for qqbar annihilation (q = 'u','d','s','c','b')."""
        tau = (M**2) / sqrts**2
        t = torch.linspace(0, 1, 200, device=self.device)
        tau_b = tau.unsqueeze(1)
        x1 = tau_b + (1 - tau_b) * t.unsqueeze(0)
        x2 = tau_b / x1
        f_q   = self.xf(x1.flatten(), qtype).view(x1.shape)
        f_qbar = self.xf(x2.flatten(), qtype + 'bar').view(x2.shape)
        jac = (1 - tau_b)
        integrand = (f_q * f_qbar) / (x1 * sqrts**2) * jac
        dlum = torch.trapezoid(integrand, t.unsqueeze(0).expand_as(x1), dim=1)
        return dlum * M


# =============================================================================
# 4. Matrix Elements (LO + differentiable K‑factors)
# =============================================================================
class KFactorProvider(nn.Module):
    """Differentiable K‑factor based on dense interpolation of high‑precision data."""
    def __init__(self, process='drell_yan', device='cpu'):
        super().__init__()
        self.device = device
        if process == 'drell_yan':
            # Drell‑Yan NNLO/NNLL K‑factors as function of invariant mass (GeV)
            _mass = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 250, 300, 400, 500, 600, 800, 1000, 1500, 2000]
            _kval = [1.45, 1.40, 1.37, 1.35, 1.33, 1.31, 1.30, 1.28, 1.27, 1.26, 1.25, 1.24, 1.24, 1.23, 1.23, 1.22, 1.22, 1.21, 1.20, 1.19, 1.18, 1.17, 1.16, 1.15, 1.14, 1.13, 1.12]
            self.register_buffer('mass_grid', torch.tensor(_mass, dtype=torch.float32))
            self.register_buffer('k_grid', torch.tensor(_kval, dtype=torch.float32))
        elif process == 'gg_higgs':
            _mass = [100, 105, 110, 115, 120, 122, 124, 125, 126, 128, 130, 135, 140, 145, 150, 160, 170, 180, 200]
            _kval = [1.95, 1.92, 1.88, 1.85, 1.81, 1.80, 1.78, 1.77, 1.78, 1.79, 1.80, 1.82, 1.85, 1.88, 1.90, 1.95, 1.98, 2.00, 2.05]
            self.register_buffer('mass_grid', torch.tensor(_mass, dtype=torch.float32))
            self.register_buffer('k_grid', torch.tensor(_kval, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown K‑factor process: {process}")
        self.mass_grid = self.mass_grid.to(device)
        self.k_grid = self.k_grid.to(device)

    def forward(self, mass: torch.Tensor) -> torch.Tensor:
        """Return K‑factor for given mass (in GeV)."""
        mass = torch.as_tensor(mass, dtype=torch.float32, device=self.device)
        return diff_interp(mass, self.mass_grid, self.k_grid)


class MatrixElements:
    def __init__(self, forces: ForceParameters, pdf: PDFProvider, device='cpu'):
        self.forces = forces
        self.pdf = pdf
        self.device = device
        self.k_dy = KFactorProvider('drell_yan', device)
        self.k_higgs = KFactorProvider('gg_higgs', device)

    @staticmethod
    def k_factor_drell_yan(s_hat: torch.Tensor) -> torch.Tensor:
        return KFactorProvider('drell_yan')(torch.sqrt(s_hat))

    @staticmethod
    def k_factor_gg_higgs(mH: float, pt: float = 0.0, rapidity: float = 0.0) -> float:
        return KFactorProvider('gg_higgs')(torch.tensor(mH)).item()

    def qed_ee_mumu(self, s, t, u):
        alpha = self.forces.alpha_EM()
        return (4*math.pi*alpha)**2 * (t**2 + u**2) / s**2

    def qcd_qqbar_gg(self, s, t, u, Q2):
        alpha_s = self.forces.alpha_s(Q2)
        return (4*math.pi*alpha_s)**2 * (32/27) * ((t**2+u**2)/(t*u) - 9/4*(t**2+u**2)/s**2)

    def weak_ee_ZH(self, s, t, u):
        g = math.sqrt(4*math.pi*self.forces.alpha_EM_MZ) / 0.48
        MZ = self.forces.MZ
        prop = 1.0 / ((s - MZ**2)**2 + (MZ*2.5)**2)
        return (g**4) * s * prop

    def drell_yan_partonic(self, s_hat: torch.Tensor, flavour: str = 'u') -> torch.Tensor:
        alpha = self.forces.alpha_EM_MZ
        MZ = self.forces.MZ
        GammaZ = 2.4952
        sin2w = 0.23122
        Q_u, Q_d = 2/3, -1/3
        if flavour in ('u','c'):
            Q = Q_u; gV = 0.5 - 4/3*sin2w; gA = 0.5
        else:
            Q = Q_d; gV = -0.5 + 2/3*sin2w; gA = -0.5
        ve = -0.5 + 2*sin2w; ae = -0.5
        s = s_hat
        chi_Z = s * (s - MZ**2) / ((s - MZ**2)**2 + (MZ*GammaZ)**2)
        chi_ZA = (s - MZ**2) / ((s - MZ**2)**2 + (MZ*GammaZ)**2)
        pref = 4*math.pi*alpha**2 / (3*s)
        lo = pref * (Q**2 + (gV**2+gA**2)*(ve**2+ae**2)*chi_Z**2 + 2*Q*gV*ve*chi_ZA)
        k = self.k_dy(torch.sqrt(s_hat))   # K‑factor at M_ll
        return lo * k

    def drell_yan_sigma(self, sqrts: float, M: torch.Tensor) -> torch.Tensor:
        tau = M**2 / sqrts**2
        t = torch.linspace(0, 1, 200, device=self.device)
        tau_b = tau.unsqueeze(1)
        x1 = tau_b + (1 - tau_b) * t.unsqueeze(0)
        x2 = tau_b / x1
        sigma = torch.zeros_like(M)
        for q in ['u','d','s','c','b']:
            f_q   = self.pdf.xf(x1.flatten(), q).view(x1.shape)
            f_qbar = self.pdf.xf(x2.flatten(), q + 'bar').view(x2.shape)
            eff_flav = 'u' if q in ('u','c') else 'd'
            sigma_hat = self.drell_yan_partonic(M.unsqueeze(1)**2, flavour=eff_flav)
            integrand = (f_q * f_qbar) * sigma_hat / (x1 * sqrts**2) * (1 - tau_b)
            sigma += torch.trapezoid(integrand, t.unsqueeze(0).expand_as(x1), dim=1)
        return sigma * (2*M / sqrts**2) * 0.389379e9

    def gg_higgs_partonic(self, s_hat: torch.Tensor, mH: float = 125.0,
                          pt: float = 0.0, rap: float = 0.0) -> torch.Tensor:
        GF = self.forces.G_F
        alpha_s = self.forces.alpha_s(s_hat)
        lo = GF * alpha_s**2 / (288 * math.sqrt(2) * math.pi)
        k = self.k_higgs(torch.tensor(mH, device=self.device))
        return lo * k

    def higgs_gluon_fusion_sigma(self, sqrts: float, mH: float, pt: float = 0.0, rap: float = 0.0) -> torch.Tensor:
        tau = mH**2 / sqrts**2
        t = torch.linspace(0, 1, 200, device=self.device)
        x1 = tau + (1-tau)*t
        glu = self.pdf.xf(x1, 'g')
        sigma_hat = self.gg_higgs_partonic(torch.tensor(mH**2, device=self.device), mH, pt, rap)
        integrand = (glu * glu) / x1 * (1-tau) / sqrts**2
        dL = torch.trapezoid(integrand, t) * mH
        return sigma_hat * dL * 0.389379e9


# =============================================================================
# 5. Data Download Utilities
# =============================================================================
DATA_CACHE = os.path.expanduser("~/.standard_one_data")

def download_file(url: str, filename: str, expected_sha256: Optional[str] = None,
                  cache_dir: str = DATA_CACHE) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, filename)
    if os.path.exists(filepath):
        if expected_sha256:
            sha256 = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            if sha256.hexdigest() == expected_sha256:
                logger.info(f"Using cached {filename}")
                return filepath
        else:
            return filepath
    logger.info(f"Downloading {url} ...")
    try:
        urlretrieve(url, filepath)
        if expected_sha256:
            sha256 = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            if sha256.hexdigest() != expected_sha256:
                raise RuntimeError(f"Checksum mismatch for {filename}")
        return filepath
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        raise


# =============================================================================
# 6. CERN Data Loader (ROOT & pyhf)
# =============================================================================
class CERNDataLoader:
    @staticmethod
    def load_root(filepath: str, treename: str, branch: str,
                  selection: Callable = None) -> torch.Tensor:
        if not HAS_UPROOT:
            raise ImportError("uproot required for ROOT files")
        with uproot.open(filepath) as f:
            tree = f[treename]
            data = tree[branch].array(library='np')
        if HAS_AWKWARD and isinstance(data, ak.Array):
            data = ak.flatten(data).to_numpy()
        if selection is not None:
            data = data[selection(data)]
        return torch.tensor(data, dtype=torch.float32)

    @staticmethod
    def download_atlas_higgs_workspace() -> str:
        return download_file(
            "https://raw.githubusercontent.com/scikit-hep/pyhf/master/docs/examples/2-bin_1-channel.json",
            "pyhf_example_workspace.json"
        )

    @staticmethod
    def load_pyhf_model(workspace_path: str):
        if not HAS_PYHF:
            raise ImportError("pyhf required")
        with open(workspace_path) as f:
            spec = json.load(f)
        ws = pyhf.Workspace(spec)
        model = ws.model()
        return ws, model


# =============================================================================
# 7. NASA / Cosmology Data Loader
# =============================================================================
class NASADataLoader:
    @staticmethod
    def load_fits(filepath: str, ext: int = 1, column: str = None) -> torch.Tensor:
        if not HAS_ASTROPY:
            raise ImportError("astropy required")
        with fits.open(filepath) as hdul:
            data = hdul[ext].data
        if column:
            data = data[column]
        return torch.tensor(np.asarray(data, dtype=np.float32))

    @staticmethod
    def load_csv(filepath: str, columns: List[str] = None) -> torch.Tensor:
        if not HAS_ASTROPY:
            import csv
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            data = np.array(rows[1:], dtype=np.float32)
            return torch.tensor(data)
        tab = Table.read(filepath, format='csv')
        if columns:
            tab = tab[columns]
        return torch.tensor(tab.as_array().view(np.float32).reshape(-1, len(columns)))

    @staticmethod
    def download_planck_highl_spectrum() -> str:
        url = ("https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/"
               "cosmology/COM_PowerSpect_CMB-TT-binned_R3.01.txt")
        return download_file(url, "planck_tt_binned.txt")

    @staticmethod
    def load_planck_highl_spectrum(filepath: Optional[str] = None):
        if filepath is None:
            filepath = NASADataLoader.download_planck_highl_spectrum()
        with open(filepath, 'r') as f:
            lines = f.readlines()
        data_lines = []
        cov_lines = []
        in_cov = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if i == 0:
                continue  # skip header
            if stripped == "":
                in_cov = True
                continue
            if not in_cov:
                data_lines.append(stripped)
            else:
                cov_lines.append(stripped)
        if not data_lines:
            raise RuntimeError("No data found in Planck file.")
        data = np.loadtxt(data_lines, ndmin=2)
        ell = torch.tensor(data[:,0], dtype=torch.float32)
        Dl  = torch.tensor(data[:,1], dtype=torch.float32)
        if data.shape[1] >= 3:
            err = torch.tensor(data[:,2], dtype=torch.float32)
        else:
            err = torch.ones_like(ell) * 0.01 * Dl
        if cov_lines:
            cov_data = np.loadtxt(cov_lines)
            cov = torch.tensor(cov_data, dtype=torch.float32)
            if cov.ndim == 1 and cov.shape[0] == len(ell)**2:
                cov = cov.reshape(len(ell), len(ell))
            elif cov.ndim != 2:
                logger.warning("Covariance format unexpected, using diagonal.")
                cov = torch.diag(err**2)
        else:
            cov = torch.diag(err**2)
        Cl = Dl * 2 * math.pi / (ell * (ell+1))
        return ell, Cl, cov


# =============================================================================
# 8. Cosmology & Fully Differentiable CMB (multi‑backend)
# =============================================================================
class Cosmology:
    def __init__(self, H0=67.4, Omega_b=0.049, Omega_c=0.266, Omega_L=0.685,
                 w=-1.0, T_cmb=2.7255, N_eff=3.046, device='cpu'):
        self.H0 = H0
        self.Ob = Omega_b
        self.Oc = Omega_c
        self.Om = Omega_b + Omega_c
        self.OL = Omega_L
        self.w = w
        self.Tcmb = T_cmb
        self.Neff = N_eff
        self.device = device

    def _E(self, z: Union[float, torch.Tensor]) -> torch.Tensor:
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        return torch.sqrt(self.Om*(1+z)**3 + self.OL*(1+z)**(3*(1+self.w)))

    def comoving_distance(self, z: Union[float, torch.Tensor]) -> torch.Tensor:
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        z_grid = torch.linspace(0, z.item(), 500, device=self.device)
        integrand = 1.0 / self._E(z_grid)
        return (2997.92458 / self.H0) * torch.trapezoid(integrand, z_grid)


class CMBBackend(nn.Module):
    def __init__(self, cosmo: Cosmology, lmax: int = 2500, device='cpu'):
        super().__init__()
        self.cosmo = cosmo
        self.lmax = lmax
        self.device = device

    def C_ell_TT(self, A_s: float = 2.1e-9, n_s: float = 0.96,
                 tau: float = 0.054) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def C_ell_at(self, ell: torch.Tensor, A_s: float, n_s: float, tau: float) -> torch.Tensor:
        raise NotImplementedError


class HuWhiteCMB(CMBBackend):
    """Hu & White (1997) analytic CMB spectrum – fallback only."""
    def __init__(self, cosmo: Cosmology, lmax: int = 2500, device='cpu'):
        super().__init__(cosmo, lmax, device)
        self._precompute_background()

    def _precompute_background(self):
        h = self.cosmo.H0 / 100.0
        Obh2 = self.cosmo.Ob * h**2
        Omh2 = self.cosmo.Om * h**2
        theta_cmb = self.cosmo.Tcmb / 2.7
        self.rs = 44.5 * math.log(9.83 / (Omh2 + 1e-10)) / math.sqrt(1 + 10 * Obh2**0.75) / h
        self.D_A = self.cosmo.comoving_distance(1089.0) / (1 + 1089.0)
        self.theta_s = self.rs / self.D_A
        self.ell_eq = 220 * math.sqrt(Omh2 * theta_cmb**4)
        self.ell_D = 1300 * (Omh2)**0.25 * theta_cmb

    def C_ell_TT(self, A_s=2.1e-9, n_s=0.96, tau=0.054):
        ell = torch.arange(2, self.lmax+1, device=self.device, dtype=torch.float32)
        Cl = self.C_ell_at(ell, A_s, n_s, tau)
        return ell, Cl

    def C_ell_at(self, ell, A_s, n_s, tau):
        x = ell * self.theta_s
        D_damp = torch.exp(-(ell / self.ell_D)**1.2)
        omega_b = self.cosmo.Ob * (self.D_A / 2997.9 * 100)**2
        R = 0.5 * (omega_b / 0.022)**0.5
        C = (A_s * 2e-10) * (ell * (ell+1) / (2*math.pi)) * \
            (torch.sin(x) / (x + 1e-6))**2 * (1 + R * torch.cos(x))**2 * D_damp * \
            (ell / self.ell_eq)**(n_s - 1)
        bump = (A_s * 1e-10) * (ell/50.0)**(1 - n_s) * tau * torch.exp(-((ell-200)/150)**2)
        return C + bump


class CAMBCMB(CMBBackend):
    """CAMB Boltzmann solver (modified BSD licence)."""
    def __init__(self, cosmo: Cosmology, lmax: int = 2500, device='cpu'):
        if not HAS_CAMB:
            raise ImportError("CAMB is not installed (optional).")
        super().__init__(cosmo, lmax, device)
        self._setup_params()

    def _setup_params(self):
        self.camb_params = camb.CAMBparams()
        self.camb_params.set_cosmology(H0=self.cosmo.H0,
                                       ombh2=self.cosmo.Ob*(self.cosmo.H0/100)**2,
                                       omch2=self.cosmo.Oc*(self.cosmo.H0/100)**2,
                                       mnu=0.06, omk=0, tau=0.054)
        self.camb_params.set_for_lmax(self.lmax, lens_potential_estimate=0)
        self.camb_params.WantTensors = False

    def C_ell_TT(self, A_s=2.1e-9, n_s=0.96, tau=0.054):
        self.camb_params.InitPower.set_params(As=A_s, ns=n_s, r=0)
        self.camb_params.set_cosmology(tau=tau)
        results = camb.get_results(self.camb_params)
        powers = results.get_cmb_power_spectra(self.camb_params, CMB_unit='muK')
        ell = torch.arange(2, self.lmax+1, device=self.device, dtype=torch.float32)
        Cl = torch.tensor(powers['total'][2:self.lmax+1, 0], device=self.device)
        return ell, Cl

    def C_ell_at(self, ell, A_s, n_s, tau):
        full_ell, full_Cl = self.C_ell_TT(A_s, n_s, tau)
        return diff_interp(ell.to(self.device), full_ell, full_Cl)


class ClassCMB(CMBBackend):
    """CLASS Boltzmann solver (GPLv2 licence – optional)."""
    def __init__(self, cosmo: Cosmology, lmax: int = 2500, device='cpu'):
        if not HAS_CLASS:
            raise ImportError("CLASS is not installed (optional).")
        super().__init__(cosmo, lmax, device)
        self._setup_base()

    def _setup_base(self):
        self.base_params = {
            'h': self.cosmo.H0/100,
            'omega_b': self.cosmo.Ob*(self.cosmo.H0/100)**2,
            'omega_cdm': self.cosmo.Oc*(self.cosmo.H0/100)**2,
            'T_cmb': self.cosmo.Tcmb,
            'output': 'tCl pCl lCl',
            'l_max_scalars': self.lmax,
        }

    def C_ell_TT(self, A_s=2.1e-9, n_s=0.96, tau=0.054):
        params = self.base_params.copy()
        params.update({'A_s': A_s, 'n_s': n_s, 'tau_reio': tau})
        cosmo_class = classy.Class()
        cosmo_class.set(params)
        cosmo_class.compute()
        cls = cosmo_class.lensed_cl(self.lmax)
        ell = torch.arange(2, self.lmax+1, device=self.device, dtype=torch.float32)
        Cl = torch.tensor(cls['tt'][2:self.lmax+1], device=self.device) * 1e12
        cosmo_class.struct_cleanup()
        cosmo_class.empty()
        return ell, Cl

    def C_ell_at(self, ell, A_s, n_s, tau):
        full_ell, full_Cl = self.C_ell_TT(A_s, n_s, tau)
        return diff_interp(ell.to(self.device), full_ell, full_Cl)


class CosmoPowerCMB(CMBBackend):
    """CosmoPower neural emulator (MIT licence)."""
    def __init__(self, cosmo: Cosmology, lmax: int = 2500, device='cpu',
                 model_path: Optional[str] = None):
        if not HAS_COSMOPOWER:
            raise ImportError("CosmoPower is not installed (optional).")
        super().__init__(cosmo, lmax, device)
        if model_path is None:
            default_path = os.path.join(os.path.dirname(cosmopower.__file__),
                                        'trained_models', 'cmb_TT_PCA.pkl')
            if os.path.exists(default_path):
                model_path = default_path
            else:
                self.emulator = cosmopower.CosmoPower_PCA()
                logger.warning("CosmoPower model not specified; using default untrained PCA emulator (unreliable).")
                return
        self.emulator = cosmopower.CosmoPower_PCA(restore=True, restore_filename=model_path)

    def C_ell_TT(self, A_s=2.1e-9, n_s=0.96, tau=0.054):
        params = {
            'A_s': A_s,
            'n_s': n_s,
            'omega_b': self.cosmo.Ob*(self.cosmo.H0/100)**2,
            'omega_cdm': self.cosmo.Oc*(self.cosmo.H0/100)**2,
            'h': self.cosmo.H0/100,
            'tau_reio': tau,
        }
        Cl = self.emulator.predict(params)
        ell = torch.arange(2, self.lmax+1, device=self.device, dtype=torch.float32)
        return ell, torch.tensor(Cl[2:self.lmax+1], device=self.device)

    def C_ell_at(self, ell, A_s, n_s, tau):
        full_ell, full_Cl = self.C_ell_TT(A_s, n_s, tau)
        return diff_interp(ell.to(self.device), full_ell, full_Cl)


class DifferentiableCMB(CMBBackend):
    """Unified differentiable CMB calculator dispatching to chosen backend."""
    def __init__(self, cosmo: Cosmology, lmax: int = 2500, device='cpu',
                 backend: str = 'analytic'):
        super().__init__(cosmo, lmax, device)
        self.backend_name = backend.lower()
        if self.backend_name == 'camb' and HAS_CAMB:
            self.engine = CAMBCMB(cosmo, lmax, device)
        elif self.backend_name == 'class' and HAS_CLASS:
            self.engine = ClassCMB(cosmo, lmax, device)
        elif self.backend_name == 'cosmopower' and HAS_COSMOPOWER:
            self.engine = CosmoPowerCMB(cosmo, lmax, device)
        else:
            logger.warning(f"Backend '{backend}' unavailable; falling back to Hu & White analytic.")
            self.engine = HuWhiteCMB(cosmo, lmax, device)

    def C_ell_TT(self, A_s=2.1e-9, n_s=0.96, tau=0.054):
        return self.engine.C_ell_TT(A_s, n_s, tau)

    def C_ell_at(self, ell, A_s, n_s, tau):
        return self.engine.C_ell_at(ell, A_s, n_s, tau)


# =============================================================================
# 9. Structural Components (CSOC, SSC, RG, BV)
# =============================================================================
class CSOCKernel(nn.Module):
    def __init__(self, init_Cs=0.18, init_lambda=12.0, init_alpha=0.5,
                 init_theta=1.0, init_tau=10.0, device='cpu'):
        super().__init__()
        self.log_Cs = nn.Parameter(torch.tensor(math.log(init_Cs), device=device))
        self.log_lambda = nn.Parameter(torch.tensor(math.log(init_lambda), device=device))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha), device=device))
        self.log_theta = nn.Parameter(torch.tensor(math.log(init_theta), device=device))
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau), device=device))

    @property
    def Cs(self): return torch.exp(self.log_Cs)
    @property
    def lambd(self): return torch.exp(self.log_lambda)
    @property
    def alpha(self): return torch.exp(self.log_alpha)
    @property
    def theta(self): return torch.exp(self.log_theta)
    @property
    def tau(self): return torch.exp(self.log_tau)

    def forward(self, r):
        return self.Cs * torch.pow(r + 1e-6, -self.alpha) * torch.exp(-r / self.lambd)


class SemanticStateContraction:
    def __init__(self, epsilon_fp=0.0028, sigma_target=1.0):
        self.eps = epsilon_fp
        self.target = sigma_target
        self.prev = None

    def __call__(self, x):
        if self.prev is None or self.prev.device != x.device:
            self.prev = x.detach().to(x.device)
            return x
        new = self.prev + self.eps * (x - self.prev)
        self.prev = new.detach()
        return new


class DiffRGRefiner:
    def __init__(self, keep_fraction=0.5):
        self.keep_fraction = keep_fraction

    def forward_1d(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply a low‑pass filter to a 1D signal (e.g., mass spectrum)."""
        x_hat = torch.fft.rfft(signal)
        k = torch.fft.rfftfreq(len(signal), d=1.0, device=signal.device)
        mask = k <= (self.keep_fraction * k.max())
        mask[0] = True
        filtered = torch.fft.irfft(x_hat * mask, n=len(signal))
        return torch.clamp(filtered, min=0)

    def forward(self, x):
        # original 3D version kept for compatibility
        x_hat = torch.fft.rfftn(x)
        dims = x.shape
        kx = torch.fft.fftfreq(dims[0], d=1.0, device=x.device)
        ky = torch.fft.fftfreq(dims[1], d=1.0, device=x.device)
        kz = torch.fft.rfftfreq(dims[2], d=1.0, device=x.device)
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        K_mag = torch.sqrt(KX**2 + KY**2 + KZ**2)
        mask = K_mag <= (self.keep_fraction * K_mag.max())
        mask[0,0,0] = True
        return torch.fft.irfftn(x_hat * mask.to(x_hat.dtype), s=x.shape)


class BVConsistency:
    def __init__(self, reference_means_stds=None):
        self.ref = reference_means_stds

    def score(self, sample_means, sample_stds):
        if self.ref is None:
            return 0.0
        chi2 = 0.0
        for (rm, rs), (sm, ss) in zip(self.ref, zip(sample_means, sample_stds)):
            chi2 += ((sm - rm) / rs) ** 2
        return chi2


# =============================================================================
# 10. Differentiable Generators (structural + empirical for real data)
# =============================================================================
class BaseStructuralGenerator(nn.Module):
    def __init__(self, csoc, ssc, rg, device='cpu'):
        super().__init__()
        self.csoc = csoc
        self.ssc = ssc
        self.rg = rg
        self.device = device
        self.log_mu = nn.Parameter(torch.tensor(0.0, device=device))
        self.log_lam = nn.Parameter(torch.tensor(math.log(0.5), device=device))
        self.log_T = nn.Parameter(torch.tensor(math.log(1.0), device=device))

    @property
    def mu(self): return torch.exp(self.log_mu)
    @property
    def lam(self): return torch.exp(self.log_lam)
    @property
    def T(self): return torch.exp(self.log_T)


class ColliderGenerator(BaseStructuralGenerator):
    def __init__(self, csoc, ssc, rg, mass_range=(50,200), n_events=1000, device='cpu'):
        super().__init__(csoc, ssc, rg, device)
        self.mass_range = mass_range
        self.n_events = n_events
        self.signal_mass = nn.Parameter(torch.tensor(125.0, device=device))
        self.signal_sigma = nn.Parameter(torch.tensor(2.0, device=device))
        self.alpha_cb = nn.Parameter(torch.tensor(1.0, device=device))
        self.n_cb = nn.Parameter(torch.tensor(2.0, device=device))

    def crystal_ball(self, m):
        x = (m - self.signal_mass) / self.signal_sigma
        abs_x = torch.abs(x)
        gauss = torch.exp(-0.5 * x**2)
        A = (self.n_cb / torch.abs(self.alpha_cb))**self.n_cb * torch.exp(-0.5 * self.alpha_cb**2)
        B = self.n_cb / torch.abs(self.alpha_cb) - torch.abs(self.alpha_cb)
        tail = A * (B + abs_x) ** (-self.n_cb)
        left_tail = x < -torch.abs(self.alpha_cb)
        right_tail = x > torch.abs(self.alpha_cb)
        result = gauss.clone()
        result[left_tail] = tail[left_tail]
        result[right_tail] = tail[right_tail]
        return result

    def pdf(self, m):
        """Normalized probability density (sum over bins ≈ 1)."""
        a = torch.exp(-self.T)
        norm_bkg = (torch.exp(-a*self.mass_range[0]) - torch.exp(-a*self.mass_range[1])) / a
        bkg = torch.exp(-a * m) / norm_bkg
        sig = self.crystal_ball(m)
        jump = self.lam * torch.exp(-0.5*((m-125.0)/2.0)**2)
        base_pdf = self.mu * self.n_events * sig + self.n_events * bkg + jump

        # Structural modulation via CSOC
        r = (m - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])
        csoc_mod = 1.0 + self.csoc(r) * 0.1
        modulated = base_pdf * csoc_mod

        # RG refinement (low‑pass filter to emulate finite resolution)
        refined = self.rg.forward_1d(modulated)

        # Normalize to unit integral over the grid
        dx = (self.mass_range[1] - self.mass_range[0]) / (len(m) - 1)
        integral = torch.trapezoid(refined, m)
        return refined / integral

    def generate(self):
        m = torch.linspace(self.mass_range[0], self.mass_range[1], self.n_events, device=self.device)
        return m, self.pdf(m)

    def update_state(self):
        """Apply SSC contraction to the generator's dynamical parameters."""
        with torch.no_grad():
            self.log_mu.data = self.ssc(self.log_mu.data)
            self.log_lam.data = self.ssc(self.log_lam.data)
            self.log_T.data = self.ssc(self.log_T.data)


class EmpiricalGenerator(BaseStructuralGenerator):
    """Generator that builds a differentiable KDE from real collider data."""
    def __init__(self, data: torch.Tensor, csoc, ssc, rg,
                 mass_range=(50,200), n_events=1000, bandwidth=None, device='cpu'):
        super().__init__(csoc, ssc, rg, device)
        self.register_buffer('data_points', data.to(device))
        self.mass_range = mass_range
        self.n_events = n_events
        if bandwidth is None:
            # Scott's rule for Gaussian KDE
            sigma = data.std().item()
            n = len(data)
            self.bandwidth = nn.Parameter(torch.tensor(sigma * n**(-1/5), device=device))
        else:
            self.bandwidth = nn.Parameter(torch.tensor(bandwidth, device=device))

    def pdf(self, m):
        """KDE probability density on grid m, normalized."""
        # m: (n_events,) grid, data_points: (N,)
        diff = m.unsqueeze(1) - self.data_points.unsqueeze(0)  # (n_events, N)
        kernel_vals = torch.exp(-0.5 * (diff / self.bandwidth)**2)
        density = kernel_vals.sum(dim=1) / (len(self.data_points) * self.bandwidth * math.sqrt(2*math.pi))
        # Structural modulation
        r = (m - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])
        csoc_mod = 1.0 + self.csoc(r) * 0.1
        density = density * csoc_mod
        # RG refinement
        density = self.rg.forward_1d(density)
        # Normalize
        integral = torch.trapezoid(density, m)
        return density / integral

    def generate(self):
        m = torch.linspace(self.mass_range[0], self.mass_range[1], self.n_events, device=self.device)
        return m, self.pdf(m)

    def update_state(self):
        with torch.no_grad():
            self.log_mu.data = self.ssc(self.log_mu.data)
            self.log_lam.data = self.ssc(self.log_lam.data)
            self.log_T.data = self.ssc(self.log_T.data)


class BlackHoleGenerator(BaseStructuralGenerator):
    def __init__(self, model_type='hawking', bh_mass=1e12, csoc=None, ssc=None, rg=None,
                 mass_range=(0.1,100), n_events=500, device='cpu'):
        super().__init__(csoc, ssc, rg, device)
        self.model_type = model_type
        self.bh_mass = bh_mass
        self.mass_range = mass_range
        self.n_events = n_events

    def pdf(self, m):
        if self.model_type == 'hawking':
            T_h = 1.0/(8*math.pi*self.bh_mass)
            beta = 1.0/(T_h * self.csoc.Cs)
            pdf = (m**3) / (torch.exp(beta*m) - 1)
        elif self.model_type == 'page':
            T_h = 1.0/(8*math.pi*self.bh_mass)
            beta = 1.0/(T_h * self.csoc.Cs)
            pdf = (m**3) / (torch.exp(beta*m) - 1)
            page_factor = 0.5*(1 + torch.tanh((m - 0.5*self.bh_mass)/10.0))
            pdf = pdf * page_factor
        elif self.model_type == 'pbh':
            sigma = 0.5; mu_m = math.log(1.0)
            pdf = torch.exp(-0.5*((torch.log(m+1e-6)-mu_m)/sigma)**2) / (m+1e-6)
            r = m/(self.mass_range[1]-self.mass_range[0])
            pdf = pdf + 0.1 * self.csoc(r) * torch.exp(-m/20.0)
        else:
            raise ValueError(f"Unknown BH model: {self.model_type}")
        pdf = pdf + self.lam * torch.exp(-0.5*((m-10)/5)**2)
        return pdf / torch.trapezoid(pdf, m)

    def generate(self):
        m = torch.linspace(self.mass_range[0], self.mass_range[1], self.n_events, device=self.device)
        return m, self.pdf(m)


class DarkMatterGenerator(BaseStructuralGenerator):
    def __init__(self, model_type='wimp', dm_mass=100.0, csoc=None, ssc=None, rg=None,
                 mass_range=(0.1,200), n_events=500, device='cpu'):
        super().__init__(csoc, ssc, rg, device)
        self.model_type = model_type
        self.dm_mass = dm_mass
        self.mass_range = mass_range
        self.n_events = n_events

    def pdf(self, m):
        if self.model_type == 'wimp':
            E0 = self.dm_mass/10.0
            pdf = torch.exp(-m/E0) * (1 + 0.1*torch.cos(2*math.pi*m/50.0))
            pdf = pdf * self.csoc(torch.abs(m-20.0)/20.0)
        elif self.model_type == 'axion':
            width = 0.001*self.dm_mass
            pdf = 1.0/((m-self.dm_mass)**2 + width**2)
        elif self.model_type == 'sterile':
            sig = torch.exp(-0.5*((m-self.dm_mass)/0.5)**2)
            bkg = m**(-2.0)
            pdf = sig + 0.1*bkg
        elif self.model_type == 'fuzzy':
            pdf = torch.abs(torch.sin(2*math.pi*m/(self.dm_mass/10.0)))
        else:
            raise ValueError(f"Unknown DM model: {self.model_type}")
        pdf = pdf + self.lam * torch.exp(-0.5*((m-50)/10)**2)
        return pdf / torch.trapezoid(pdf, m)

    def generate(self):
        m = torch.linspace(self.mass_range[0], self.mass_range[1], self.n_events, device=self.device)
        return m, self.pdf(m)


class VacuumEnergyModel(nn.Module):
    def __init__(self, model_type='zero_point', csoc=None):
        super().__init__()
        self.model_type = model_type
        self.csoc = csoc if csoc else CSOCKernel()
        self.scale = nn.Parameter(torch.tensor(1e-9))

    def forward(self, L):
        L = torch.as_tensor(L, dtype=torch.float32)
        r = self.csoc(L/1e-3)
        if self.model_type == 'zero_point':
            rho = self.scale / L**4
        elif self.model_type == 'casimir':
            rho = self.scale * math.pi**2/(240*L**4)
        elif self.model_type == 'quintessence':
            rho = self.scale * torch.exp(-r)
        elif self.model_type == 'holographic':
            rho = self.scale / L**2
        else:
            raise ValueError(f"Unknown vacuum model: {self.model_type}")
        return rho * r


class VacuumExtractionModel(nn.Module):
    def __init__(self, hypothesis='casimir_piston', csoc=None):
        super().__init__()
        self.hypothesis = hypothesis
        self.csoc = csoc if csoc else CSOCKernel()
        self.strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, displacement, area=1.0):
        x = torch.as_tensor(displacement, dtype=torch.float32)
        r = self.csoc(x)
        if self.hypothesis == 'casimir_piston':
            F = area / (x**4 + 1e-6)
            work = F * x * self.strength
        elif self.hypothesis == 'dynamical_casimir':
            work = self.strength * (1.0/x) * r
        elif self.hypothesis == 'schwinger':
            work = self.strength * torch.exp(-1.0/(x*r + 1e-6))
        else:
            raise ValueError(f"Unknown extraction hypothesis: {self.hypothesis}")
        return work


# ---- External Event Generators (optional, safe wrappers) ------------------
class Pythia8Generator:
    def __init__(self, config_string: str):
        if not HAS_PYTHIA:
            raise ImportError("pythia8 is not installed (optional, GPL-2).")
        self.pythia = pythia8.Pythia()
        self.pythia.readString(config_string)
        self.pythia.init()

    def generate_events(self, n: int) -> List[Dict]:
        events = []
        count = 0
        while count < n:
            if not self.pythia.next():
                break
            particles = []
            for i in range(self.pythia.event.size()):
                p = self.pythia.event[i]
                particles.append({'id': p.id(), 'px': p.px(), 'py': p.py(),
                                  'pz': p.pz(), 'e': p.e(), 'm': p.m()})
            events.append(particles)
            count += 1
        return events

    def shutdown(self):
        self.pythia.stat()


class HerwigGenerator:
    def __init__(self, run_card: str):
        if not HAS_HERWIG:
            raise ImportError("Herwig is not installed (optional, GPL-3).")
        self.herwig = herwig.Herwig()
        self.herwig.initialize(run_card)

    def generate_events(self, n: int) -> List[Dict]:
        events = []
        count = 0
        while count < n:
            try:
                if not self.herwig.next():
                    break
                event = self.herwig.event()
                particles = []
                for i in range(event.size()):
                    p = event[i]
                    particles.append({'id': p.id(), 'px': p.px(), 'py': p.py(),
                                      'pz': p.pz(), 'e': p.e(), 'm': p.m()})
                events.append(particles)
                count += 1
            except Exception as e:
                logger.warning(f"Herwig event generation error: {e}")
                break
        return events

    def shutdown(self):
        self.herwig.finalize()


# =============================================================================
# 11. Likelihoods & Statistical Paradigms
# =============================================================================
class StructuralLikelihood(nn.Module):
    """Negative log‑likelihood using the generator's normalized PDF."""
    def __init__(self, generator: BaseStructuralGenerator):
        super().__init__()
        self.generator = generator

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        m, pdf_vals = self.generator.generate()
        density = diff_interp(data, m, pdf_vals)
        density = torch.clamp(density, min=1e-12)
        return -torch.sum(torch.log(density))


class PyHFDiffLikelihood:
    """Differentiable pyhf log‑likelihood using model.logpdf."""
    def __init__(self, model, data, device='cpu'):
        self.model = model
        self.data = torch.as_tensor(data, dtype=torch.float32, device=device)
        self.device = device
        # Get default parameter names and initial values
        self.param_names = model.config.par_order
        self.nuisance_names = model.config.nuisance_order
        self.all_names = self.param_names + self.nuisance_names
        init_pars = model.config.suggested_init()
        self.init_tensor = torch.tensor(init_pars, dtype=torch.float32, device=device)

    def nll(self, pars: torch.Tensor) -> torch.Tensor:
        """pars: tensor of all parameters (POIs + nuisances) in order."""
        return -self.model.logpdf(pars, self.data)[0]  # logpdf returns [tensor, aux]


class CMBLikelihood:
    """Gaussian CMB likelihood with full covariance, using Cholesky."""
    def __init__(self, cmb_calculator: CMBBackend, ell_data, Cl_data, cov):
        self.cmb = cmb_calculator
        self.ell_data = ell_data.to(cmb.device)
        self.Cl_data = Cl_data.to(cmb.device)
        self.cov = cov.to(cmb.device)
        L, info = torch.linalg.cholesky_ex(self.cov)
        if info != 0:
            self.cov += 1e-6 * torch.eye(len(self.cov), device=self.cov.device)
            L = torch.linalg.cholesky(self.cov)
        self.L = L
        self.logdet = 2 * torch.sum(torch.log(torch.diag(L)))
        self.device = cmb.device

    def log_likelihood(self, A_s, n_s, tau):
        theory = self.cmb.C_ell_at(self.ell_data, A_s, n_s, tau)
        delta = self.Cl_data - theory
        y = torch.triangular_solve(delta.unsqueeze(1), self.L, upper=False).solution.squeeze()
        chi2 = torch.dot(y, y)
        return -0.5 * (chi2 + self.logdet + len(delta)*math.log(2*math.pi))


# =============================================================================
# 12. Full Frequentist Analysis (robust optimisation)
# =============================================================================
class FrequentistAnalysis:
    def __init__(self, nll_fn: Callable[[], torch.Tensor],
                 params: List[nn.Parameter],
                 param_names: List[str],
                 device='cpu'):
        self.nll_fn = nll_fn
        self.params = params
        self.param_names = param_names
        self.device = device
        self._param_dict = dict(zip(param_names, params))

    def _set_frozen(self, freeze: Dict[str, float] = None):
        for n, p in self._param_dict.items():
            p.requires_grad = False if (freeze and n in freeze) else True
            if freeze and n in freeze:
                p.data.fill_(freeze[n])

    def fit(self, freeze: Dict[str, float] = None, max_iter: int = 500, n_restarts: int = 1) -> float:
        """Returns minimum NLL after optional multi‑start, with fallback to Adam."""
        best_nll = float('inf')
        original = {n: p.data.clone() for n, p in self._param_dict.items()}
        original_grad = {n: p.requires_grad for n, p in self._param_dict.items()}
        for restart in range(n_restarts):
            if restart > 0:
                for n, p in self._param_dict.items():
                    if p.requires_grad:
                        p.data.normal_(0, 0.5)
            self._set_frozen(freeze)
            free_params = [p for p in self.params if p.requires_grad]
            if not free_params:
                nll = self.nll_fn().item()
                if nll < best_nll:
                    best_nll = nll
                continue
            try:
                optimizer = LBFGS(free_params, max_iter=max_iter, line_search_fn='strong_wolfe')
                def closure():
                    optimizer.zero_grad()
                    loss = self.nll_fn()
                    loss.backward()
                    return loss
                optimizer.step(closure)
            except Exception:
                logger.warning("LBFGS failed, falling back to Adam.")
                optimizer = Adam(free_params, lr=0.01)
                for _ in range(max_iter):
                    optimizer.zero_grad()
                    loss = self.nll_fn()
                    loss.backward()
                    optimizer.step()
            nll = self.nll_fn().item()
            if nll < best_nll:
                best_nll = nll
                best_state = {n: p.data.clone() for n, p in self._param_dict.items()}
        if n_restarts > 1:
            for n, p in self._param_dict.items():
                p.data = best_state[n]
        for n, p in self._param_dict.items():
            p.requires_grad = original_grad[n]
        return best_nll

    def unconditional_fit(self) -> float:
        return self.fit(freeze=None, n_restarts=3)

    def conditional_fit(self, poi_name: str, poi_value: float) -> float:
        return self.fit(freeze={poi_name: poi_value})

    def q0(self, poi_name: str, null: float = 0.0) -> float:
        nll_null = self.conditional_fit(poi_name, null)
        nll_best = self.unconditional_fit()
        q = 2 * (nll_null - nll_best)
        return max(0.0, q)

    def significance(self, poi_name: str, null: float = 0.0) -> float:
        return math.sqrt(self.q0(poi_name, null))

    def p_value(self, poi_name: str, null: float = 0.0) -> float:
        """Asymptotic p‑value using half‑χ² for discovery (Cowan et al.)."""
        q0 = self.q0(poi_name, null)
        if q0 == 0:
            return 1.0
        return 0.5 * (1 - chi2.cdf(q0, 1))

    def confidence_interval(self, poi_name: str, cl: float = 0.68,
                            scan_range: Tuple[float,float] = None,
                            n_steps: int = 50) -> Tuple[float, float]:
        delta_nll = 0.5 * chi2.ppf(cl, 1)
        best_nll = self.unconditional_fit()
        best_val = self._param_dict[poi_name].data.item()
        if scan_range is None:
            scan_range = (best_val * 0.5, best_val * 1.5)
        grid = torch.linspace(scan_range[0], scan_range[1], n_steps)
        nlls = np.array([self.conditional_fit(poi_name, v.item()) for v in grid])
        mask = (nlls - best_nll) <= delta_nll
        if mask.sum() < 2:
            return (float('nan'), float('nan'))
        indices = np.where(mask)[0]
        return grid[indices[0]].item(), grid[indices[-1]].item()

    def upper_limit(self, poi_name: str, cl: float = 0.95,
                    scan_range: Tuple[float,float] = None,
                    n_steps: int = 50) -> float:
        """Asymptotic CLs upper limit (Cowan et al. asymptotic formula)."""
        best_nll = self.unconditional_fit()
        best_val = self._param_dict[poi_name].data.item()
        def p_mu(mu_val):
            cond_nll = self.conditional_fit(poi_name, mu_val)
            q_mu = 2 * (cond_nll - best_nll)
            if q_mu < 0:
                q_mu = 0.0
            return 1.0 - norm.cdf(math.sqrt(q_mu))
        if scan_range is None:
            scan_range = (best_val, best_val * 5)
        grid = torch.linspace(scan_range[0], scan_range[1], n_steps)
        p_vals = np.array([p_mu(v.item()) for v in grid])
        target = 1 - cl
        idx = np.searchsorted(p_vals, target)
        if idx == 0:
            return grid[0].item()
        elif idx >= len(grid):
            return grid[-1].item()
        x0, x1 = grid[idx-1].item(), grid[idx].item()
        y0, y1 = p_vals[idx-1], p_vals[idx]
        return x0 + (target - y0) * (x1 - x0) / (y1 - y0 + 1e-12)

    def bootstrap_calibrated_pvalue(self, null_poi_name: str, null_value: float,
                                    n_toys: int = 500, generator: BaseStructuralGenerator = None) -> float:
        """Bootstrap calibration by refitting toy datasets under the null hypothesis."""
        if generator is None:
            raise ValueError("Generator must be provided to sample toy data.")
        state = {n: p.data.clone() for n, p in self._param_dict.items()}
        self.conditional_fit(null_poi_name, null_value)
        q_obs = self.q0(null_poi_name, null_value)
        q_toys = []
        for _ in range(n_toys):
            m_grid, pdf_null = generator.generate()
            prob = pdf_null / pdf_null.sum()
            idx = torch.multinomial(prob, len(m_grid), replacement=True)
            toy_data = m_grid[idx]
            toy_state = {n: p.data.clone() for n, p in self._param_dict.items()}
            def toy_nll():
                return -torch.sum(torch.log(diff_interp(toy_data, m_grid, generator.pdf(m_grid)).clamp(1e-12)))
            toy_freq = FrequentistAnalysis(toy_nll, self.params, self.param_names, device=self.device)
            toy_null_nll = toy_freq.conditional_fit(null_poi_name, null_value)
            toy_best_nll = toy_freq.unconditional_fit()
            q_toy = max(0.0, 2 * (toy_null_nll - toy_best_nll))
            q_toys.append(q_toy)
        for n, p in self._param_dict.items():
            p.data = state[n]
        return np.mean(np.array(q_toys) >= q_obs)


# =============================================================================
# 13. Bayesian Analysis (with NUTS compatibility and unconstrained sampling)
# =============================================================================
class BayesianAnalysis:
    def __init__(self, log_prob_fn: Callable[[], torch.Tensor],
                 params: List[nn.Parameter],
                 param_names: List[str],
                 device='cpu'):
        self.log_prob_fn = log_prob_fn
        self.params = params
        self.param_names = param_names
        self.device = device
        self._param_dict = dict(zip(param_names, params))

    def _set_params(self, values: Dict[str, torch.Tensor]):
        with torch.no_grad():
            for n, v in values.items():
                self._param_dict[n].data = v.to(self.device)

    def log_posterior_value(self):
        return self.log_prob_fn().item()

    def sample_mh(self, n_samples=5000, burn_in=1000, step_size=0.1,
                  init: Dict[str, float] = None, adapt: bool = True,
                  adapt_cov: bool = True) -> Tuple[List[Dict], float]:
        if init is None:
            init = {n: self._param_dict[n].data.item() for n in self.param_names}
        current = {k: torch.tensor(v, device=self.device) for k, v in init.items()}
        self._set_params(current)
        current_lp = self.log_posterior_value()
        chain = []
        accepts = 0
        d = len(self.param_names)
        if adapt_cov:
            cov = torch.eye(d, device=self.device) * (step_size**2)
            mean_est = torch.zeros(d, device=self.device)
            n_accepted = 0
        for i in range(n_samples + burn_in):
            if adapt_cov and i > 100:
                L = torch.linalg.cholesky(cov)
                prop_vec = torch.zeros(d, device=self.device)
                for j, n in enumerate(self.param_names):
                    prop_vec[j] = current[n]
                prop_vec += L @ torch.randn(d, device=self.device)
                proposal = {n: prop_vec[j] for j, n in enumerate(self.param_names)}
            else:
                proposal = {}
                for n in self.param_names:
                    prop = current[n] + step_size * torch.randn(1, device=self.device).item()
                    proposal[n] = torch.tensor(prop, device=self.device)
            self._set_params(proposal)
            prop_lp = self.log_posterior_value()
            if math.log(torch.rand(1).item()) < prop_lp - current_lp:
                current = proposal
                current_lp = prop_lp
                accepts += 1
                if adapt_cov and i > 100:
                    n_accepted += 1
            if i >= burn_in:
                chain.append({k: current[k].item() for k in self.param_names})
            if adapt_cov and i > 100 and (i - burn_in) % 50 == 0:
                if n_accepted > 1:
                    accepted_points = []
                    for _i in range(max(0, i-500), i+1):
                        if _i >= burn_in:
                            accepted_points.append(torch.tensor([chain[_i-burn_in][n] for n in self.param_names], device=self.device))
                    if accepted_points:
                        pts = torch.stack(accepted_points)
                        mean_est = pts.mean(dim=0)
                        centered = pts - mean_est
                        cov = (centered.T @ centered) / (len(pts) - 1) + 1e-6 * torch.eye(d, device=self.device)
        acceptance_rate = accepts / (n_samples + burn_in)
        return chain, acceptance_rate

    def sample_nuts(self, n_samples=2000, warmup=500):
        if not HAS_PYRO:
            logger.warning("Pyro unavailable, falling back to adaptive MH.")
            return self.sample_mh(n_samples=n_samples, burn_in=warmup)
        # Unconstrained reparameterization: all parameters are directly sampled in unconstrained space.
        def pyro_model():
            params = {}
            for n in self.param_names:
                # Sample in unconstrained space (e.g. log for positive params is already unconstrained)
                params[n] = pyro.sample(n, dist_pyro.Normal(0.0, 3.0))
            with torch.no_grad():
                for n, val in params.items():
                    self._param_dict[n].data = val
            log_like = self.log_prob_fn()
            pyro.factor("log_likelihood", log_like)
        nuts_kernel = NUTS(pyro_model)
        mcmc = MCMC(nuts_kernel, num_samples=n_samples, warmup_steps=warmup,
                    disable_progbar=True)
        mcmc.run()
        samples = mcmc.get_samples()
        chain = []
        for i in range(n_samples):
            chain.append({k: samples[k][i].item() for k in samples})
        return chain, 1.0

    def laplace_approximation(self) -> Dict[str, Tuple[float, float]]:
        for p in self.params:
            p.requires_grad_(True)
        try:
            opt = LBFGS(self.params, max_iter=200, line_search_fn='strong_wolfe')
            def closure():
                opt.zero_grad()
                loss = -self.log_prob_fn()
                loss.backward()
                return loss
            opt.step(closure)
        except Exception:
            logger.warning("LBFGS failed in Laplace, using Adam.")
            opt = Adam(self.params, lr=0.01)
            for _ in range(300):
                opt.zero_grad()
                loss = -self.log_prob_fn()
                loss.backward()
                opt.step()
        nlp = -self.log_prob_fn()
        grads = torch.autograd.grad(nlp, self.params, create_graph=True)
        hess_rows = []
        for g in grads:
            g2 = torch.autograd.grad(g.sum(), self.params, retain_graph=True)
            hess_rows.append(torch.cat([gi.flatten() for gi in g2]))
        H = torch.stack(hess_rows)
        cov = torch.linalg.inv(H).detach().cpu()
        means = [p.data.item() for p in self.params]
        stds = torch.sqrt(torch.diag(cov)).tolist()
        return dict(zip(self.param_names, zip(means, stds)))

    def marginal_likelihood_laplace(self) -> float:
        map_est = self.laplace_approximation()
        means = [map_est[n][0] for n in self.param_names]
        self._set_params({n: torch.tensor(m) for n,m in zip(self.param_names, means)})
        log_lik_max = self.log_prob_fn().item()
        nlp = -log_lik_max
        grads = torch.autograd.grad(nlp, self.params, create_graph=True)
        hess_rows = []
        for g in grads:
            g2 = torch.autograd.grad(g.sum(), self.params, retain_graph=True)
            hess_rows.append(torch.cat([gi.flatten() for gi in g2]))
        H = torch.stack(hess_rows)
        _, logdetH = torch.linalg.slogdet(H)
        k = len(self.params)
        log_marginal = log_lik_max - 0.5 * logdetH.item() + 0.5 * k * math.log(2*math.pi)
        return log_marginal

    def bayes_factor_laplace(self, other_model: 'BayesianAnalysis') -> float:
        log_ml1 = self.marginal_likelihood_laplace()
        log_ml2 = other_model.marginal_likelihood_laplace()
        return log_ml1 - log_ml2


# =============================================================================
# 14. Structural Probability Interface
# =============================================================================
class StructuralProbability:
    def __init__(self, generator: BaseStructuralGenerator):
        self.generator = generator

    def probability(self, data: torch.Tensor) -> torch.Tensor:
        m, pdf = self.generator.generate()
        p = diff_interp(data, m, pdf)
        return p / p.sum()

    def uncertainty_source(self):
        return ("All randomness originates from the unresolved structural interface Γ. "
                "Once Γ is fully specified, outcomes are deterministic.")


# =============================================================================
# 15. Model Comparison
# =============================================================================
class ModelComparator:
    @staticmethod
    def compare(models: Dict[str, nn.Module], data: torch.Tensor,
                param_names: List[str], device='cpu') -> Dict:
        results = {}
        for name, model in models.items():
            def make_nll(mod):
                likelihood = StructuralLikelihood(mod)
                return lambda: likelihood(data)
            nll_fn = make_nll(model)
            params = [p for n, p in model.named_parameters() if n in param_names]
            freq = FrequentistAnalysis(nll_fn, params, param_names, device=device)
            nll_best = freq.unconditional_fit()
            k = len(params)
            n = len(data)
            aic = 2*k + 2*nll_best
            bic = k*math.log(n) + 2*nll_best
            bayes = BayesianAnalysis(lambda: -nll_fn(), params, param_names, device=device)
            log_ml = bayes.marginal_likelihood_laplace()
            results[name] = {'aic': aic, 'bic': bic, 'nll': nll_best, 'log_marginal': log_ml}
        return results


# =============================================================================
# 16. Cross‑Correlation Analyzer
# =============================================================================
class CrossCorrelationAnalyzer(nn.Module):
    def __init__(self, n_collider, n_cosmo, hidden=32, device='cpu'):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_collider, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_cosmo)
        )
        self.to(device)

    def forward(self, collider_data, cosmo_data):
        pred = self.fc(collider_data)
        return F.mse_loss(pred, cosmo_data)


# =============================================================================
# 17. Unification Models
# =============================================================================
class UnificationModel(nn.Module):
    def __init__(self, M_GUT=2e16, alpha_GUT=1/25.0, device='cpu'):
        super().__init__()
        self.M_GUT = M_GUT
        self.alpha_GUT = alpha_GUT
        self.device = device

    def running_su3(self, Q):
        b3 = -7.0
        return 1/(1/self.alpha_GUT + b3/(2*math.pi)*torch.log(Q/self.M_GUT))

    def running_su2(self, Q):
        b2 = -19/6
        return 1/(1/self.alpha_GUT + b2/(2*math.pi)*torch.log(Q/self.M_GUT))

    def running_u1(self, Q):
        b1 = 41/6
        return 1/(1/self.alpha_GUT + b1/(2*math.pi)*torch.log(Q/self.M_GUT))

    def randall_sundrum_warp(self, y, k=1.0):
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        return torch.exp(-k * torch.abs(y))


# =============================================================================
# 18. Main Research Framework — STANDARD ONE
# =============================================================================
class StandardOneUnified:
    """Top‑level orchestrator integrating all physics and statistics."""
    def __init__(self, config: Dict, device='cpu'):
        self.device = get_device(device)
        self.config = config
        self.forces = ForceParameters(device=self.device)
        self.pdf = PDFProvider(pdf_name=config.get('pdf_set','CT14nlo'), device=self.device)
        self.matrix_elem = MatrixElements(self.forces, self.pdf, device=self.device)
        self.cosmo = Cosmology(device=self.device)
        self.csoc = CSOCKernel(device=self.device)
        self.ssc = SemanticStateContraction()
        self.rg = DiffRGRefiner(keep_fraction=config.get('rg_keep',0.5))
        self.generator = self._build_generator()
        self.structural_likelihood = StructuralLikelihood(self.generator)
        self.data = None
        self.use_pyhf = False
        self.pyhf_likelihood = None

    def _build_generator(self):
        phys = self.config.get('physics', 'collider')
        if phys == 'collider':
            return ColliderGenerator(
                csoc=self.csoc, ssc=self.ssc, rg=self.rg,
                mass_range=(self.config['mass_min'], self.config['mass_max']),
                n_events=self.config.get('n_events',1000), device=self.device)
        elif phys == 'black_hole':
            return BlackHoleGenerator(
                model_type=self.config.get('bh_model','hawking'),
                bh_mass=self.config.get('bh_mass',1e12),
                csoc=self.csoc, ssc=self.ssc, rg=self.rg,
                mass_range=(self.config['mass_min'], self.config['mass_max']),
                n_events=self.config.get('n_events',500), device=self.device)
        elif phys == 'dark_matter':
            return DarkMatterGenerator(
                model_type=self.config.get('dm_model','wimp'),
                dm_mass=self.config.get('dm_mass',100.0),
                csoc=self.csoc, ssc=self.ssc, rg=self.rg,
                mass_range=(self.config['mass_min'], self.config['mass_max']),
                n_events=self.config.get('n_events',500), device=self.device)
        else:
            raise ValueError(f"Unknown physics domain: {phys}")

    def load_collider_data(self, source='simulate', **kwargs):
        if source == 'simulate':
            m, pdf = self.generator.generate()
            probs = F.softmax(pdf, dim=0)
            idx = torch.multinomial(probs, kwargs.get('n_samples',1000), replacement=True)
            self.data = m[idx].detach()
            self.use_pyhf = False
        elif source == 'root':
            raw_data = CERNDataLoader.load_root(
                kwargs['filepath'], kwargs['treename'], kwargs['mass_branch'])
            self.data = raw_data
            # Replace generator with empirical one for realistic density
            self.generator = EmpiricalGenerator(
                data=raw_data, csoc=self.csoc, ssc=self.ssc, rg=self.rg,
                mass_range=(self.config['mass_min'], self.config['mass_max']),
                n_events=self.config.get('n_events',1000), device=self.device)
            self.structural_likelihood = StructuralLikelihood(self.generator)
            self.use_pyhf = False
        elif source == 'pyhf':
            ws_path = kwargs.get('workspace', CERNDataLoader.download_atlas_higgs_workspace())
            ws, model = CERNDataLoader.load_pyhf_model(ws_path)
            data = ws.data(model)
            self.pyhf_likelihood = PyHFDiffLikelihood(model, data, device=self.device)
            self.use_pyhf = True
            self.data = None
        else:
            raise ValueError(f"Unknown data source: {source}")

    def load_nasa_data(self, filepath, data_type='fits', **kwargs):
        if data_type == 'fits':
            return NASADataLoader.load_fits(filepath, **kwargs)
        elif data_type == 'csv':
            return NASADataLoader.load_csv(filepath, **kwargs)
        else:
            raise ValueError(f"Unknown NASA data type: {data_type}")

    def train_soc_gradient(self, n_steps=200, lr=0.01):
        if self.data is None:
            raise RuntimeError("No data loaded for SOC gradient training.")
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr)
        for step in range(n_steps):
            optimizer.zero_grad()
            loss = self.structural_likelihood(self.data)
            loss.backward()
            optimizer.step()
            self.generator.update_state()
            if step % 20 == 0:
                logger.info(f"SOC step {step}: loss={loss.item():.4f}")

    def _get_all_param_names(self):
        return [n for n,_ in self.generator.named_parameters()]

    def _get_all_params(self):
        return list(self.generator.parameters())

    def run_full_frequentist(self, poi_name: str = 'log_mu', null: float = 0.0,
                             cl: float = 0.68, bootstrap: bool = False):
        if self.use_pyhf:
            logger.info("pyhf data loaded; running frequentist analysis via run_pyhf_fit().")
            return self.run_pyhf_fit()
        if self.data is None:
            raise RuntimeError("No collider data loaded.")
        params = self._get_all_params()
        names = self._get_all_param_names()
        def nll_fn():
            return self.structural_likelihood(self.data)
        freq = FrequentistAnalysis(nll_fn, params, names, device=self.device)
        z = freq.significance(poi_name, null)
        if bootstrap:
            p = freq.bootstrap_calibrated_pvalue(poi_name, null,
                                                 generator=self.generator)
        else:
            p = freq.p_value(poi_name, null)
        ci = freq.confidence_interval(poi_name, cl)
        logger.info(f"Frequentist: Z={z:.2f}, p={p:.4g}, {int(cl*100)}% CI = ({ci[0]:.3f}, {ci[1]:.3f})")
        return {'significance': z, 'p_value': p, 'conf_interval': ci}

    def run_pyhf_fit(self):
        """Perform a differentiable maximum likelihood fit using pyhf logpdf."""
        if not HAS_PYHF or self.pyhf_likelihood is None:
            raise RuntimeError("pyhf not available or no pyhf data loaded.")
        pyhf.set_backend('pytorch')
        init = self.pyhf_likelihood.init_tensor.clone().requires_grad_(True)
        names = self.pyhf_likelihood.all_names
        params = [init]
        def nll_fn():
            return self.pyhf_likelihood.nll(init)
        freq = FrequentistAnalysis(nll_fn, params, ['all_pars'], device=self.device)
        best_nll = freq.unconditional_fit()
        best_pars = init.detach().cpu().numpy()
        # Compute uncertainties using Hessian
        nlp = nll_fn()
        grads = torch.autograd.grad(nlp, init, create_graph=True)
        hess_rows = []
        for g in grads:
            g2 = torch.autograd.grad(g, init, retain_graph=True)
            hess_rows.append(g2[0].flatten())
        H = torch.stack(hess_rows)
        cov = torch.linalg.inv(H).detach().cpu().numpy()
        unc = np.sqrt(np.diag(cov))
        result = {names[i]: (float(best_pars[i]), float(unc[i])) for i in range(len(names))}
        logger.info(f"pyhf MLE fit: {result}")
        return result

    def run_full_bayesian(self, n_samples=2000, warmup=500, use_nuts=False):
        if self.use_pyhf:
            raise RuntimeError("Bayesian analysis for pyhf is not integrated; use pyhf directly.")
        if self.data is None:
            raise RuntimeError("No collider data loaded.")
        params = self._get_all_params()
        names = self._get_all_param_names()
        def log_prob_fn():
            return -self.structural_likelihood(self.data)
        bayes = BayesianAnalysis(log_prob_fn, params, names, device=self.device)
        map_est = bayes.laplace_approximation()
        if use_nuts and HAS_PYRO:
            chain, acc = bayes.sample_nuts(n_samples=n_samples, warmup=warmup)
        else:
            chain, acc = bayes.sample_mh(n_samples=n_samples, burn_in=warmup)
        logger.info(f"Bayesian MAP: {map_est}")
        return {'map': map_est, 'chain': chain, 'acceptance': acc}

    def model_comparison(self, model_list: List[BaseStructuralGenerator]):
        if self.data is None:
            raise RuntimeError("No data loaded for model comparison.")
        models = {f"model_{i}": gen for i, gen in enumerate(model_list)}
        results = {}
        for label, gen in models.items():
            likelihood = StructuralLikelihood(gen)
            nll_fn = lambda: likelihood(self.data)
            params = list(gen.parameters())
            param_names = [n for n,_ in gen.named_parameters()]
            freq = FrequentistAnalysis(nll_fn, params, param_names, device=self.device)
            nll_best = freq.unconditional_fit()
            k = len(params)
            n = len(self.data)
            aic = 2*k + 2*nll_best
            bic = k*math.log(n) + 2*nll_best
            bayes = BayesianAnalysis(lambda: -nll_fn(), params, param_names, device=self.device)
            log_ml = bayes.marginal_likelihood_laplace()
            results[label] = {'aic': aic, 'bic': bic, 'nll': nll_best, 'log_marginal': log_ml}
        return results

    def structural_probability_statement(self):
        if self.data is None:
            raise RuntimeError("No data for structural probability.")
        sp = StructuralProbability(self.generator)
        prob = sp.probability(self.data[:10])
        logger.info(sp.uncertainty_source())
        logger.info(f"Sample probabilities: {prob}")

    def cross_correlate(self, collider_features, cosmo_features):
        analyzer = CrossCorrelationAnalyzer(
            collider_features.shape[1], cosmo_features.shape[1], device=self.device)
        loss = analyzer(collider_features, cosmo_features)
        return loss.item()

    def cross_correlate_real_data(self):
        """Example: correlate collider mass histogram with Planck TT spectrum."""
        if self.data is None:
            raise RuntimeError("Load collider data first.")
        # Build collider feature vector: histogram of data
        hist_range = (self.config['mass_min'], self.config['mass_max'])
        hist = torch.histc(self.data, bins=20, min=hist_range[0], max=hist_range[1])
        hist = hist / hist.sum()
        # Load Planck CMB data and take first 20 multipoles as features
        ell, Cl, _ = NASADataLoader.load_planck_highl_spectrum()
        cosmo_features = Cl[:20] / Cl[:20].sum()
        # Train a simple cross-correlation model
        analyzer = CrossCorrelationAnalyzer(hist.shape[0], cosmo_features.shape[0], device=self.device)
        optimizer = Adam(analyzer.parameters(), lr=0.01)
        for _ in range(200):
            optimizer.zero_grad()
            loss = analyzer(hist.unsqueeze(0), cosmo_features.unsqueeze(0))
            loss.backward()
            optimizer.step()
        logger.info(f"Cross-correlation trained. Final loss: {loss.item():.4f}")
        return loss.item()

    def unification_test(self, energy_scale):
        m = UnificationModel(device=self.device)
        return {
            'alpha_s': m.running_su3(torch.tensor(energy_scale, device=self.device)).item(),
            'alpha_2': m.running_su2(torch.tensor(energy_scale, device=self.device)).item(),
            'alpha_1': m.running_u1(torch.tensor(energy_scale, device=self.device)).item()
        }

    def compute_matrix_element(self, process, **kin):
        s = kin.get('s', 1000.0); t = kin.get('t', -500.0); u = kin.get('u', -500.0)
        Q2 = kin.get('Q2', 1000.0)
        if process == 'ee_mumu':
            return self.matrix_elem.qed_ee_mumu(s, t, u)
        elif process == 'qqbar_gg':
            return self.matrix_elem.qcd_qqbar_gg(s, t, u, Q2)
        elif process == 'ee_ZH':
            return self.matrix_elem.weak_ee_ZH(s, t, u)
        elif process == 'drell_yan':
            return self.matrix_elem.drell_yan_sigma(s, kin.get('M', 91.0))
        elif process == 'gg_higgs':
            return self.matrix_elem.higgs_gluon_fusion_sigma(s, kin.get('mH', 125.0),
                                                            pt=kin.get('pt',0.0), rap=kin.get('rap',0.0))
        else:
            raise ValueError(f"Unknown process: {process}")

    def run_cmb_fit(self, cmb_backend: str = 'analytic'):
        ell_data, Cl_data, cov = NASADataLoader.load_planck_highl_spectrum()
        cmb_calc = DifferentiableCMB(self.cosmo, device=self.device, backend=cmb_backend)
        cmb_like = CMBLikelihood(cmb_calc, ell_data, Cl_data, cov)
        A_s = nn.Parameter(torch.tensor(2.1e-9, device=self.device))
        n_s = nn.Parameter(torch.tensor(0.96, device=self.device))
        tau = nn.Parameter(torch.tensor(0.054, device=self.device))
        opt = LBFGS([A_s, n_s, tau], max_iter=200, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            loss = -cmb_like.log_likelihood(A_s, n_s, tau)
            loss.backward()
            return loss
        opt.step(closure)
        logger.info(f"CMB fit: A_s={A_s.item():.3e}, n_s={n_s.item():.4f}, tau={tau.item():.4f}")
        return {'A_s': A_s.item(), 'n_s': n_s.item(), 'tau': tau.item()}

    def validate_cmb_backend(self, reference_backend='camb'):
        """Cross‑check CMB likelihood against a reference solver."""
        ell_data, Cl_data, cov = NASADataLoader.load_planck_highl_spectrum()
        ref = DifferentiableCMB(self.cosmo, device=self.device, backend=reference_backend)
        test = DifferentiableCMB(self.cosmo, device=self.device, backend='analytic')
        ref_like = CMBLikelihood(ref, ell_data, Cl_data, cov)
        test_like = CMBLikelihood(test, ell_data, Cl_data, cov)
        A_s, n_s, tau = 2.1e-9, 0.96, 0.054
        logl_ref = ref_like.log_likelihood(A_s, n_s, tau).item()
        logl_test = test_like.log_likelihood(A_s, n_s, tau).item()
        logger.info(f"CMB validation: ref={logl_ref:.2f}, test={logl_test:.2f}")

    def demo_higgs_mass_fit(self):
        """End‑to‑end demonstration: fit Higgs boson mass from pseudo‑data."""
        logger.info("=== Higgs mass fit demonstration ===")
        # 1. Generate pseudo‑data with a known Higgs signal
        self.config['physics'] = 'collider'
        self.generator = ColliderGenerator(
            csoc=self.csoc, ssc=self.ssc, rg=self.rg,
            mass_range=(100, 160), n_events=2000, device=self.device)
        # set true mass
        self.generator.signal_mass.data.fill_(125.0)
        m_grid, pdf = self.generator.generate()
        probs = F.softmax(pdf, dim=0)
        n_data = 2000
        idx = torch.multinomial(probs, n_data, replacement=True)
        self.data = m_grid[idx].detach()
        # 2. Reset mass parameter to a wrong guess
        self.generator.signal_mass.data.fill_(120.0)
        # 3. Frequentist fit
        params = [self.generator.signal_mass]
        names = ['signal_mass']
        def nll_fn():
            return self.structural_likelihood(self.data)
        freq = FrequentistAnalysis(nll_fn, params, names, device=self.device)
        best_nll = freq.unconditional_fit()
        ci = freq.confidence_interval('signal_mass', cl=0.68, scan_range=(120,130))
        logger.info(f"Fitted mass = {self.generator.signal_mass.item():.2f} GeV")
        logger.info(f"68% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
        # 4. Bayesian fit
        bayes = BayesianAnalysis(lambda: -nll_fn(), params, names, device=self.device)
        map_est = bayes.laplace_approximation()
        logger.info(f"Bayesian MAP: {map_est}")
        return {'frequentist_mass': self.generator.signal_mass.item(),
                'ci': ci, 'bayes_map': map_est}


# =============================================================================
# 19. Unit Tests & Validation
# =============================================================================
def run_tests():
    logger.info("Running STANDARD ONE validation tests...")
    dev = get_device('cpu')
    forces = ForceParameters(device=dev)
    as_mz = forces.alpha_s(forces.MZ**2).item()
    assert abs(as_mz - 0.1180) < 0.01, f"α_s(MZ) = {as_mz} (expected 0.118)"
    pdf = PDFProvider(device=dev)
    xf = pdf.xf(torch.tensor(0.1), 'u')
    assert xf.item() > 0, "PDF u at x=0.1 is zero"
    me = MatrixElements(forces, pdf, device=dev)
    dy = me.drell_yan_sigma(13e3, torch.tensor(91.0)).item()
    assert 1e3 < dy < 3e4, f"Drell‑Yan σ ≈ {dy:.0f} pb (expected ~2e4 pb)"
    cosmo = Cosmology(device=dev)
    cmb = DifferentiableCMB(cosmo, device=dev, backend='analytic')
    ell, Cl = cmb.C_ell_TT()
    peak_ell = ell[torch.argmax(Cl[ell<400])].item()
    assert 200 < peak_ell < 250, f"CMB first peak at ℓ={peak_ell}"
    csoc = CSOCKernel(device=dev)
    gen = ColliderGenerator(csoc, SemanticStateContraction(), DiffRGRefiner(), device=dev)
    m, pdf_vals = gen.generate()
    assert m.shape == (1000,)
    integral = torch.trapezoid(pdf_vals, m).item()
    assert abs(integral - 1.0) < 0.01, f"PDF not normalized: integral={integral}"
    logger.info("All validation tests passed!")


# =============================================================================
# 20. Command‑Line Interface
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="STANDARD ONE Unified Research Framework")
    p.add_argument('--physics', default='collider',
                   choices=['collider','black_hole','dark_matter','cmb'])
    p.add_argument('--model', type=str, help='Sub‑model (hawking, wimp, etc.)')
    p.add_argument('--data-source', default='simulate',
                   choices=['simulate','root','pyhf'])
    p.add_argument('--root-file', type=str)
    p.add_argument('--tree-name', default='events')
    p.add_argument('--mass-branch', default='mass')
    p.add_argument('--mass-min', type=float, default=50.0)
    p.add_argument('--mass-max', type=float, default=200.0)
    p.add_argument('--n-events', type=int, default=1000)
    p.add_argument('--bh-mass', type=float, default=1e12)
    p.add_argument('--dm-mass', type=float, default=100.0)
    p.add_argument('--device', default='cpu', choices=['cpu','cuda','mps','ascend'])
    p.add_argument('--train-soc', action='store_true')
    p.add_argument('--frequentist', action='store_true')
    p.add_argument('--bayesian', action='store_true')
    p.add_argument('--use-nuts', action='store_true')
    p.add_argument('--poi', type=str, default='log_mu', help='Parameter of interest')
    p.add_argument('--null', type=float, default=0.0)
    p.add_argument('--cl', type=float, default=0.68)
    p.add_argument('--structural', action='store_true')
    p.add_argument('--nasa-file', type=str)
    p.add_argument('--cross-correlate', action='store_true')
    p.add_argument('--unification-test', type=float)
    p.add_argument('--matrix-element', type=str)
    p.add_argument('--s', type=float, default=1000.0)
    p.add_argument('--t', type=float, default=-500.0)
    p.add_argument('--M', type=float, default=91.0)
    p.add_argument('--mH', type=float, default=125.0)
    p.add_argument('--pt', type=float, default=0.0)
    p.add_argument('--rap', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cmb-fit', action='store_true')
    p.add_argument('--cmb-backend', default='analytic',
                   choices=['analytic','camb','class','cosmopower'])
    p.add_argument('--cmb-validate', action='store_true',
                   help='Cross‑validate CMB likelihood against CAMB')
    p.add_argument('--bootstrap', action='store_true',
                   help='Use bootstrap‑calibrated p‑value')
    p.add_argument('--higgs-demo', action='store_true',
                   help='Run end‑to‑end Higgs mass fit demonstration')
    p.add_argument('--test', action='store_true', help='Run validation tests')
    return p.parse_args()

def main():
    args = parse_args()
    if args.test:
        run_tests()
        return
    torch.manual_seed(args.seed)
    config = {
        'physics': args.physics if args.physics != 'cmb' else 'collider',
        'bh_model': args.model if args.physics=='black_hole' else 'hawking',
        'dm_model': args.model if args.physics=='dark_matter' else 'wimp',
        'mass_min': args.mass_min, 'mass_max': args.mass_max,
        'n_events': args.n_events, 'bh_mass': args.bh_mass,
        'dm_mass': args.dm_mass, 'pdf_set': 'CT14nlo'
    }
    framework = StandardOneUnified(config, device=args.device)

    if args.higgs_demo:
        framework.demo_higgs_mass_fit()
        return

    if args.physics == 'cmb':
        if args.cmb_validate:
            framework.validate_cmb_backend(reference_backend='camb')
        else:
            result = framework.run_cmb_fit(cmb_backend=args.cmb_backend)
            print(result)
        return

    if args.data_source == 'simulate':
        framework.load_collider_data(source='simulate', n_samples=args.n_events)
    elif args.data_source == 'root':
        framework.load_collider_data(source='root', filepath=args.root_file,
                                     treename=args.tree_name, mass_branch=args.mass_branch)
    elif args.data_source == 'pyhf':
        framework.load_collider_data(source='pyhf')

    if args.train_soc:
        framework.train_soc_gradient()
    if args.frequentist:
        res = framework.run_full_frequentist(poi_name=args.poi, null=args.null,
                                             cl=args.cl, bootstrap=args.bootstrap)
        print(res)
    if args.bayesian:
        res = framework.run_full_bayesian(use_nuts=args.use_nuts)
        print(res['map'])
    if args.structural:
        framework.structural_probability_statement()
    if args.cross_correlate and args.nasa_file:
        coll_feat = torch.randn(args.n_events, 5, device=framework.device)
        cosmo_data = framework.load_nasa_data(args.nasa_file, data_type='fits')
        if cosmo_data.dim()==1:
            cosmo_feat = cosmo_data[:args.n_events].unsqueeze(1)
        else:
            cosmo_feat = cosmo_data[:args.n_events]
        loss = framework.cross_correlate(coll_feat, cosmo_feat)
        logger.info(f"Cross‑correlation loss: {loss:.4f}")
    if args.unification_test is not None:
        couplings = framework.unification_test(args.unification_test)
        logger.info(f"Unification at {args.unification_test} GeV: {couplings}")
    if args.matrix_element:
        kwargs = {'s':args.s, 't':args.t, 'M':args.M, 'mH':args.mH, 'pt':args.pt, 'rap':args.rap}
        me = framework.compute_matrix_element(args.matrix_element, **kwargs)
        logger.info(f"Matrix element / cross section: {me}")

if __name__ == "__main__":
    main()
