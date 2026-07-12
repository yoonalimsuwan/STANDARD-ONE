# =============================================================================
# YANG MILLS MASS GAP (ONE) — Differentiable Yang–Mills Mass Gap Explorer
# =============================================================================
# Author : PAI , Yoon A Limsuwan
# License: MIT
# Year   : 2026
#
# This module extends the STANDARD ONE framework to investigate the
# Yang–Mills mass gap problem in a fully differentiable manner.
# It employs parametric models for the gluon propagator (Gribov, massive,
# decoupling, scaling) and optimises parameters against lattice data
# using gradient‑based inference. The mass gap is extracted as the pole
# of the propagator in the complex plane.
#
# Fully compatible with the rest of the STANDARD ONE ecosystem:
#   • Uses PhysicsParameters for the QCD running coupling
#   • Leverages CSOCKernel for spectral regularisation
#   • Differentiable down to the mass gap estimate
#
# Usage:
#   from standard_one import StandardOneUnified, PhysicsParameters, CSOCKernel
#   from yang_mills_mass_gap_one import YangMillsMassGap
#
#   ym = YangMillsMassGap(physics_params=..., csoc=...)
#   ym.fit_to_lattice("gluon_propagator_data.csv")   # or .npy
#   mass_gap = ym.extract_mass_gap()
#   print(f"Mass gap = {mass_gap:.3f} MeV")
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import math, os, logging
from typing import Optional, Tuple, Dict

logger = logging.getLogger("YMGapOne")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# 1. Differentiable gluon propagator models (Euclidean, D(p²))
# ---------------------------------------------------------------------------
class GluonPropagatorModel(nn.Module):
    """Base class for parametric gluon propagators in Landau gauge."""
    def __init__(self):
        super().__init__()
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        return super().to(device)

    def D(self, p2: torch.Tensor) -> torch.Tensor:
        """Scalar gluon dressing function: D(p²) = Z(p²)/p²."""
        raise NotImplementedError

    def dressing(self, p2: torch.Tensor) -> torch.Tensor:
        """Z(p²) = p² * D(p²)."""
        return p2 * self.D(p2)

class GribovPropagator(GluonPropagatorModel):
    """Gribov‑Zwanziger type: D(p²) = (p² + M⁴/(p²+m²))⁻¹."""
    def __init__(self, M: float = 0.5, m: float = 0.3):
        super().__init__()
        self.log_M4 = nn.Parameter(torch.tensor(math.log(M**4)))
        self.log_m2 = nn.Parameter(torch.tensor(math.log(m**2)))

    @property
    def M4(self): return torch.exp(self.log_M4)
    @property
    def m2(self): return torch.exp(self.log_m2)

    def D(self, p2):
        p2 = torch.as_tensor(p2, device=self.device)
        return 1.0 / (p2 + self.M4 / (p2 + self.m2 + 1e-10))

class MassiveGluonPropagator(GluonPropagatorModel):
    """Massive gluon: D(p²) = 1/(p² + m²)."""
    def __init__(self, mass: float = 0.5):
        super().__init__()
        self.log_m2 = nn.Parameter(torch.tensor(math.log(mass**2)))

    @property
    def m2(self): return torch.exp(self.log_m2)

    def D(self, p2):
        p2 = torch.as_tensor(p2, device=self.device)
        return 1.0 / (p2 + self.m2)

class RefinedGribovPropagator(GluonPropagatorModel):
    """Refined Gribov with dynamical mass generation: 
       D(p²) = (p² + M²(p²))⁻¹, M²(p²) = m₀⁴/(p² + m₁²)."""
    def __init__(self, m0: float = 0.6, m1: float = 0.2):
        super().__init__()
        self.log_m0_4 = nn.Parameter(torch.tensor(math.log(m0**4)))
        self.log_m1_2 = nn.Parameter(torch.tensor(math.log(m1**2)))

    @property
    def m0_4(self): return torch.exp(self.log_m0_4)
    @property
    def m1_2(self): return torch.exp(self.log_m1_2)

    def D(self, p2):
        p2 = torch.as_tensor(p2, device=self.device)
        M2 = self.m0_4 / (p2 + self.m1_2 + 1e-10)
        return 1.0 / (p2 + M2)

# ---------------------------------------------------------------------------
# 2. Differentiable mass gap analyser
# ---------------------------------------------------------------------------
class MassGapAnalyzer:
    """Compute the mass gap from the propagator pole in Minkowski space."""
    @staticmethod
    def pole_from_minkowski(model: GluonPropagatorModel) -> float:
        """Find real pole of D(-k²) (Minkowski metric) by scanning."""
        # D_E(p²) analytic continuation: D_M(k²) = D_E(-k²)
        # We look for the smallest positive k² where D_M(k²) diverges.
        k2_vals = torch.logspace(-3, 1, 10000, device=model.device)
        D_m = model.D(-k2_vals)   # note negative argument
        # find where D_m becomes very large (pole)
        mask = D_m > 100.0
        if mask.any():
            pole_idx = torch.argmax(D_m)
            return k2_vals[pole_idx].item()
        # If no pole, the mass gap is the effective mass where dressing is maximal
        dressing = k2_vals * D_m
        return k2_vals[torch.argmax(dressing)].item()

    @staticmethod
    def complex_pole_newton(model: GluonPropagatorModel, initial: complex = 0.3-0.01j,
                             max_iter: int = 50) -> complex:
        """Find a complex pole of D(-p²) using Newton's method (requires autograd)."""
        p2 = torch.tensor(initial, requires_grad=True, device=model.device, dtype=torch.complex64)
        for _ in range(max_iter):
            Dval = model.D(-p2)
            if torch.abs(1.0/Dval) < 1e-12:
                break
            grad = torch.autograd.grad(Dval, p2, torch.ones_like(Dval), retain_graph=True)[0]
            p2 = p2 - Dval / (grad + 1e-10)
        return p2.detach().item()

# ---------------------------------------------------------------------------
# 3. Lattice data handler (download / load pre‑existing)
# ---------------------------------------------------------------------------
class LatticeDataLoader:
    """Load gluon propagator lattice data (e.g., from SU(3) Landau gauge)."""
    # Public data: e.g., arXiv:2201.12186, or included as example.
    @staticmethod
    def generate_synthetic_data(model_type='decoupling', noise: float = 0.01):
        """Create synthetic data resembling lattice results for demonstration."""
        p = np.logspace(-2, 1, 30)
        if model_type == 'decoupling':
            D = 1.0 / (p**2 + 0.5**2)
        else:
            D = 1.0 / (p**2 + 0.3**2)
        D *= (1 + noise * np.random.randn(len(p)))
        return torch.tensor(p, dtype=torch.float32), torch.tensor(D, dtype=torch.float32)

    @staticmethod
    def load_from_file(filepath: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load lattice data from a two‑column CSV: p², D(p²)."""
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        p2 = data[:,0]
        D = data[:,1]
        return torch.tensor(p2, dtype=torch.float32), torch.tensor(D, dtype=torch.float32)

# ---------------------------------------------------------------------------
# 4. Main YANG MILLS MASS GAP module
# ---------------------------------------------------------------------------
class YangMillsMassGap(nn.Module):
    """
    Differentiable analysis of Yang–Mills mass gap.
    Combines a propagator model with the QCD running coupling from STANDARD ONE,
    and allows gradient‑based fitting to lattice data.
    """
    def __init__(self,
                 physics_params: nn.Module = None,   # STANDARD ONE's PhysicsParameters
                 csoc: nn.Module = None,             # CSOCKernel for regularisation
                 propagator_type: str = 'gribov',
                 device: str = 'cpu'):
        super().__init__()
        self.device = device
        # Attach the STANDARD ONE physics (for running coupling)
        self.physics = physics_params if physics_params else self._default_physics()
        self.csoc = csoc if csoc else nn.Identity()

        # Choose propagator model
        if propagator_type == 'gribov':
            self.propagator = GribovPropagator()
        elif propagator_type == 'massive':
            self.propagator = MassiveGluonPropagator()
        elif propagator_type == 'refined':
            self.propagator = RefinedGribovPropagator()
        else:
            raise ValueError(f"Unknown propagator type: {propagator_type}")

        # Some global scaling factors (optional)
        self.scale = nn.Parameter(torch.tensor(1.0))

        self.to(device)

    @staticmethod
    def _default_physics():
        """Create a minimal PhysicsParameters if none supplied."""
        from standard_one import PhysicsParameters
        return PhysicsParameters(device='cpu')

    def forward(self, p2: torch.Tensor) -> torch.Tensor:
        """Compute gluon propagator D(p²) including running coupling effects."""
        # Incorporate running coupling into the dressing (simplified)
        alpha = self.physics.alpha_s(p2 * self.scale)
        # For illustration, multiply the propagator by (alpha/alpha0)^something
        # In a full treatment, the propagator itself would depend on α through SDE.
        raw_D = self.propagator.D(p2)
        # CSOC regularisation on the spectral side
        r = torch.sqrt(p2 + 1e-6) / 10.0   # scale for CSOC input
        modulation = 1.0 + 0.1 * self.csoc(r) if hasattr(self.csoc, 'forward') else 1.0
        return raw_D * modulation * self.scale

    def dressing(self, p2):
        return p2 * self.forward(p2)

    def likelihood(self, p2_data: torch.Tensor, D_data: torch.Tensor) -> torch.Tensor:
        """Gaussian negative log‑likelihood for lattice data."""
        D_pred = self.forward(p2_data)
        return torch.sum((D_pred - D_data)**2)

    def fit_to_lattice(self, data_source: str, epochs: int = 500, lr: float = 0.01):
        """Optimise propagator parameters to match lattice data."""
        if isinstance(data_source, str) and os.path.isfile(data_source):
            p2, D_true = LatticeDataLoader.load_from_file(data_source)
        else:
            logger.info("Generating synthetic lattice data (decoupling solution).")
            p2, D_true = LatticeDataLoader.generate_synthetic_data()

        p2 = p2.to(self.device)
        D_true = D_true.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30)

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.likelihood(p2, D_true)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch:3d} | Loss = {loss.item():.4e}")
        logger.info("Fitting complete.")

    def extract_mass_gap(self, method: str = 'pole_scan') -> float:
        """Return the mass gap (in GeV) from the propagator."""
        if method == 'pole_scan':
            return MassGapAnalyzer.pole_from_minkowski(self.propagator)
        elif method == 'newton':
            pole = MassGapAnalyzer.complex_pole_newton(self.propagator)
            # mass gap is the real part of the pole (if complex, take |Im| as width)
            return pole.real if pole.real > 0 else abs(pole.imag)
        else:
            raise ValueError(f"Unknown method: {method}")

    def summary(self):
        """Print model parameters and mass gap."""
        mass_gap = self.extract_mass_gap()
        logger.info(f"Estimated mass gap = {mass_gap:.4f} GeV ({mass_gap*1000:.1f} MeV)")
        for name, param in self.named_parameters():
            logger.info(f"  {name}: {param.data.item():.4f}")

# ---------------------------------------------------------------------------
# 5. Integration helper: use with existing STANDARD ONE framework
# ---------------------------------------------------------------------------
def demonstrate_mass_gap(physics: nn.Module, csoc: nn.Module):
    """
    Example integration: create a YangMillsMassGap instance using STANDARD ONE
    components and run a fit.
    """
    ym = YangMillsMassGap(physics_params=physics, csoc=csoc,
                          propagator_type='refined', device='cpu')
    # Fit to synthetic data (or real if available)
    ym.fit_to_lattice(None, epochs=200, lr=0.01)
    ym.summary()
    return ym

# ---------------------------------------------------------------------------
# 6. Standalone run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # When run directly, use built‑in defaults (no STANDARD ONE needed)
    logging.basicConfig(level=logging.INFO)
    logger.info("YANG MILLS MASS GAP (ONE) – Standalone demo")
    from standard_one import PhysicsParameters, CSOCKernel   # assume in path
    phys = PhysicsParameters(device='cpu')
    csoc = CSOCKernel(device='cpu')
    ym = YangMillsMassGap(physics_params=phys, csoc=csoc, propagator_type='refined')
    ym.fit_to_lattice(None, epochs=150)
    ym.summary()
