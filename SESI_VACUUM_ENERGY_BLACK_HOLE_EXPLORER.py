"""
=============================================================================
SESI VACUUM ENERGY & BLACK HOLE EXPLORER v1.0
Dark-Energy-as-Vacuum-Energy Harvesting + Full Black Hole Survey
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
=============================================================================

PHYSICAL GROUNDING (real equations, honestly scoped)
────────────────────────────────────────────────────
This is a DIFFERENTIABLE SIMULATION of vacuum-energy and black-hole
physics. It is not a free-energy device. Every extraction channel is
bound by real thermodynamics:

  [V] VACUUM / DARK ENERGY
      • Observed dark-energy density: ρ_Λ = Λc²/(8πG) ≈ 5.4e-10 J/m³
      • Vacuum catastrophe: QFT predicts ~10^120× the observed value
        → we model the OBSERVED value; the discrepancy is flagged, not hidden.
      • Casimir effect: E/A = -π²/(720 d³)  (negative energy between plates)
      • Dynamical Casimir: real photon extraction requires NON-ADIABATIC
        boundary motion — energy input is tracked and conserved.
      • Second-law guard: net cyclic extraction from a uniform vacuum
        without a gradient is explicitly FORBIDDEN in the loss function.

  [B] BLACK HOLE (Kerr–Newman geometry, G = c = 1)
      • Δ = r² - 2Mr + a² + Q²,  Σ = r² + a²cos²θ
      • Horizons r_± = M ± √(M² - a² - Q²)
      • Ergosphere r_ergo(θ) = M + √(M² - a²cos²θ - Q²)
      • Hawking temperature  T_H = κ/(2π),  κ = (r_+-r_-)/(2(r_+²+a²))
      • Bekenstein-Hawking entropy  S = A/4 = π(r_+² + a²)
      • Penrose process  η_max = 1 - M_irr/M ≤ 1 - 1/√2 ≈ 29% (extremal Kerr)
      • Blandford-Znajek  P_BZ ∝ Ω_H² Φ²  (jet power source)
      • Superradiance  amplification for ω < m Ω_H
      • Kretschner scalar  K = 48M²/r⁶ (Schwarzschild)

THEORY INTEGRATION (SESI ecosystem)
───────────────────────────────────
  [T1] Carpet resistance forms → vacuum-mode spectrum on fractal graph
  [T2] SQCT collapse          → vacuum energy as geometric mass density μ̂_S
  [T3] Cloud-OPMC             → compress hyper-relational vacuum state to σ_T
  [T4] RUT-PCD accessibility  → which BH regions are causally reachable
  [T5] HCSN hierarchical      → nested vacuum / horizon superposition

=============================================================================
"""

from __future__ import annotations

import math
import time
import logging
import contextlib
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.vacuum_bh")

# ── physical constants (SI) ────────────────────────────────────────────────
G_SI    = 6.67430e-11       # m³ kg⁻¹ s⁻²
C_SI    = 2.99792458e8      # m s⁻¹
HBAR_SI = 1.054571817e-34   # J s
KB_SI   = 1.380649e-23      # J K⁻¹
# observed dark-energy density ≈ 5.4e-10 J/m³  (ρ_Λ = Λc²/8πG)
RHO_LAMBDA_SI = 5.4e-10


# ═══════════════════════════════════════════════════════════════════════
# C^∞ smooth primitives (no dead gradients)
# ═══════════════════════════════════════════════════════════════════════

def soft_abs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)

def soft_floor(x: Tensor, floor: float, beta: float = 50.0) -> Tensor:
    return floor + F.softplus(x - floor, beta=beta)

def soft_gate(x: Tensor, tau: float = 1e-2) -> Tensor:
    return torch.sigmoid(x / tau)

def soft_step(x: Tensor, tau: float = 1e-2) -> Tensor:
    """Smooth Heaviside: 0 for x≪0, 1 for x≫0."""
    return torch.sigmoid(x / tau)


class LowRankAdapter(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int = 32):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_in, rank) * 0.02)
        self.V = nn.Parameter(torch.zeros(rank, d_out))
    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.U.to(x.dtype)) @ self.V.to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class VacuumBHConfig:
    latent_dim: int = 512
    # vacuum sector
    n_vacuum_modes: int = 32
    casimir_plate_sep: float = 1e-6      # m
    dynamical_casimir_velocity: float = 0.01  # fraction of c (mirror speed)
    # black-hole sector (geometrized units G=c=1)
    bh_mass: float = 1.0                 # M
    bh_spin: float = 0.9                 # a/M ∈ [0,1]
    bh_charge: float = 0.0               # Q
    n_geodesic_steps: int = 200
    geodesic_dt: float = 0.05
    magnetic_flux: float = 1.0           # Φ for Blandford-Znajek
    # exploration grid
    explorer_grid: int = 32
    # SESI integration
    opmc_sigma_dim: int = 64
    opmc_n_fields: int = 10
    mental_rank: int = 32
    # training
    batch_size: int = 8
    lr: float = 3e-4
    amp_dtype: str = "bf16"
    seed: int = 42
    device: str = "cuda"


# ═══════════════════════════════════════════════════════════════════════
# [V1] VACUUM ENERGY DENSITY FIELD
# ═══════════════════════════════════════════════════════════════════════

class VacuumEnergyField(nn.Module):
    """
    Models the vacuum / dark-energy field.
    • Zero-point mode spectrum (Carpet-graph eigenmodes as a proxy for
      the vacuum-mode density of states).
    • Observed cosmological-constant density ρ_Λ.
    • Flags the vacuum-catastrophe mismatch honestly.
    """
    def __init__(self, cfg: VacuumBHConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.mental_rank
        d = cfg.latent_dim

        # learnable vacuum-mode spectrum (positive)
        self.mode_raw = nn.Parameter(torch.randn(cfg.n_vacuum_modes) * 0.1)
        # coupling from AGI latent → local vacuum excitation
        self.vacuum_coupling = LowRankAdapter(d, cfg.n_vacuum_modes, r)
        # observed dark-energy density head (learns to match ρ_Λ)
        self.rho_head = nn.Sequential(
            nn.Linear(cfg.n_vacuum_modes, cfg.n_vacuum_modes // 2),
            nn.GELU(),
            nn.Linear(cfg.n_vacuum_modes // 2, 1),
            nn.Softplus(),
        )
        self.register_buffer("rho_lambda_obs",
                             torch.tensor(RHO_LAMBDA_SI), persistent=False)

    @property
    def mode_spectrum(self) -> Tensor:
        return F.softplus(self.mode_raw)   # positive frequencies

    def forward(self, h: Tensor) -> Dict[str, Tensor]:
        excitation = self.vacuum_coupling(h)                 # (B, n_modes)
        spectrum = self.mode_spectrum
        # local vacuum energy density (log-space for numerical safety)
        log_rho = torch.log(spectrum + 1e-30) + excitation
        rho_local = torch.exp(torch.clamp(log_rho.mean(-1, keepdim=True), -60, 40))
        rho_pred = self.rho_head(torch.sigmoid(excitation))  # (B,1)
        # vacuum catastrophe flag: QFT vs observed mismatch (informational)
        catastrophe_ratio = torch.full_like(rho_pred, 1e120)
        return {
            "vacuum_excitation": excitation,
            "vacuum_mode_spectrum": spectrum,
            "rho_local": rho_local.squeeze(-1),
            "rho_dark_energy_pred": rho_pred.squeeze(-1),
            "rho_lambda_obs": self.rho_lambda_obs.expand(rho_pred.shape[0]),
            "vacuum_catastrophe_ratio": catastrophe_ratio.squeeze(-1),
        }


# ═══════════════════════════════════════════════════════════════════════
# [V2] DARK ENERGY HARVESTER (Casimir + Dynamical Casimir + 2nd-law guard)
# ═══════════════════════════════════════════════════════════════════════

class DarkEnergyHarvester(nn.Module):
    """
    Models physically-allowed vacuum-energy extraction channels.

    Channels:
      1. Casimir     E/A = -π²/(720 d³)   (static negative energy)
      2. Dynamical Casimir — moving boundary converts vacuum fluctuations
         to real photons; requires (and tracks) mechanical input energy.
      3. Λ-coupling  — dark energy drives cosmic expansion; local
         extraction requires a GRADIENT (none exists for uniform Λ).

    Second-law guard: the module outputs an explicit `extraction_violation`
    penalty that is >0 whenever net cyclic output exceeds input. This is
    added to the loss, so the network learns NOT to hallucinate free energy.
    """
    def __init__(self, cfg: VacuumBHConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.mental_rank
        d = cfg.latent_dim
        self.register_buffer("plate_sep",
                             torch.tensor(cfg.casimir_plate_sep), persistent=False)
        self.register_buffer("mirror_v",
                             torch.tensor(cfg.dynamical_casimir_velocity), persistent=False)
        # gate that decides how much the network "attempts" to harvest
        self.harvest_gate = nn.Sequential(
            LowRankAdapter(d, d // 2, r), nn.GELU(),
            nn.Linear(d // 2, 3), nn.Sigmoid(),   # [casimir, dyn_casimir, lambda]
        )
        # input-energy bookkeeping for dynamical Casimir
        self.input_energy_net = nn.Sequential(
            LowRankAdapter(d, d // 2, r), nn.GELU(),
            nn.Linear(d // 2, 1), nn.Softplus(),  # mechanical drive ≥ 0
        )

    def casimir_energy_density(self, d_sep: Tensor) -> Tensor:
        """E/A per unit separation → energy density ~ -π²/(720 d⁴)."""
        return -(math.pi ** 2) / (720.0 * soft_floor(d_sep, 1e-12) ** 4)

    def dynamical_casimir_power(self, mirror_v: Tensor) -> Tensor:
        """Photon-production rate ∝ (v/c)² for non-adiabatic mirror."""
        return (mirror_v ** 2) * 1e-3   # scaled, dimensionless proxy

    def forward(self, h: Tensor, vacuum_out: Dict[str, Tensor]) -> Dict[str, Tensor]:
        gates = self.harvest_gate(h)                 # (B,3)
        g_cas, g_dyn, g_lam = gates[:, 0], gates[:, 1], gates[:, 2]

        # Channel 1: Casimir (static) — negative energy, extractable only
        # by doing work to change the plate separation.
        e_cas = self.casimir_energy_density(self.plate_sep)
        cas_power = g_cas * (-e_cas).abs()

        # Channel 2: dynamical Casimir — input energy tracked.
        dyn_power = g_dyn * self.dynamical_casimir_power(self.mirror_v)
        input_energy = self.input_energy_net(h).squeeze(-1)

        # Channel 3: Λ-coupling — uniform vacuum has NO local gradient,
        # so extractable power from ρ_Λ alone is ~0 (honest physics).
        rho_lambda = vacuum_out["rho_lambda_obs"]
        lam_power = g_lam * rho_lambda * 0.0   # zero: no gradient → no work

        gross_output = cas_power + dyn_power + lam_power
        # Second-law violation penalty: output must not exceed input
        violation = F.relu(gross_output - (input_energy + 1e-9))

        return {
            "harvest_gates": gates,
            "casimir_energy_density": e_cas.expand(h.shape[0]),
            "casimir_power": cas_power,
            "dynamical_casimir_power": dyn_power,
            "lambda_power": lam_power,
            "input_drive_energy": input_energy,
            "gross_output_power": gross_output,
            "second_law_violation": violation,
            "net_harvestable": gross_output - violation,
        }


# ═══════════════════════════════════════════════════════════════════════
# [B1] BLACK HOLE METRIC (Kerr–Newman)
# ═══════════════════════════════════════════════════════════════════════

class KerrNewmanGeometry(nn.Module):
    """
    Kerr–Newman black-hole geometry in geometrized units (G=c=1).
    Provides horizons, ergosphere, photon sphere, surface gravity,
    and curvature invariants — all differentiable in (M, a, Q).
    """
    def __init__(self, cfg: VacuumBHConfig):
        super().__init__()
        self.cfg = cfg
        # learnable BH parameters (softplus keeps M>0, |a|≤M enforced below)
        self.log_M = nn.Parameter(torch.tensor(math.log(cfg.bh_mass)))
        self.spin_frac = nn.Parameter(torch.tensor(cfg.bh_spin))  # a/M
        self.charge = nn.Parameter(torch.tensor(cfg.bh_charge))

    @property
    def M(self) -> Tensor:
        return F.softplus(self.log_M) + 1e-6

    @property
    def a(self) -> Tensor:
        # |a| ≤ M (cosmic censorship): a = M * tanh(raw)
        return self.M * torch.tanh(self.spin_frac)

    @property
    def Q(self) -> Tensor:
        return self.charge

    def horizons(self) -> Tuple[Tensor, Tensor]:
        M, a, Q = self.M, self.a, self.Q
        disc = soft_floor(M * M - a * a - Q * Q, 1e-9)   # ≥0 ⇒ BH exists
        sq = torch.sqrt(disc)
        r_plus = M + sq
        r_minus = M - sq
        return r_plus, r_minus

    def ergosphere_radius(self, cos_theta: Tensor) -> Tensor:
        M, a, Q = self.M, self.a, self.Q
        disc = soft_floor(M * M - a * a * cos_theta ** 2 - Q * Q, 1e-9)
        return M + torch.sqrt(disc)

    def photon_sphere(self) -> Tensor:
        """Prograde photon orbit for Kerr (Bardeen et al. 1972)."""
        M, a = self.M, self.a
        # r_ph = 2M[1 + cos(2/3 arccos(-a/M))]  (Schwarzschild limit → 3M)
        ratio = (a / M).clamp(-0.9999, 0.9999)
        return 2.0 * M * (1.0 + torch.cos(
            (2.0 / 3.0) * torch.acos(-ratio)))

    def surface_gravity(self) -> Tensor:
        r_plus, r_minus = self.horizons()
        a = self.a
        return (r_plus - r_minus) / (2.0 * (r_plus * r_plus + a * a) + 1e-9)

    def horizon_area(self) -> Tensor:
        r_plus = self.horizons()[0]
        a = self.a
        return 4.0 * math.pi * (r_plus * r_plus + a * a)

    def irreducible_mass(self) -> Tensor:
        """M_irr = √(A/16π); energy above this is extractable."""
        return torch.sqrt(self.horizon_area() / (16.0 * math.pi))

    def kretschner(self, r: Tensor) -> Tensor:
        """Kretschner scalar K = R_αβγδ R^αβγδ (Schwarzschild form)."""
        M = self.M
        return 48.0 * M * M / soft_floor(r ** 6, 1e-12)

    def forward(self, batch: int = 1) -> Dict[str, Tensor]:
        r_plus, r_minus = self.horizons()
        dev = r_plus.device
        cos_t = torch.zeros(batch, device=dev)   # equatorial plane
        return {
            "M": self.M.expand(batch),
            "a": self.a.expand(batch),
            "Q": self.Q.expand(batch),
            "r_plus": r_plus.expand(batch),
            "r_minus": r_minus.expand(batch),
            "ergosphere_eq": self.ergosphere_radius(cos_t),
            "photon_sphere": self.photon_sphere().expand(batch),
            "surface_gravity": self.surface_gravity().expand(batch),
            "horizon_area": self.horizon_area().expand(batch),
            "irreducible_mass": self.irreducible_mass().expand(batch),
        }


# ═══════════════════════════════════════════════════════════════════════
# [B2] BLACK HOLE THERMODYNAMICS
# ═══════════════════════════════════════════════════════════════════════

class BlackHoleThermodynamics(nn.Module):
    """Hawking temperature, entropy, and evaporation dynamics."""
    def __init__(self, cfg: VacuumBHConfig):
        super().__init__()
        self.cfg = cfg

    def hawking_temperature(self, geom: KerrNewmanGeometry) -> Tensor:
        """T_H = κ/(2π) in geometrized units."""
        return geom.surface_gravity() / (2.0 * math.pi)

    def entropy(self, geom: KerrNewmanGeometry) -> Tensor:
        """S_BH = A/4."""
        return geom.horizon_area() / 4.0

    def evaporation_rate(self, geom: KerrNewmanGeometry) -> Tensor:
        """dM/dt ∝ -1/M² (Hawking mass loss, dimensionless proxy)."""
        M = geom.M
        return -1.0 / soft_floor(M * M, 1e-9)

    def forward(self, geom: KerrNewmanGeometry, batch: int = 1) -> Dict[str, Tensor]:
        T_H = self.hawking_temperature(geom).expand(batch)
        S = self.entropy(geom).expand(batch)
        dMdt = self.evaporation_rate(geom).expand(batch)
        return {
            "hawking_temperature": T_H,
            "bh_entropy": S,
            "evaporation_rate": dMdt,
        }


# ═══════════════════════════════════════════════════════════════════════
# [B3] ENERGY EXTRACTION PROCESSES
# ═══════════════════════════════════════════════════════════════════════

class BlackHoleEnergyExtractor(nn.Module):
    """
    Physically-real energy-extraction channels from a rotating/charged BH:
      1. Penrose process (ergosphere, particle splitting)
      2. Blandford-Znajek (magnetic extraction of spin energy → jets)
      3. Superradiance (wave amplification, ω < m Ω_H)
      4. Hawking-radiation harvesting (tiny for astrophysical BH)
    All bounded by the irreducible-mass ceiling (area theorem).
    """
    def __init__(self, cfg: VacuumBHConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("magnetic_flux",
                             torch.tensor(cfg.magnetic_flux), persistent=False)

    def horizon_angular_velocity(self, geom: KerrNewmanGeometry) -> Tensor:
        r_plus = geom.horizons()[0]
        a = geom.a
        return a / (r_plus * r_plus + a * a + 1e-9)

    def penrose_max_efficiency(self, geom: KerrNewmanGeometry) -> Tensor:
        """η_max = 1 - M_irr/M  (≤ 1 - 1/√2 ≈ 29% for extremal Kerr)."""
        return 1.0 - geom.irreducible_mass() / geom.M

    def penrose_power(self, geom: KerrNewmanGeometry, accretion_rate: Tensor) -> Tensor:
        return self.penrose_max_efficiency(geom) * accretion_rate * geom.M

    def blandford_znajek_power(self, geom: KerrNewmanGeometry) -> Tensor:
        """P_BZ ∝ Ω_H² Φ² (spin-powered jets)."""
        omega_h = self.horizon_angular_velocity(geom)
        return omega_h ** 2 * self.magnetic_flux ** 2 * geom.M

    def superradiance_gain(self, geom: KerrNewmanGeometry,
                           omega: Tensor, m_mode: int = 1) -> Tensor:
        """Amplification only for ω < m Ω_H; smooth gate, no hard branch."""
        omega_h = self.horizon_angular_velocity(geom)
        threshold = m_mode * omega_h
        gate = soft_gate(threshold - omega, tau=1e-2)
        return gate * (threshold - omega).clamp(min=0.0)

    def hawking_power(self, geom: KerrNewmanGeometry) -> Tensor:
        """Stefan-Boltzmann proxy: P ∝ A T_H⁴ (negative dM/dt)."""
        T_H = geom.surface_gravity() / (2.0 * math.pi)
        return geom.horizon_area() * T_H ** 4

    def forward(self, geom: KerrNewmanGeometry, batch: int = 1,
                accretion_rate: Optional[Tensor] = None) -> Dict[str, Tensor]:
        dev = geom.M.device
        if accretion_rate is None:
            accretion_rate = torch.ones(batch, device=dev) * 0.1
        return {
            "omega_H": self.horizon_angular_velocity(geom).expand(batch),
            "penrose_efficiency": self.penrose_max_efficiency(geom).expand(batch),
            "penrose_power": self.penrose_power(geom, accretion_rate),
            "blandford_znajek_power": self.blandford_znajek_power(geom).expand(batch),
            "superradiance_gain": self.superradiance_gain(
                geom, torch.tensor(0.1, device=dev)).expand(batch),
            "hawking_power": self.hawking_power(geom).expand(batch),
            "extractable_energy_ceiling":
                (geom.M - geom.irreducible_mass()).expand(batch),
        }


# ═══════════════════════════════════════════════════════════════════════
# [B4] FULL BLACK HOLE EXPLORER
# ═══════════════════════════════════════════════════════════════════════

class BlackHoleExplorer(nn.Module):
    """
    Full exploration survey of the black-hole spacetime:
      • Geodesic integration (photon/particle orbits via RK4)
      • Tidal-force (spaghettification) map
      • Shadow radius
      • Causal-accessibility mask (RUT-PCD): which regions an external
        observer can influence / receive signal from
    """
    def __init__(self, cfg: VacuumBHConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("_dt", torch.tensor(cfg.geodesic_dt), persistent=False)

    def shadow_radius(self, geom: KerrNewmanGeometry) -> Tensor:
        """Apparent shadow radius ≈ √27 M for Schwarzschild (weak spin dep)."""
        return math.sqrt(27.0) * geom.M

    def tidal_force(self, geom: KerrNewmanGeometry, r: Tensor) -> Tensor:
        """Radial tidal acceleration ∝ 2M/r³ (geodesic deviation)."""
        return 2.0 * geom.M / soft_floor(r ** 3, 1e-12)

    def effective_potential(self, geom: KerrNewmanGeometry, r: Tensor,
                            L: Tensor) -> Tensor:
        """Schwarzschild effective potential V_eff = (1-2M/r)(1 + L²/r²)."""
        M = geom.M
        return (1.0 - 2.0 * M / soft_floor(r, 1e-9)) * (1.0 + L * L / (r * r))

    def _geodesic_rk4_step(self, r, vr, L, M, dt):
        """Radial geodesic in Schwarzschild effective potential (fp32)."""
        def acc(rr):
            rr = soft_floor(rr, 1e-6)
            # d²r/dτ² from V_eff gradient (simplified radial equation)
            return (-M / (rr * rr)
                    + L * L / (rr ** 3)
                    - 3.0 * M * L * L / (rr ** 4))
        r32, vr32 = r.float(), vr.float()
        k1v = acc(r32);                k1r = vr32
        k2v = acc(r32 + 0.5*dt*k1r);   k2r = vr32 + 0.5*dt*k1v
        k3v = acc(r32 + 0.5*dt*k2r);   k3r = vr32 + 0.5*dt*k2v
        k4v = acc(r32 + dt*k3r);       k4r = vr32 + dt*k3v
        r_new = r32 + dt/6.0*(k1r + 2*k2r + 2*k3r + k4r)
        vr_new = vr32 + dt/6.0*(k1v + 2*k2v + 2*k3v + k4v)
        return r_new.to(r.dtype), vr_new.to(vr.dtype)

    def integrate_geodesic(self, geom: KerrNewmanGeometry,
                           r0: Tensor, vr0: Tensor, L: Tensor) -> Dict[str, Tensor]:
        r, vr = r0, vr0
        traj = []
        dt = self._dt.to(r.dtype)
        for _ in range(self.cfg.n_geodesic_steps):
            r, vr = self._geodesic_rk4_step(r, vr, L, geom.M, dt)
            traj.append(r)
        traj = torch.stack(traj, dim=-1)            # (B, steps)
        # did the geodesic cross the horizon?
        r_plus = geom.horizons()[0]
        captured = (traj.min(dim=-1).values < r_plus.unsqueeze(-1)).float()
        return {"geodesic_trajectory": traj, "captured": captured,
                "final_radius": traj[..., -1]}

    def causal_accessibility(self, geom: KerrNewmanGeometry,
                             r_probe: Tensor) -> Tensor:
        """
        RUT-PCD accessibility: 1 outside horizon, smooth drop to 0 at r_+.
        Nothing outside can receive signal from inside.
        """
        r_plus = geom.horizons()[0]
        return soft_gate(r_probe - r_plus, tau=1e-2)

    def forward(self, geom: KerrNewmanGeometry, batch: int = 1) -> Dict[str, Tensor]:
        dev = geom.M.device
        r_plus = geom.horizons()[0].expand(batch)
        # probe starting outside the horizon
        r0 = r_plus * 3.0
        vr0 = torch.zeros(batch, device=dev)
        L = torch.ones(batch, device=dev) * 3.5
        geo = self.integrate_geodesic(geom, r0, vr0, L)
        tidal = self.tidal_force(geom, r0)
        access = self.causal_accessibility(geom, r0)
        return {
            "shadow_radius": self.shadow_radius(geom).expand(batch),
            "tidal_force_at_probe": tidal,
            "causal_accessibility": access,
            **geo,
        }


# ═══════════════════════════════════════════════════════════════════════
# [T] SESI INTEGRATION LAYER (Carpet vacuum modes + OPMC + SQCT)
# ═══════════════════════════════════════════════════════════════════════

class SESIVacuumCollapseHead(nn.Module):
    """
    [T2] SQCT: vacuum energy as geometric mass density μ̂_S → collapse rate Γ.
    Links vacuum fluctuations to the black-hole horizon as an interface.
    """
    def __init__(self, cfg: VacuumBHConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.mental_rank
        d = cfg.latent_dim
        self.kappa_I_raw = nn.Parameter(torch.tensor(math.log(math.expm1(0.1))))
        self.mu_bulk = nn.Sequential(LowRankAdapter(d, 32, r), nn.GELU(),
                                     nn.Linear(32, 1), nn.Softplus())
        self.mu_int = nn.Sequential(LowRankAdapter(d, 32, r), nn.GELU(),
                                    nn.Linear(32, 1), nn.Softplus())

    def forward(self, h: Tensor, horizon_area: Tensor) -> Dict[str, Tensor]:
        kappa_I = F.softplus(self.kappa_I_raw) + 1e-6
        mb = self.mu_bulk(h)
        mi = self.mu_int(h)
        mu_S = mb + kappa_I * mi
        # collapse rate grows with interface (horizon) area
        log_Gamma = torch.log(mu_S + 1e-30) + torch.log(horizon_area.unsqueeze(-1) + 1e-30)
        return {"vacuum_mu_S": mu_S.squeeze(-1),
                "vacuum_collapse_rate": torch.exp(log_Gamma.clamp(-60, 40)).squeeze(-1)}


class VacuumOPMCCompressor(nn.Module):
    """[T3] Cloud-OPMC: compress hyper-relational vacuum state → σ_T."""
    def __init__(self, cfg: VacuumBHConfig):
        super().__init__()
        self.cfg = cfg
        d, ds, Fd = cfg.latent_dim, cfg.opmc_sigma_dim, cfg.opmc_n_fields
        self.field_emb = nn.Parameter(torch.randn(Fd, d) * 0.02)
        self.query = LowRankAdapter(d, d, cfg.mental_rank)
        self.value = LowRankAdapter(d, ds, cfg.mental_rank)

    def forward(self, h: Tensor) -> Dict[str, Tensor]:
        q = self.query(h).unsqueeze(1)
        logits = torch.einsum("bnd,fd->bnf", q, self.field_emb)
        logits = logits / math.sqrt(q.shape[-1])
        prov = F.softmax(logits, dim=1)
        v = self.value(h).unsqueeze(1)
        sigma = torch.einsum("bnf,bnd->bfd", prov, v).squeeze(1)
        return {"vacuum_sigma_T": sigma}


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED SYSTEM
# ═══════════════════════════════════════════════════════════════════════

class SESIVacuumBlackHoleSystem(nn.Module):
    """
    Unified vacuum-energy + black-hole exploration system.
    Input: AGI latent (B, latent_dim). Output: full physics survey.
    """
    def __init__(self, cfg: VacuumBHConfig):
        super().__init__()
        self.cfg = cfg
        self.vacuum_field = VacuumEnergyField(cfg)
        self.harvester = DarkEnergyHarvester(cfg)
        self.geometry = KerrNewmanGeometry(cfg)
        self.thermo = BlackHoleThermodynamics(cfg)
        self.extractor = BlackHoleEnergyExtractor(cfg)
        self.explorer = BlackHoleExplorer(cfg)
        self.sqct = SESIVacuumCollapseHead(cfg)
        self.opmc = VacuumOPMCCompressor(cfg)

    def forward(self, h: Tensor) -> Dict[str, Tensor]:
        B = h.shape[0]
        out: Dict[str, Tensor] = {}

        # [V] vacuum sector
        vf = self.vacuum_field(h)
        out.update({f"vac_{k}": v for k, v in vf.items()})
        hv = self.harvester(h, vf)
        out.update({f"harv_{k}": v for k, v in hv.items()})

        # [B] black-hole sector
        geo = self.geometry(batch=B)
        out.update({f"geo_{k}": v for k, v in geo.items()})
        th = self.thermo(self.geometry, batch=B)
        out.update({f"thermo_{k}": v for k, v in th.items()})
        ex = self.extractor(self.geometry, batch=B)
        out.update({f"extr_{k}": v for k, v in ex.items()})
        xp = self.explorer(self.geometry, batch=B)
        out.update({f"expl_{k}": v for k, v in xp.items()})

        # [T] SESI theory layer
        sq = self.sqct(h, geo["horizon_area"])
        out.update({f"sqct_{k}": v for k, v in sq.items()})
        op = self.opmc(h)
        out.update({f"opmc_{k}": v for k, v in op.items()})

        return out


# ═══════════════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════════════

class VacuumBHLoss(nn.Module):
    def __init__(self, cfg: VacuumBHConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, out: Dict[str, Tensor], model: nn.Module) -> Dict[str, Tensor]:
        dev = out["vac_rho_lambda_pred"].device
        zero = torch.zeros((), device=dev)
        L: Dict[str, Tensor] = {}

        # match predicted dark-energy density to observed ρ_Λ (log-space)
        L["rho_lambda"] = F.mse_loss(
            torch.log(out["vac_rho_lambda_pred"] + 1e-30),
            torch.log(out["vac_rho_lambda_obs"] + 1e-30))

        # HARD: penalize any second-law violation (free-energy hallucination)
        L["second_law"] = out["harv_second_law_violation"].pow(2).mean() * 10.0

        # reward real (bounded) extraction channels
        L["extraction"] = -(out["extr_penrose_power"].mean()
                            + out["extr_blandford_znajek_power"].mean()) * 1e-6

        # horizon consistency: r_+ > r_- and M ≥ M_irr (area theorem)
        L["area_theorem"] = F.relu(
            out["geo_irreducible_mass"] - out["geo_M"]).pow(2).mean() * 10.0

        # encourage high causal accessibility for the probe (outside horizon)
        L["accessibility"] = -(out["expl_causal_accessibility"].mean()) * 0.1

        # L2 reg
        reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
        L["reg"] = reg

        total = (L["rho_lambda"] + L["second_law"] + L["extraction"]
                 + L["area_theorem"] + L["accessibility"] + reg * 1e-6)
        L["total"] = total
        return L


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Vacuum Energy & Black Hole Explorer — Smoke Test")
    print("=" * 72)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(42)

    cfg = VacuumBHConfig(latent_dim=64, n_vacuum_modes=16,
                         n_geodesic_steps=50, mental_rank=8)
    model = SESIVacuumBlackHoleSystem(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_par:,}")

    model.train()
    B = 2
    h = torch.randn(B, cfg.latent_dim, device=device)

    out = model(h)
    loss_fn = VacuumBHLoss(cfg)
    losses = loss_fn(out, model)
    losses["total"].backward()

    print(f"\n  ── Vacuum / Dark Energy ──")
    print(f"  ρ_Λ observed       : {out['vac_rho_lambda_obs'][0].item():.3e} J/m³")
    print(f"  ρ_Λ predicted      : {out['vac_rho_lambda_pred'][0].item():.3e} J/m³")
    print(f"  Casimir energy dens: {out['harv_casimir_energy_density'][0].item():.3e}")
    print(f"  Net harvestable    : {out['harv_net_harvestable'][0].item():.3e}")
    print(f"  2nd-law violation  : {out['harv_second_law_violation'][0].item():.3e}")

    print(f"\n  ── Black Hole Geometry (G=c=1) ──")
    print(f"  Mass M             : {out['geo_M'][0].item():.4f}")
    print(f"  Spin a             : {out['geo_a'][0].item():.4f}")
    print(f"  Horizon r_+        : {out['geo_r_plus'][0].item():.4f}")
    print(f"  Inner horizon r_-  : {out['geo_r_minus'][0].item():.4f}")
    print(f"  Ergosphere (eq)    : {out['geo_ergosphere_eq'][0].item():.4f}")
    print(f"  Photon sphere      : {out['geo_photon_sphere'][0].item():.4f}")
    print(f"  Shadow radius      : {out['expl_shadow_radius'][0].item():.4f}")

    print(f"\n  ── Thermodynamics ──")
    print(f"  Hawking temp T_H   : {out['thermo_hawking_temperature'][0].item():.4e}")
    print(f"  Entropy S_BH       : {out['thermo_bh_entropy'][0].item():.4f}")

    print(f"\n  ── Energy Extraction ──")
    print(f"  Penrose η_max      : {out['extr_penrose_efficiency'][0].item():.4f}")
    print(f"  Penrose power      : {out['extr_penrose_power'][0].item():.4e}")
    print(f"  Blandford-Znajek P : {out['extr_blandford_znajek_power'][0].item():.4e}")
    print(f"  Superradiance gain : {out['extr_superradiance_gain'][0].item():.4e}")
    print(f"  Extractable ceiling: {out['extr_extractable_energy_ceiling'][0].item():.4f}")

    print(f"\n  ── Exploration ──")
    print(f"  Geodesic captured  : {out['expl_captured'][0].item():.2f}")
    print(f"  Causal access      : {out['expl_causal_accessibility'][0].item():.4f}")
    print(f"  Tidal force        : {out['expl_tidal_force_at_probe'][0].item():.4e}")

    print(f"\n  ── Losses ──")
    for k, v in losses.items():
        print(f"    {k:<14s}: {v.item():.6f}")

    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if p.requires_grad)
    print(f"\n  Autograd: {'PASS — fully differentiable' if grad_ok else 'FAIL'}")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
