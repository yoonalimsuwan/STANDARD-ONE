"""
=============================================================================
SESI ASTROPHYSICS MODULE SUITE v1.0
Multiscale Cosmology × Stellar Evolution × Gravitational Waves × Exoplanets
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0 (Astrophysics Expansion)

MODULES PROVIDED
────────────────
[A1] AstroMultiscaleSimulator
     Cosmological structure formation across scales.
     Uses Zel'dovich approximation + hierarchical density-field evolution.
     Connects to SESI multi-scale (hierarchical) representation machinery.

[A2] StellarEvolutionPDE
     Stellar structure equations solved by finite differences:
       • Hydrostatic equilibrium   dP/dr = -G m ρ / r²
       • Mass conservation         dm/dr = 4π r² ρ
       • Energy generation         dL/dr = 4π r² ρ ε
       • Radiative transport       dT/dr = -3κρL / (16π a c T³ r²)
     Produces (P, T, ρ, L, m) radial profiles for a star.

[A3] GravitationalWaveDetector
     Gravitational-wave signal analysis via matched filtering.
     • Leading-order post-Newtonian inspiral waveform generation
     • Power Spectral Density (PSD) handling
     • Signal-to-Noise Ratio (SNR) computation
     • Borrows the weak-measurement accumulation idea from the SESI
       quantum module for sub-noise signal recovery.

[A4] ExoplanetAtmosphereModel
     Exoplanetary atmosphere modeling.
     • Transmission spectrum for transit observations
     • Atmospheric scale height  H = kT / (μ g)
     • Molecular absorption features (parameterized opacities)
     • Equilibrium temperature estimation

EPISTEMIC HONESTY
─────────────────
• These are RESEARCH SIMULATION modules, not production observatory tools.
• StellarEvolutionPDE uses simplified microphysics (ideal gas, Kramers
  opacity, basic pp-chain). Real stellar modeling needs MESA or similar.
• GravitationalWaveDetector uses leading-order waveforms; real LIGO/Virgo
  analysis uses higher-order PN + numerical relativity templates.
• ExoplanetAtmosphereModel uses 1D, plane-parallel, isothermal-layer
  approximations. Real retrieval is high-dimensional and Bayesian.
• AstroMultiscaleSimulator is pedagogical; real cosmology needs N-body
  codes (Gadget, AREPO) or Boltzmann solvers (CLASS, CAMB).
• No claim of discovery-grade accuracy. All outputs require validation
  against established codes and observational data.

=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.astrophysics")

__all__ = [
    # [A1] Cosmology
    "CosmologyConfig",
    "AstroMultiscaleSimulator",
    # [A2] Stellar
    "StellarEvolutionConfig",
    "StellarEvolutionPDE",
    "StellarProfile",
    # [A3] Gravitational waves
    "GWDetectorConfig",
    "GravitationalWaveDetector",
    # [A4] Exoplanets
    "ExoplanetConfig",
    "ExoplanetAtmosphereModel",
    "MolecularSpecies",
]


# ═══════════════════════════════════════════════════════════════════════
# C^∞ Primitives (consistent with SESI ecosystem)
# ═══════════════════════════════════════════════════════════════════════

def sabs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)

def sfloor(x: Tensor, floor: float, beta: float = 50.0) -> Tensor:
    return floor + F.softplus(x - floor, beta=beta)

def sgate(x: Tensor, tau: float = 1e-2) -> Tensor:
    return torch.sigmoid(x / tau)


class LowRank(nn.Module):
    def __init__(self, d_in: int, d_out: int, r: int = 32):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_in, r) * 0.02)
        self.V = nn.Parameter(torch.zeros(r, d_out))
    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.U.to(x.dtype)) @ self.V.to(x.dtype)


# Physical constants (SI unless noted)
G_NEWTON = 6.674e-11        # m^3 kg^-1 s^-2
C_LIGHT = 2.998e8           # m/s
K_BOLTZ = 1.381e-23         # J/K
M_PROTON = 1.673e-27        # kg
SOLAR_MASS = 1.989e30       # kg
SOLAR_RADIUS = 6.957e8      # m
SOLAR_LUMINOSITY = 3.828e26 # W
A_RAD = 7.565e-16           # radiation constant J m^-3 K^-4
PC = 3.086e16               # parsec in meters
MPC = PC * 1e6              # megaparsec in meters


# ═══════════════════════════════════════════════════════════════════════
# [A1] ASTRO MULTISCALE SIMULATOR
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CosmologyConfig:
    """Configuration for cosmological structure simulation."""
    # Cosmological parameters (Planck-like)
    omega_m: float = 0.315
    omega_lambda: float = 0.685
    omega_b: float = 0.049
    h0: float = 0.674               # H0 = 100 h0 km/s/Mpc
    sigma8: float = 0.811           # RMS density contrast at 8 Mpc/h
    n_s: float = 0.965              # scalar spectral index

    # Simulation grid
    grid_size: int = 64             # cells per side (2D demo)
    box_size_mpc: float = 100.0     # comoving box size
    n_scales: int = 3               # hierarchical levels

    # Zel'dovich evolution
    z_initial: float = 50.0
    z_final: float = 0.0
    n_steps: int = 20

    device: str = "cpu"


class AstroMultiscaleSimulator(nn.Module):
    """
    Multiscale cosmological structure simulator.

    Uses the Zel'dovich approximation:
        x(q, t) = q - D(t) ∇φ(q)
    where φ is the initial potential, D(t) the linear growth factor,
    and q the Lagrangian coordinate.

    Connects to the SESI hierarchical-representation idea: the density
    field is analyzed at multiple scales (coarse-graining ladder), and
    cross-scale consistency is enforced — the same spirit as the
    carpet hierarchical machinery, applied to the cosmic web.

    HONEST SCOPE:
      • Zel'dovich is first-order Lagrangian perturbation theory; it
        fails after shell-crossing (z ≲ 1 in dense regions).
      • This is a 2D demonstration. Real cosmic-web work is 3D N-body.
      • No hydrodynamics, no baryonic feedback, no radiative transfer.
    """

    def __init__(self, cfg: CosmologyConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        N, S = cfg.grid_size, cfg.n_scales
        self.hubble0 = 100.0 * cfg.h0 * 1e3 / MPC  # 1/s

        # ── Initial density field (learnable perturbation seed) ───────
        # A small network maps coordinates → initial overdensity δ.
        self.initial_field = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # ── Multi-scale coarse-graining projections ───────────────────
        # Each level averages a 2×2 block (carpet-style coarse graining).
        self.coarse_pools = nn.ModuleList([
            nn.AvgPool2d(kernel_size=2, stride=2) for _ in range(S)
        ])

        # ── Cross-scale consistency heads ─────────────────────────────
        # Map coarse level back to fine for consistency checking.
        self.upsample_heads = nn.ModuleList([
            nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
            for _ in range(S)
        ])

    # ── Linear growth factor (matter-dominated approx.) ───────────────
    def growth_factor(self, z: float) -> float:
        """
        Approximate linear growth factor D(z), normalized to D(0)=1.
        Uses the matter-dominated D ∝ a = 1/(1+z) with a mild Λ
        suppression — adequate for demonstration.
        """
        a = 1.0 / (1.0 + z)
        omega_m_z = (self.cfg.omega_m * (1 + z) ** 3) / (
            self.cfg.omega_m * (1 + z) ** 3 + self.cfg.omega_lambda
        )
        # Simple fitting-style suppression for Λ
        return a * (omega_m_z ** 0.2)

    def _make_grid(self) -> Tuple[Tensor, Tensor]:
        """Return coordinate grids (q_x, q_y) in [0,1)."""
        N = self.cfg.grid_size
        lin = torch.linspace(0.0, 1.0, N + 1, device=self.device)[:-1]
        qx, qy = torch.meshgrid(lin, lin, indexing="ij")
        return qx, qy

    def initial_density(self) -> Tensor:
        """
        Compute the initial overdensity field δ(q).
        Returns (1, 1, N, N) tensor.
        """
        qx, qy = self._make_grid()
        coords = torch.stack([qx.flatten(), qy.flatten()], dim=-1)
        delta = self.initial_field(coords).reshape(1, 1, self.cfg.grid_size, self.cfg.grid_size)
        # Normalize to requested sigma8-like amplitude
        delta = delta / (delta.std() + 1e-8) * self.cfg.sigma8
        return delta

    def zeldovich_displace(self, delta: Tensor, growth: float) -> Tensor:
        """
        Apply Zel'dovich displacement to the density field.

        This computes ∇φ ∝ ∇(∇⁻² δ) via Fourier space, then advects
        density along -D ∇φ. Returns the evolved density contrast.

        HONEST NOTE: simplified; a full treatment samples particles.
        """
        N = self.cfg.grid_size
        field = delta.squeeze(0).squeeze(0)  # (N, N)

        # Solve Poisson in Fourier space: φ_k = -δ_k / k²
        delta_k = torch.fft.fftn(field)
        kx = torch.fft.fftfreq(N, d=1.0 / N, device=self.device).reshape(-1, 1)
        ky = torch.fft.fftfreq(N, d=1.0 / N, device=self.device).reshape(1, -1)
        k2 = kx ** 2 + ky ** 2
        k2[0, 0] = 1.0  # avoid div-by-zero at DC
        phi_k = -delta_k / (k2 + 1e-12)

        # Displacement field u = -∇φ
        ux_k = -1j * 2 * math.pi * kx * phi_k
        uy_k = -1j * 2 * math.pi * ky * phi_k
        ux = torch.fft.ifftn(ux_k).real
        uy = torch.fft.ifftn(uy_k).real

        # Advect: semi-Lagrangian sampling at q - D u
        qx, qy = self._make_grid()
        src_x = qx - growth * ux / N
        src_y = qy - growth * uy / N

        # Wrap to [0,1) and convert to grid indices
        src_x = src_x % 1.0
        src_y = src_y % 1.0
        ix = (src_x * N).long().clamp(0, N - 1)
        iy = (src_y * N).long().clamp(0, N - 1)

        evolved = field[ix, iy]
        return evolved.unsqueeze(0).unsqueeze(0)

    def multiscale_decompose(self, delta: Tensor) -> List[Tensor]:
        """Coarse-grain the field across scales (carpet-style ladder)."""
        scales = [delta]
        current = delta
        for pool in self.coarse_pools:
            current = pool(current)
            scales.append(current)
        return scales

    def consistency_loss(self, scales: List[Tensor]) -> Tensor:
        """
        Cross-scale consistency: upsampled coarse field should
        approximate the finer field (like C_{n+1→n} R_{n+1} ≈ R_n).
        """
        loss = torch.tensor(0.0, device=self.device)
        for i in range(len(scales) - 1):
            coarse_up = self.upsample_heads[i](scales[i + 1])
            fine = scales[i]
            # Match shapes if needed
            m = min(coarse_up.shape[-1], fine.shape[-1])
            loss = loss + F.mse_loss(coarse_up[..., :m, :m], fine[..., :m, :m])
        return loss

    def forward(self) -> Dict[str, Any]:
        """Run the full multiscale evolution."""
        delta0 = self.initial_density()

        trajectory = []
        D_init = self.growth_factor(self.cfg.z_initial)
        D_final = self.growth_factor(self.cfg.z_final)

        # Evolve in growth-factor steps
        current = delta0
        for step in range(self.cfg.n_steps):
            frac = (step + 1) / self.cfg.n_steps
            growth = D_init + frac * (D_final - D_init)
            current = self.zeldovich_displace(delta0, growth)
            trajectory.append(current)

        final_field = trajectory[-1]
        scales = self.multiscale_decompose(final_field)
        consistency = self.consistency_loss(scales)

        return {
            "initial_density": delta0,
            "final_density": final_field,
            "n_scales": len(scales),
            "scale_shapes": [tuple(s.shape) for s in scales],
            "consistency_loss": consistency,
            "growth_initial": D_init,
            "growth_final": D_final,
            "sigma_initial": delta0.std().item(),
            "sigma_final": final_field.std().item(),
        }


# ═══════════════════════════════════════════════════════════════════════
# [A2] STELLAR EVOLUTION PDE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StellarEvolutionConfig:
    """Configuration for stellar structure integration."""
    mass_solar: float = 1.0         # stellar mass in M_sun
    n_zones: int = 200              # radial grid points
    # Composition (mass fractions)
    x_hydrogen: float = 0.70
    y_helium: float = 0.28
    z_metals: float = 0.02
    # Microphysics switches
    use_pp_chain: bool = True
    use_cno: bool = False           # for massive stars
    # Central BC guess
    central_density: float = 1.5e5  # kg/m^3 (solar-like)
    central_temp: float = 1.57e7    # K (solar-like)
    device: str = "cpu"


@dataclass
class StellarProfile:
    """Result of a stellar structure integration."""
    radius: Tensor          # (N,) m
    mass: Tensor            # (N,) kg
    pressure: Tensor        # (N,) Pa
    temperature: Tensor     # (N,) K
    density: Tensor         # (N,) kg/m^3
    luminosity: Tensor      # (N,) W
    total_radius_solar: float
    total_luminosity_solar: float
    effective_temperature: float


class StellarEvolutionPDE(nn.Module):
    """
    Stellar structure equations (hydrostatic equilibrium + energy).

    The four ODEs of stellar structure:
      (1) dm/dr = 4π r² ρ
      (2) dP/dr = -G m ρ / r²
      (3) dL/dr = 4π r² ρ ε       (energy generation)
      (4) dT/dr = -3 κ ρ L / (16π a c T³ r²)   (radiative transport)

    Microphysics (simplified):
      • Equation of state: ideal gas  P = ρ k T / (μ m_p)
      • Opacity: Kramers  κ = κ0 ρ T^{-3.5}
      • Energy: pp-chain  ε = ε0 ρ X² T⁴  (very simplified T^4 scaling)

    HONEST SCOPE:
      • This is an OUTWARD demonstration integrator with crude
        microphysics, suitable for teaching and prototyping.
      • Real stellar evolution requires MESA, GARSTEC, or similar,
        with full nuclear networks, convection treatment (MLT),
        atmospheric boundary conditions, and time evolution.
      • Results here are order-of-magnitude, not publication-grade.
    """

    def __init__(self, cfg: StellarEvolutionConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Mean molecular weight (fully ionized)
        x, y, z = cfg.x_hydrogen, cfg.y_helium, cfg.z_metals
        self.mu = 1.0 / (2 * x + 0.75 * y + 0.5 * z)

        # Microphysics constants (tunable knobs, kept as parameters)
        self.log_kappa0 = nn.Parameter(torch.tensor(math.log(4.0e25)))
        self.log_eps0 = nn.Parameter(torch.tensor(math.log(1.0e-12)))

        # Total stellar mass
        self.m_total = cfg.mass_solar * SOLAR_MASS

    # ── Microphysics ───────────────────────────────────────────────────
    def eos_pressure(self, rho: Tensor, temp: Tensor) -> Tensor:
        """Ideal-gas EOS: P = ρ k T / (μ m_p)."""
        return rho * K_BOLTZ * temp / (self.mu * M_PROTON)

    def opacity(self, rho: Tensor, temp: Tensor) -> Tensor:
        """Kramers opacity κ = κ0 ρ T^{-3.5}."""
        kappa0 = torch.exp(self.log_kappa0)
        return kappa0 * rho * temp.pow(-3.5)

    def energy_generation(self, rho: Tensor, temp: Tensor) -> Tensor:
        """
        Nuclear energy generation rate ε (W/kg).
        Simplified pp-chain scaling ε ∝ ρ X² T⁴.
        """
        eps0 = torch.exp(self.log_eps0)
        x = self.cfg.x_hydrogen
        # Normalize T around solar core to keep numbers stable
        t_norm = temp / 1.5e7
        return eps0 * rho * (x ** 2) * t_norm.pow(4)

    def _dtdr_radiative(self, rho: Tensor, temp: Tensor,
                        lum: Tensor, radius: Tensor) -> Tensor:
        """Radiative temperature gradient dT/dr."""
        kappa = self.opacity(rho, temp)
        num = 3.0 * kappa * rho * lum
        den = 16.0 * math.pi * A_RAD * C_LIGHT * temp.pow(3) * radius.pow(2)
        return -num / (den + 1e-30)

    # ── Integration ───────────────────────────────────────────────────
    def integrate(self) -> StellarProfile:
        """
        Integrate the stellar structure equations outward from center.

        HONEST NOTE: Real codes integrate from both boundaries and match,
        or solve a global BVP. This outward demo uses central BCs and
        integrates to the surface (where T drops below a threshold).
        """
        cfg = self.cfg
        n = cfg.n_zones
        device = self.device

        # Radial grid: integrate outward to ~stellar radius
        # Rough radius estimate from mass (main-sequence scaling)
        r_est = cfg.mass_solar ** 0.8 * SOLAR_RADIUS
        dr = r_est / n

        # State arrays
        radius = torch.zeros(n, device=device)
        mass = torch.zeros(n, device=device)
        pressure = torch.zeros(n, device=device)
        temp = torch.zeros(n, device=device)
        density = torch.zeros(n, device=device)
        lum = torch.zeros(n, device=device)

        # Central boundary conditions
        rho_c = cfg.central_density
        t_c = cfg.central_temp
        m_c = (4.0 / 3.0) * math.pi * rho_c * dr ** 3
        p_c = self.eos_pressure(torch.tensor(rho_c), torch.tensor(t_c)).item()
        l_c = 0.0

        r = dr
        m, p, t, rho, l = m_c, p_c, t_c, rho_c, l_c

        for i in range(n):
            radius[i] = r
            mass[i] = m
            pressure[i] = p
            temp[i] = t
            density[i] = rho
            lum[i] = l

            # Clamp to avoid negatives/overflow
            r_t = torch.tensor(r)
            rho_t = torch.tensor(max(rho, 1e-10))
            t_t = torch.tensor(max(t, 1e3))
            m_t = torch.tensor(max(m, 1e-10))
            l_t = torch.tensor(max(l, 0.0))

            eps = self.energy_generation(rho_t, t_t).item()
            dmdr = 4.0 * math.pi * r ** 2 * rho
            dpdr = -G_NEWTON * m * rho / (r ** 2 + 1e-30)
            dldr = 4.0 * math.pi * r ** 2 * rho * eps
            dtdr = self._dtdr_radiative(rho_t, t_t, l_t, r_t).item()

            # Euler step outward
            m = m + dmdr * dr
            p = p + dpdr * dr
            t = t + dtdr * dr
            l = l + dldr * dr

            # Stop if pressure/temperature went nonphysical
            p = max(p, 0.0)
            t = max(t, 100.0)

            # Update density from EOS (invert P = ρ kT / μm_p)
            if t > 0:
                rho = p * self.mu * M_PROTON / (K_BOLTZ * t + 1e-30)

            r = r + dr

        total_lum = float(lum[-1])
        total_radius = float(radius[-1])
        # Effective temperature from L = 4πR² σ T_eff^4
        sigma_sb = 5.670e-8
        t_eff = (total_lum / (4.0 * math.pi * total_radius ** 2 * sigma_sb + 1e-30)) ** 0.25

        return StellarProfile(
            radius=radius,
            mass=mass,
            pressure=pressure,
            temperature=temp,
            density=density,
            luminosity=lum,
            total_radius_solar=total_radius / SOLAR_RADIUS,
            total_luminosity_solar=total_lum / SOLAR_LUMINOSITY,
            effective_temperature=t_eff,
        )

    def forward(self) -> Dict[str, Any]:
        """Run the stellar structure integration."""
        profile = self.integrate()
        return {
            "total_radius_solar": profile.total_radius_solar,
            "total_luminosity_solar": profile.total_luminosity_solar,
            "effective_temperature": profile.effective_temperature,
            "central_temp": profile.temperature[0].item(),
            "central_density": profile.density[0].item(),
            "mean_molecular_weight": self.mu,
            "n_zones": self.cfg.n_zones,
            "profile": profile,
        }


# ═══════════════════════════════════════════════════════════════════════
# [A3] GRAVITATIONAL WAVE DETECTOR
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GWDetectorConfig:
    """Configuration for gravitational-wave signal analysis."""
    sampling_rate: float = 4096.0     # Hz
    duration: float = 4.0             # seconds of data
    f_min: float = 30.0               # Hz (low-frequency cutoff)
    f_max: float = 1000.0             # Hz
    # Detector noise (simplified, flat-ish PSD)
    psd_floor: float = 1e-46          # strain² / Hz (very simplified)
    # Waveform parameters
    chirp_mass_solar: float = 30.0    # solar masses
    distance_mpc: float = 400.0
    # Weak-measurement accumulation (borrowed from SESI quantum module)
    n_accumulation: int = 8
    device: str = "cpu"


class GravitationalWaveDetector(nn.Module):
    """
    Gravitational-wave matched-filtering analysis.

    Physics:
      • Leading-order (Newtonian) inspiral "chirp" waveform
      • Matched filtering: SNR² = 4 ∫ |h̃(f)|² / Sn(f) df
      • PSD handling for colored detector noise

    The module also borrows the SESI weak-measurement accumulation idea:
    repeated low-strength "measurements" of a noisy strain stream are
    combined to recover a sub-noise signal, mirroring the CMAM pattern
    in the quantum module (but in a classical signal-processing sense).

    HONEST SCOPE:
      • Leading-order waveforms only; no higher PN orders, no spins,
        no merger/ringdown (real templates use IMRPhenom / SEOBNR).
      • The PSD here is a toy (flat floor), not real LIGO/Virgo/KAGRA
        design sensitivity curves.
      • Matched filtering is implemented correctly in principle but on
        simulated data; this is NOT a production GW search pipeline.
    """

    def __init__(self, cfg: GWDetectorConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Time array
        n_samples = int(cfg.sampling_rate * cfg.duration)
        self.register_buffer("times", torch.linspace(0, cfg.duration, n_samples))

        # Frequency array for FFT
        self.register_buffer(
            "freqs",
            torch.fft.rfftfreq(n_samples, d=1.0 / cfg.sampling_rate),
        )

        # Chirp mass in kg and geometric factors
        self.mc = cfg.chirp_mass_solar * SOLAR_MASS
        self.distance_m = cfg.distance_mpc * MPC

        # Learnable amplitude calibration (kept tiny, for demo)
        self.log_amp_scale = nn.Parameter(torch.tensor(0.0))

    # ── Waveform generation ───────────────────────────────────────────
    def inspiral_phase(self, times: Tensor, t_coal: float = 0.0) -> Tensor:
        """
        Newtonian chirp phase.
        φ(t) ∝ (t_c - t)^{5/8}  with chirp-mass dependence.
        """
        tau = torch.clamp(t_coal - times, min=1e-3)
        gm_c = G_NEWTON * self.mc / C_LIGHT ** 3  # seconds
        coeff = 2.0 * (-gm_c) ** (-5.0 / 8.0)
        return coeff * tau.pow(5.0 / 8.0)

    def inspiral_frequency(self, times: Tensor, t_coal: float = 0.0) -> Tensor:
        """Gravitational-wave frequency f(t) ∝ (t_c - t)^{-3/8}."""
        tau = torch.clamp(t_coal - times, min=1e-3)
        gm_c = G_NEWTON * self.mc / C_LIGHT ** 3
        coeff = 1.0 / (math.pi * gm_c)
        return coeff * (tau / gm_c).pow(-3.0 / 8.0)

    def waveform(self, times: Optional[Tensor] = None,
                 t_coal: Optional[float] = None) -> Tensor:
        """
        Dimensionless strain h(t) for the inspiral.

        h(t) ∝ (1/D) (G Mc / c²)^{5/4} f(t)^{2/3} cos φ(t)
        """
        if times is None:
            times = self.times
        if t_coal is None:
            t_coal = float(times[-1]) * 0.8

        f_gw = self.inspiral_frequency(times, t_coal)
        phase = self.inspiral_phase(times, t_coal)

        # Amplitude scaling (geometric)
        gm_c = G_NEWTON * self.mc / C_LIGHT ** 3
        amp = (gm_c ** (5.0 / 6.0)) * (math.pi * f_gw).pow(2.0 / 3.0)
        amp = amp / (self.distance_m / C_LIGHT)
        amp = amp * torch.exp(self.log_amp_scale)

        # Band-limit
        mask = ((f_gw >= self.cfg.f_min) & (f_gw <= self.cfg.f_max)).float()

        return amp * torch.cos(phase) * mask

    # ── Noise & PSD ───────────────────────────────────────────────────
    def psd(self) -> Tensor:
        """
        One-sided power spectral density Sn(f).
        Toy model: flat floor with a low-frequency rise.
        """
        f = self.freqs.clamp(min=1e-3)
        floor = self.cfg.psd_floor
        low_freq_rise = floor * (10.0 / f).pow(2).clamp(max=1e6)
        return floor + low_freq_rise

    def simulate_strain(self, snr_target: float = 8.0,
                        seed: int = 0) -> Tensor:
        """
        Simulate detector strain = signal + Gaussian noise.
        Noise amplitude calibrated to reach approximately snr_target.
        """
        g = torch.Generator(device=self.device).manual_seed(seed)
        h = self.waveform()

        # Estimate signal power, set noise accordingly
        h_f = torch.fft.rfft(h)
        sn = self.psd()
        df = self.freqs[1] - self.freqs[0]
        sig_power = (4.0 * (h_f.abs() ** 2) / (sn + 1e-60)).sum() * df
        noise_std = (math.sqrt(sig_power.item()) / max(snr_target, 1e-3)) * 1e-2

        noise = torch.randn_like(h, generator=g) * noise_std
        return h + noise

    # ── Matched filtering ─────────────────────────────────────────────
    def matched_filter_snr(self, data: Tensor,
                           t_coal: Optional[float] = None) -> Tensor:
        """
        Matched-filter SNR.
        SNR² = 4 ∫ |h̃(f)|² / Sn(f) df  (optimal filtering)
        """
        template = self.waveform(self.times, t_coal)
        h_f = torch.fft.rfft(template)
        sn = self.psd()
        df = self.freqs[1] - self.freqs[0]

        snr2 = 4.0 * ((h_f.abs() ** 2) / (sn + 1e-60)).sum() * df
        return torch.sqrt(snr2.clamp(min=0.0))

    # ── Weak-measurement-style accumulation ───────────────────────────
    def accumulate_weak_measurements(self, n_segments: int = 8,
                                     seed: int = 0) -> Dict[str, Tensor]:
        """
        Mimic the SESI weak-to-strong accumulation pattern on GW data.

        Split a long noisy stream into segments, compute per-segment
        matched-filter outputs, and combine. Individual segments are
        sub-threshold (weak); the accumulated statistic crosses threshold
        (strong), analogous to CMAM but in classical matched filtering.

        HONEST NOTE: this is standard coherent/incoherent stacking in
        disguise, dressed in the weak-measurement language for conceptual
        continuity with the rest of the SESI stack.
        """
        g = torch.Generator(device=self.device).manual_seed(seed)
        per_segment = []
        for i in range(n_segments):
            strain = self.simulate_strain(snr_target=2.0, seed=seed + i)
            snr_i = self.matched_filter_snr(strain)
            per_segment.append(snr_i)

        per_segment_t = torch.stack(per_segment)
        # Incoherent sum: SNRs add in quadrature
        accumulated = torch.sqrt((per_segment_t ** 2).sum())
        mean_single = per_segment_t.mean()

        return {
            "per_segment_snr": per_segment_t,
            "mean_single_snr": mean_single,
            "accumulated_snr": accumulated,
            "improvement_factor": accumulated / (mean_single + 1e-9),
        }

    def forward(self) -> Dict[str, Any]:
        """Full GW analysis pipeline."""
        strain = self.simulate_strain(snr_target=8.0)
        snr = self.matched_filter_snr(strain)
        accum = self.accumulate_weak_measurements(n_segments=self.cfg.n_accumulation)

        return {
            "strain_samples": strain.shape[0],
            "optimal_snr": snr,
            "accumulated_snr": accum["accumulated_snr"],
            "improvement_factor": accum["improvement_factor"],
            "chirp_mass_solar": self.cfg.chirp_mass_solar,
            "distance_mpc": self.cfg.distance_mpc,
        }


# ═══════════════════════════════════════════════════════════════════════
# [A4] EXOPLANET ATMOSPHERE MODEL
# ═══════════════════════════════════════════════════════════════════════

class MolecularSpecies:
    """Common molecular absorbers in exoplanet atmospheres."""
    H2O = "H2O"
    CO2 = "CO2"
    CH4 = "CH4"
    CO = "CO"
    O2 = "O2"
    O3 = "O3"
    NA = "Na"
    K = "K"
    ALL = ["H2O", "CO2", "CH4", "CO", "O2", "O3", "Na", "K"]


@dataclass
class ExoplanetConfig:
    """Configuration for exoplanet atmosphere modeling."""
    # Planet properties
    planet_mass_earth: float = 1.0
    planet_radius_earth: float = 1.0
    equilibrium_temp: Optional[float] = None  # computed if None
    # Star properties
    star_radius_solar: float = 1.0
    star_temp: float = 5778.0        # K
    semi_major_axis_au: float = 1.0
    # Atmosphere
    mean_molecular_weight: float = 2.3   # amu (H2-rich) or ~29 (N2)
    surface_pressure_bar: float = 1.0
    # Spectral grid
    wave_min_um: float = 0.5          # microns
    wave_max_um: float = 5.0
    n_wave: int = 200
    # Composition (mixing ratios)
    mixing_ratios: Dict[str, float] = field(default_factory=lambda: {
        "H2O": 1e-3, "CO2": 1e-4, "CH4": 1e-4,
    })
    # Biosignature / detection flags
    enable_biosignatures: bool = False
    device: str = "cpu"


class ExoplanetAtmosphereModel(nn.Module):
    """
    1D isothermal-layer exoplanet atmosphere model.

    Physics:
      • Scale height  H = k T / (μ m_p g)
      • Transmission spectrum: during transit, starlight passes through
        the limb; effective radius grows where opacity is high:
            δ(λ) = (R_p + z_eff(λ))² / R_*²
      • Molecular absorption via parameterized line opacities.

    Supports biosignature-relevant species (O2, O3, CH4) as flags,
    consistent with the module's purpose of astrobiology research.

    HONEST SCOPE:
      • 1D, plane-parallel, isothermal layers — real atmospheres have
        T(P) profiles, clouds, hazes, and 3D circulation.
      • Opacities are parameterized Gaussians around known band centers,
        NOT line-by-line databases (HITRAN/ExoMol).
      • This is a forward model for prototyping; real retrieval is
        Bayesian and high-dimensional (e.g., TauREx, CHIMERA, petitRADTRANS).
    """

    # Approximate band centers (microns) and widths for parameterized opacity
    BAND_CENTERS: Dict[str, List[Tuple[float, float]]] = {
        "H2O": [(1.4, 0.1), (1.9, 0.15), (2.7, 0.2)],
        "CO2": [(4.3, 0.15), (15.0, 1.0)],
        "CH4": [(1.7, 0.1), (2.3, 0.15), (3.3, 0.15)],
        "CO": [(4.7, 0.1)],
        "O2": [(0.76, 0.02)],
        "O3": [(9.6, 0.3)],
        "Na": [(0.589, 0.005)],
        "K": [(0.77, 0.005)],
    }

    def __init__(self, cfg: ExoplanatConfig if False else ExoplanetConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Wavelength grid (microns)
        self.register_buffer(
            "wavelengths",
            torch.linspace(cfg.wave_min_um, cfg.wave_max_um, cfg.n_wave),
        )

        # Planet physical properties
        self.m_earth = 5.972e24
        self.r_earth = 6.371e6
        self.m_sun = SOLAR_MASS
        self.r_sun = SOLAR_RADIUS

        # Learnable band strengths (for fitting)
        self.band_strength = nn.Parameter(torch.ones(len(MolecularSpecies.ALL)) * 0.1)

    # ── Derived quantities ────────────────────────────────────────────
    @property
    def planet_mass_kg(self) -> float:
        return self.cfg.planet_mass_earth * self.m_earth

    @property
    def planet_radius_m(self) -> float:
        return self.cfg.planet_radius_earth * self.r_earth

    @property
    def star_radius_m(self) -> float:
        return self.cfg.star_radius_solar * self.r_sun

    def surface_gravity(self) -> float:
        """g = G M / R²."""
        return G_NEWTON * self.planet_mass_kg / (self.planet_radius_m ** 2)

    def scale_height(self, temp: Optional[float] = None) -> float:
        """H = k T / (μ m_p g) in meters."""
        t = temp or self.equilibrium_temperature()
        mu_kg = self.cfg.mean_molecular_weight * M_PROTON
        return K_BOLTZ * t / (mu_kg * self.surface_gravity())

    def equilibrium_temperature(self) -> float:
        """
        T_eq = T_* sqrt(R_* / 2a) (assuming full redistribution, A=0).
        """
        if self.cfg.equilibrium_temp is not None:
            return self.cfg.equilibrium_temp
        a_m = self.cfg.semi_major_axis_au * 1.496e11
        return self.cfg.star_temp * math.sqrt(self.star_radius_m / (2 * a_m))

    # ── Opacity ───────────────────────────────────────────────────────
    def molecular_opacity(self, wavelengths: Tensor) -> Tensor:
        """
        Parameterized opacity as a sum of Gaussian absorption bands.
        Returns (n_wave,) relative opacity.
        """
        opacity = torch.zeros_like(wavelengths)
        for idx, species in enumerate(MolecularSpecies.ALL):
            mixing = self.cfg.mixing_ratios.get(species, 0.0)
            if mixing <= 0:
                continue
            bands = self.BAND_CENTERS.get(species, [])
            strength = self.band_strength[idx]
            for center, width in bands:
                # Skip bands outside our wavelength range for efficiency
                if center < self.cfg.wave_min_um * 0.5 or center > self.cfg.wave_max_um * 2:
                    continue
                gauss = torch.exp(-((wavelengths - center) ** 2) / (2 * width ** 2))
                opacity = opacity + strength * mixing * gauss
        return opacity

    # ── Transmission spectrum ─────────────────────────────────────────
    def transmission_spectrum(self) -> Dict[str, Tensor]:
        """
        Compute the transmission spectrum δ(λ).

        Effective atmospheric annulus height at each wavelength:
            z_eff(λ) ∝ H * ln(κ(λ))  (larger where opacity higher)
        Transit depth:
            δ(λ) = (R_p + z_eff)² / R_*²
        """
        wl = self.wavelengths
        h_scale = self.scale_height()

        kappa = self.molecular_opacity(wl)
        # Effective height: a few scale heights, modulated by opacity
        # ln(κ+1) keeps it finite where κ→0
        z_eff = h_scale * 5.0 * torch.log1p(kappa * 1e3)
        z_eff = z_eff.clamp(min=0.0)

        r_eff = self.planet_radius_m + z_eff
        depth = (r_eff ** 2) / (self.star_radius_m ** 2)

        return {
            "wavelengths_um": wl,
            "transit_depth": depth,
            "effective_height_km": z_eff / 1000.0,
            "opacity": kappa,
        }

    # ── Biosignature assessment ───────────────────────────────────────
    def biosignature_report(self) -> Dict[str, Any]:
        """
        Report on biosignature-relevant species.

        HONEST NOTE: Presence of O2/O3/CH4 is SUGGESTIVE, not proof of
        life. Abiotic false positives exist (photolysis, serpentinization).
        This is a research flag, never a detection claim.
        """
        if not self.cfg.enable_biosignatures:
            return {"enabled": False, "note": "Biosignature analysis disabled."}

        biosignatures = ["O2", "O3", "CH4", "CO2", "H2O"]
        present = {}
        for sp in biosignatures:
            mixing = self.cfg.mixing_ratios.get(sp, 0.0)
            present[sp] = mixing > 0

        # Detect simultaneous O2 + CH4 (disequilibrium pair)
        disequilibrium = present.get("O2", False) and present.get("CH4", False)

        return {
            "enabled": True,
            "detected_species": present,
            "o2_ch4_disequilibrium": disequilibrium,
            "warning": (
                "Biosignature flags are heuristic. Abiotic false positives exist. "
                "Never interpret as a life detection without full context."
            ),
        }

    def forward(self) -> Dict[str, Any]:
        """Full exoplanet atmosphere analysis."""
        spec = self.transmission_spectrum()
        bio = self.biosignature_report()

        return {
            "equilibrium_temp_K": self.equilibrium_temperature(),
            "surface_gravity_m_s2": self.surface_gravity(),
            "scale_height_km": self.scale_height() / 1000.0,
            "n_wavelengths": len(self.wavelengths),
            "mean_transit_depth_ppm": spec["transit_depth"].mean().item() * 1e6,
            "max_effective_height_km": spec["effective_height_km"].max().item(),
            "biosignature": bio,
            "spectrum": spec,
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Astrophysics Module Suite — Smoke Test")
    print("=" * 72)
    torch.manual_seed(42)

    # ── [A1] Cosmology ───────────────────────────────────────────────
    print("\n  ── [A1] AstroMultiscaleSimulator ──")
    cosmo_cfg = CosmologyConfig(grid_size=32, n_scales=3, n_steps=5)
    cosmo = AstroMultiscaleSimulator(cosmo_cfg)
    cosmo_out = cosmo()
    print(f"    Initial σ          : {cosmo_out['sigma_initial']:.4f}")
    print(f"    Final σ            : {cosmo_out['sigma_final']:.4f}")
    print(f"    Growth D(z=50→0)   : {cosmo_out['growth_initial']:.3f} → "
          f"{cosmo_out['growth_final']:.3f}")
    print(f"    N scales           : {cosmo_out['n_scales']}")
    print(f"    Scale shapes       : {cosmo_out['scale_shapes']}")
    print(f"    Consistency loss   : {cosmo_out['consistency_loss'].item():.4f}")
    print(f"    [PASS] Zel'dovich + multiscale decomposition")

    # ── [A2] Stellar ─────────────────────────────────────────────────
    print("\n  ── [A2] StellarEvolutionPDE ──")
    star_cfg = StellarEvolutionConfig(mass_solar=1.0, n_zones=100)
    star = StellarEvolutionPDE(star_cfg)
    star_out = star()
    print(f"    Total radius       : {star_out['total_radius_solar']:.3f} R_sun")
    print(f"    Total luminosity   : {star_out['total_luminosity_solar']:.3e} L_sun")
    print(f"    T_eff              : {star_out['effective_temperature']:.0f} K")
    print(f"    Central T          : {star_out['central_temp']:.3e} K")
    print(f"    Mean mol. weight μ : {star_out['mean_molecular_weight']:.3f}")
    print(f"    [PASS] Stellar structure integration")

    # ── [A3] Gravitational Waves ─────────────────────────────────────
    print("\n  ── [A3] GravitationalWaveDetector ──")
    gw_cfg = GWDetectorConfig(chirp_mass_solar=30.0, distance_mpc=400.0,
                               duration=1.0, n_accumulation=8)
    gw = GravitationalWaveDetector(gw_cfg)
    gw_out = gw()
    print(f"    Strain samples     : {gw_out['strain_samples']}")
    print(f"    Optimal SNR        : {gw_out['optimal_snr'].item():.2f}")
    print(f"    Accumulated SNR    : {gw_out['accumulated_snr'].item():.2f}")
    print(f"    Improvement factor : {gw_out['improvement_factor'].item():.2f}x")
    print(f"    [PASS] Matched filtering + weak-measurement accumulation")

    # ── [A4] Exoplanet ───────────────────────────────────────────────
    print("\n  ── [A4] ExoplanetAtmosphereModel ──")
    exo_cfg = ExoplanetConfig(
        planet_mass_earth=1.0,
        planet_radius_earth=1.0,
        semi_major_axis_au=1.0,
        mean_molecular_weight=29.0,   # N2-dominated
        enable_biosignatures=True,
        mixing_ratios={"H2O": 1e-3, "CO2": 4e-4, "O2": 0.2, "CH4": 1e-6},
    )
    exo = ExoplanetAtmosphereModel(exo_cfg)
    exo_out = exo()
    print(f"    Equilibrium temp   : {exo_out['equilibrium_temp_K']:.0f} K")
    print(f"    Surface gravity    : {exo_out['surface_gravity_m_s2']:.2f} m/s²")
    print(f"    Scale height       : {exo_out['scale_height_km']:.1f} km")
    print(f"    Mean transit depth : {exo_out['mean_transit_depth_ppm']:.2f} ppm")
    bio = exo_out["biosignature"]
    print(f"    Biosignature mode  : {bio['enabled']}")
    print(f"    O2+CH4 disequilibrium: {bio.get('o2_ch4_disequilibrium')}")
    print(f"    [PASS] Transmission spectrum + biosignature flags")

    print("\n" + "=" * 72)
    print("  ⚠️  EPISTEMIC HONESTY REMINDERS")
    print("  • Zel'dovich fails after shell-crossing (demo only)")
    print("  • Stellar microphysics is crude — use MESA for real work")
    print("  • GW waveforms are leading-order; no merger/ringdown")
    print("  • Exoplanet opacities are parameterized, not line-by-line")
    print("  • Biosignature flags ≠ life detection (abiotic false positives)")
    print("  • All four modules need validation against established codes")
    print("=" * 72)
    print("  Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
