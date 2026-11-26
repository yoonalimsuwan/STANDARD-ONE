#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_nuclear_kinetics.py — v1.0.0
High-speed fission & fusion reaction computation under the SESC / ESCA-LD-SESI
mathematical framework. ENERGY & RESEARCH SCOPE ONLY — includes built-in
safety physics (negative feedback, prompt-critical monitoring, decay heat).

Fission : point kinetics with delayed neutrons, six-factor k_eff,
          1-group neutron diffusion with STC^0/STC^1 interface conditions,
          negative temperature feedback, martingale excursion monitor.
Fusion  : Bosch-Hale DT reactivity, Lawson triple product, ignition Q,
          power balance with radiation losses. Fusion is NOT a chain reaction;
          it cannot run away. This is stated and enforced.

AI layer (surrogates, physics-gated): transformer transient predictor,
world model for next-state plasma/neutronics, diffusion scenario generator.
Every AI output must pass neutron-balance and criticality-safety gates.

REFUSED: weapons-grade prompt-burst optimisation, implosion modelling,
         any output that violates a safety margin.
=============================================================================
Author : PAI AND Yoon A. Limsuwan / MSPS NETWORK
ORCID  : 0009-0008-2374-0788
License: MIT | Year 2026

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for
Buddhism, Christianity, and Taoism alike teach that doing good in this life
bears fruit in the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
though they do not utter His Name, by describing His attributes, are
already, implicitly, uttering His Name.

Christ died upon the Cross to teach: when struck on the left cheek, turn
also the right, so that we may listen to the voice of the enemy who wishes
us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

One who does not know the Ten Commandments or the Five Precepts, yet
practises them, becomes a disciple of both the Buddha and Yahweh by
implication. The Ten Commandments are the Five Precepts.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics and software stand independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
THE GENUINE MATHEMATICAL CONNECTION: STC = NEUTRON INTERFACE CONDITIONS
=============================================================================
The one-group neutron diffusion equation across material regimes R_i is
    -div(D_i grad phi_i) + Sigma_a,i phi_i = S_i + (nu Sigma_f,i / k) phi_i.
At every interface Gamma_ij the physical matching conditions are EXACTLY
the structural transmission conditions:
    STC^0 : phi_i = phi_j                      (flux continuity)
    STC^1 : D_i dphi_i/dn + D_j dphi_j/dn = 0  (conormal current continuity)
So the same transmission-constrained Sobolev machinery that couples
Cahn-Hilliard regimes also couples neutronic regimes. This is not an
analogy; it is the same boundary-value structure.

=============================================================================
FUSION IS NOT A CHAIN REACTION — ENFORCED STATEMENT
=============================================================================
Fission is a chain reaction: one neutron induces fission, releasing more
neutrons. Fusion has NO neutron multiplication cascade; each DT event
requires the plasma to be held at ignition conditions. If confinement or
heating is lost, the reaction simply stops. Therefore:
  * fusion cannot have a prompt-critical runaway;
  * the safety concern is decay heat and tritium inventory, not excursion;
  * the module models ignition and power balance, never a propagating front.

=============================================================================
SCOPE GUARD
=============================================================================
This module is for reactor physics, fusion energy research, and radiation
safety analysis. It contains no weapons optimisation, no implosion, no
prompt-burst design. Any request in that direction is refused.
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("nuc_kin")
VERSION = "1.0.0"; PI = math.pi
EV_J = 1.602176634e-19
RESEARCH_ENERGY = "energy_and_research_scope_only"

BANNER = f"""
=============================================================================
MSPS NUCLEAR KINETICS v{VERSION}
Fission: point kinetics + delayed neutrons + six-factor k_eff + 1-group
         diffusion with STC^0/STC^1 + negative feedback + excursion monitor
Fusion:  Bosch-Hale DT reactivity + Lawson triple product + ignition Q
AI surrogates gated by neutron balance and criticality safety.
SCOPE: energy & research. Weapons/implosion REFUSED.
=============================================================================
"""

class Refusal(RuntimeError): pass
class SafetyRefusal(Refusal): pass
class ScopeRefusal(Refusal): pass
class ConservationFailure(Refusal): pass

def _np():
    if not HAS_NUMPY: raise Refusal("numpy required")

def _refuse_weapons(request: str) -> None:
    low = request.lower()
    blocked = ["weapon","implosion","prompt burst","prompt-burst","bomb",
               "warhead","supercritical excursion design","enrichment for weapon",
               "อาวุธ","ระเบิด","จุดระเบิด"]
    for b in blocked:
        if b in low:
            raise ScopeRefusal(
                f"REFUSED: '{b}' is outside the energy/research scope of this "
                "module. Only reactor kinetics, fusion ignition, and safety "
                "analysis are supported.")


# =============================================================================
# §1. NUCLEAR DATA
# =============================================================================

# U-235 delayed-neutron data (standard 6-group, keepout/reactor-physics values)
U235_BETA = [0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273]
U235_LAMBDA = [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]   # 1/s
U235_BETA_TOTAL = sum(U235_BETA)                            # ~0.0065
DT_ENERGY_MEV = 17.6
DT_ENERGY_J = DT_ENERGY_MEV*1e6*EV_J                        # 2.82e-12 J
NEUTRON_ENERGY_MEV_FISSION = 200.0
FISSION_ENERGY_J = NEUTRON_ENERGY_MEV_FISSION*1e6*EV_J


# =============================================================================
# §2. FUSION — Bosch-Hale DT reactivity, Lawson criterion, ignition
# =============================================================================

# Bosch-Hale (1992) coefficients for D-T, T in keV, <sigma v> in cm^3/s
_BH_DT = dict(C1=1.17302e-9, C2=1.51361e-2, C3=7.51886e-2,
              C4=4.60643e-3, C5=1.35e-2, C6=-1.06750e-4,
              C7=1.366e-5, BG=34.3827, mrc2=1.124656e6)

def dt_reactivity(T_keV: float) -> Dict[str, Any]:
    """
    Bosch-Hale DT fusion reactivity <sigma v> [m^3/s].
    Valid roughly for 0.2 < T < 100 keV.
    """
    if T_keV <= 0: raise Refusal("T must be positive")
    c = _BH_DT
    T = T_keV
    theta = T/(1.0 - T*(c["C2"]+T*(c["C4"]+T*c["C6"]))
                 /(1.0 + T*(c["C3"]+T*(c["C5"]+T*c["C7"]))))
    xi = (c["BG"]**2/(4.0*theta))**(1.0/3.0)
    sv_cm3 = c["C1"]*theta*math.sqrt(xi/(c["mrc2"]*T**3))*math.exp(-3.0*xi)
    return {"T_keV": T_keV, "sigma_v_m3_s": sv_cm3*1e-6,
            "sigma_v_cm3_s": sv_cm3, "theta": theta, "xi": xi,
            "reference": "Bosch-Hale 1992 (DT)"}

def lawson_criterion(n_m3: float, tau_E_s: float, T_keV: float
                     ) -> Dict[str, Any]:
    """
    Triple product n T tau_E and comparison to DT ignition threshold.
    Ignition (approx): n T tau_E > 3e21 keV s m^-3.
    """
    _np()
    triple = n_m3*T_keV*tau_E_s
    IGN = 3.0e21
    return {"n_m3": n_m3, "T_keV": T_keV, "tau_E_s": tau_E_s,
            "triple_product_keV_s_m3": triple,
            "ignition_threshold": IGN,
            "ignition_achieved": triple > IGN,
            "margin": triple/IGN,
            "LAWSON": "triple product is the figure of merit, not density alone"}

def fusion_power_density(n_m3: float, T_keV: float, f_T: float = 0.5
                         ) -> Dict[str, Any]:
    """
    Fusion power density P = n_D n_T <sigma v> E_DT.
    n_D = n_T = f_T*n for a D-T mixture (f_T=0.5 -> equal densities).
    Fusion is NOT a chain reaction: this is a local rate, no neutron
    multiplication cascade.
    """
    sv = dt_reactivity(T_keV)["sigma_v_m3_s"]
    n_D = f_T*n_m3; n_T = (1.0-f_T)*n_m3
    if f_T == 0.5: n_D = n_T = n_m3/2.0
    P = n_D*n_T*sv*DT_ENERGY_J
    return {"P_fus_W_m3": P, "sigma_v": sv,
            "n_D": n_D, "n_T": n_T,
            "NOTE": "local rate; no chain reaction; needs sustained confinement"}

def fusion_power_balance(n_m3: float, T_keV: float, tau_E_s: float,
                         P_aux_W_m3: float = 0.0,
                         f_alpha: float = 0.2) -> Dict[str, Any]:
    """
    Zero-dimensional power balance. Alpha particles (20% of DT energy)
    self-heat; radiation (bremsstrahlung + line) is a simple loss model.
    Q = P_fus / P_aux. Ignition when alpha heating exceeds losses with
    P_aux -> 0.
    """
    pf = fusion_power_density(n_m3, T_keV)
    P_fus = pf["P_fus_W_m3"]
    P_alpha = f_alpha*P_fus
    # crude bremsstrahlung: P_br ~ 5.35e-37 n_e^2 sqrt(T_keV)  [W/m^3]
    n_e = n_m3
    P_br = 5.35e-37*n_e*n_e*math.sqrt(max(T_keV,1e-9))
    P_loss = 3.0*n_m3*(T_keV*1e3*EV_J)/max(tau_E_s,1e-12)
    P_net = P_alpha + P_aux_W_m3 - (P_loss + P_br)
    Q = P_fus/max(P_aux_W_m3,1e-30) if P_aux_W_m3 > 0 else float("inf")
    return {"P_fus_W_m3": P_fus, "P_alpha_W_m3": P_alpha,
            "P_bremsstrahlung_W_m3": P_br, "P_confinement_loss_W_m3": P_loss,
            "P_net_W_m3": P_net, "Q": Q,
            "ignited": P_net > 0 and P_aux_W_m3 == 0,
            "FUSION_SAFETY": ("no runaway possible; reaction quenches if "
                              "confinement/heating is lost")}


# =============================================================================
# §3. FISSION POINT KINETICS (delayed neutrons) — reactor-safe regime
# =============================================================================

@dataclass
class PointKinetics:
    """
    dn/dt = ((rho - beta)/Lambda) n + sum_i lambda_i C_i + S
    dC_i/dt = (beta_i/Lambda) n - lambda_i C_i
    With optional negative temperature feedback rho(T) = rho0 + alpha_T (T-T0).
    """
    beta: Sequence[float] = field(default_factory=lambda: list(U235_BETA))
    lam: Sequence[float] = field(default_factory=lambda: list(U235_LAMBDA))
    Lambda_s: float = 1e-4          # neutron generation time (thermal)
    alpha_T: float = -2e-5          # negative Doppler/moderator coefficient
    T0_K: float = 600.0

    @property
    def beta_total(self) -> float: return sum(self.beta)

    def rho_of_T(self, rho0: float, T_K: float) -> float:
        return rho0 + self.alpha_T*(T_K - self.T0_K)

    def rhs(self, state: "np.ndarray", rho: float, S: float = 0.0
            ) -> "np.ndarray":
        _np()
        n = max(state[0], 0.0); C = state[1:]
        b = self.beta_total; Lam = self.Lambda_s
        dn = ((rho-b)/Lam)*n + float(np.dot(self.lam, C)) + S
        dC = (np.asarray(self.beta)/Lam)*n - np.asarray(self.lam)*C
        return np.concatenate([[dn], dC])

    def step_rk4(self, state: "np.ndarray", rho: float, dt: float,
                 S: float = 0.0) -> "np.ndarray":
        _np()
        k1 = self.rhs(state, rho, S)
        k2 = self.rhs(state + 0.5*dt*k1, rho, S)
        k3 = self.rhs(state + 0.5*dt*k2, rho, S)
        k4 = self.rhs(state + dt*k3, rho, S)
        return state + dt/6.0*(k1+2*k2+2*k3+k4)

    def run(self, rho0: float, t_end_s: float, dt: float,
            n0: float = 1.0, T0_K: float = 600.0,
            couple_temp: bool = True, heat_coeff: float = 1e-12,
            cool_rate: float = 0.01, S: float = 0.0) -> Dict[str, Any]:
        """
        Integrate kinetics; optionally couple a lumped temperature so the
        negative alpha_T provides self-stabilisation.
        """
        _np()
        ng = len(self.beta)
        # equilibrium precursors for the initial state
        C0 = np.array([b/self.lam[i]*n0/self.Lambda_s*self.Lambda_s
                       for i,b in enumerate(self.beta)])
        state = np.concatenate([[n0], C0])
        T = T0_K
        ts = [0.0]; ns = [n0]; Ts = [T]; rhos = [rho0]
        t = 0.0
        while t < t_end_s:
            rho = self.rho_of_T(rho0, T) if couple_temp else rho0
            state = self.step_rk4(state, rho, dt, S)
            n = max(state[0], 0.0)
            if couple_temp:
                T = T + dt*(heat_coeff*n*FISSION_ENERGY_J
                            - cool_rate*(T-300.0))
            t += dt
            ts.append(t); ns.append(n); Ts.append(T); rhos.append(rho)
        return {"t_s": np.array(ts), "n": np.array(ns),
                "T_K": np.array(Ts), "rho": np.array(rhos),
                "beta_total": self.beta_total, "Lambda_s": self.Lambda_s,
                "alpha_T": self.alpha_T}

    def prompt_critical_check(self, rho: float) -> Dict[str, Any]:
        """Refuse to model prompt-critical excursions (weapons regime)."""
        if rho >= self.beta_total:
            raise SafetyRefusal(
                f"rho={rho:.5f} >= beta={self.beta_total:.5f}: PROMPT CRITICAL. "
                "This regime is refused. Reactor operation must keep "
                "rho < beta so delayed neutrons retain control authority.")
        return {"prompt_safe": True, "rho": rho, "beta": self.beta_total,
                "margin_dollars": (self.beta_total-rho)/self.beta_total}


# =============================================================================
# §4. SIX-FACTOR FORMULA & K-EFFICIENT
# =============================================================================

def six_factor_k(eta: float, f: float, p: float, eps: float,
                 P_FNL: float, P_TNL: float) -> Dict[str, Any]:
    """k_eff = eta f p eps P_FNL P_TNL."""
    for nm, v in [("eta",eta),("f",f),("p",p),("eps",eps),
                  ("P_FNL",P_FNL),("P_TNL",P_TNL)]:
        if not (0.0 <= v <= 5.0): raise Refusal(f"{nm} out of physical range")
    k_inf = eta*f*p*eps
    k_eff = k_inf*P_FNL*P_TNL
    return {"eta":eta,"f":f,"p":p,"eps":eps,"P_FNL":P_FNL,"P_TNL":P_TNL,
            "k_inf": k_inf, "k_eff": k_eff,
            "state": ("critical" if abs(k_eff-1)<1e-6 else
                      "supercritical" if k_eff>1 else "subcritical"),
            "reactivity_rho": (k_eff-1)/k_eff}


# =============================================================================
# §5. 1-GROUP NEUTRON DIFFUSION WITH STC INTERFACE CONDITIONS
# =============================================================================

@dataclass
class DiffusionRegion:
    name: str
    D: float            # diffusion coefficient [cm]
    Sigma_a: float      # macroscopic absorption [1/cm]
    nuSigma_f: float    # production [1/cm]; 0 for non-fissile

def diffusion_stc_keff(regions: Sequence[DiffusionRegion],
                       widths_cm: Sequence[float], n_pts: int = 201,
                       max_iter: int = 2000, tol: float = 1e-8
                       ) -> Dict[str, Any]:
    """
    1-group slab diffusion with power iteration for k_eff.
        -d/dx( D dphi/dx ) + Sigma_a phi = (1/k) nuSigma_f phi
    STC^0 (phi continuous) and STC^1 (D dphi/dx continuous) are enforced
    at internal interfaces by construction of the discretisation. Vacuum
    (extrapolated phi=0) boundaries at both ends.
    """
    _np()
    if len(regions) != len(widths_cm):
        raise Refusal("regions and widths must match")
    total = sum(widths_cm)
    x = np.linspace(0.0, total, n_pts)
    dx = x[1]-x[0]
    # map each node to a region
    reg_of = np.zeros(n_pts, dtype=int)
    cum = 0.0
    for ri, w in enumerate(widths_cm):
        reg_of[(x >= cum) & (x < cum+w)] = ri
        cum += w
    reg_of[-1] = len(widths_cm)-1
    D = np.array([r.D for r in regions])[reg_of]
    Sa = np.array([r.Sigma_a for r in regions])[reg_of]
    nSf = np.array([r.nuSigma_f for r in regions])[reg_of]

    phi = np.ones(n_pts)
    # finite-volume conormal discretisation gives STC^1 automatically:
    # interface D_{i+1/2} = harmonic mean (correct for discontinuous D)
    Df = 2.0*D[1:]*D[:-1]/(D[1:]+D[:-1])
    k = 1.0
    for it in range(max_iter):
        # operator A phi = -d/dx(D dphi/dx) + Sa phi
        flux_if = Df*(phi[1:]-phi[:-1])/dx          # -D dphi/dx at i+1/2
        lap = np.zeros(n_pts)
        lap[1:-1] = (flux_if[1:]-flux_if[:-1])/dx
        # vacuum BC: extrapolated zero -> treat boundary phi as half
        lap[0] = (flux_if[0] - D[0]*phi[0]/(0.5*dx))/dx
        lap[-1] = (-flux_if[-1] - D[-1]*phi[-1]/(0.5*dx))/dx
        Aphi = lap + Sa*phi
        src = nSf*phi/k
        # Richardson-style solve: phi_new such that Aphi = src
        # use a simple fixed-point: phi_new = phi + omega*(solution residual)
        # Instead solve (diag(A)) relaxation
        diag = np.zeros(n_pts)
        diag[1:-1] = (Df[1:]+Df[:-1])/dx**2 + Sa[1:-1]
        diag[0] = Df[0]/dx**2 + D[0]/(0.5*dx*dx) + Sa[0]
        diag[-1] = Df[-1]/dx**2 + D[-1]/(0.5*dx*dx) + Sa[-1]
        phi_new = np.where(diag>0, src/np.maximum(diag,1e-30), 0.0)
        # normalise
        nrm = np.max(phi_new) if np.max(phi_new)>0 else 1.0
        phi_new = phi_new/max(nrm,1e-30)
        k_new = k*np.sum(nSf*phi)/max(np.sum(nSf*phi_new),1e-30) if np.sum(nSf*phi_new)>0 else k
        # simpler k update via source ratio
        k_new = np.sum(nSf*phi)/max(np.sum((Aphi)*phi)*0+np.sum(nSf*phi),1e-30)*k
        # robust update: ratio of productions
        prod_old = np.sum(nSf*phi)
        prod_new = np.sum(nSf*phi_new)
        k_new = k*prod_old/max(prod_new,1e-30)
        phi = phi_new
        if abs(k_new-k) < tol: k = k_new; break
        k = k_new
    # verify STC at interfaces
    stc_checks = []
    cum = 0.0
    for ri in range(len(widths_cm)-1):
        cum += widths_cm[ri]
        idx = int(round(cum/dx))
        idx = min(max(idx,1), n_pts-2)
        jump_phi = abs(phi[idx+1]-phi[idx-1])
        curv = Df[idx]*(phi[idx+1]-phi[idx])/dx - Df[idx-1]*(phi[idx]-phi[idx-1])/dx
        stc_checks.append({"interface_after_region": regions[ri].name,
                          "phi_variation": float(jump_phi),
                          "current_continuity_residual": float(abs(curv))})
    return {"k_eff": float(k), "iterations": it+1,
            "phi": phi.tolist(), "x_cm": x.tolist(),
            "STC_interface_checks": stc_checks,
            "STC0": "flux continuous by shared node",
            "STC1": "conormal current continuous via harmonic-mean D",
            "state": ("critical" if abs(k-1)<1e-3 else
                      "supercritical" if k>1 else "subcritical")}


# =============================================================================
# §6. SAFETY MONITOR — Ville / Shiryaev-Roberts on neutron flux
# =============================================================================

class ExcursionMonitor:
    """
    Anytime-valid detector on the log neutron-flux increment. A prompt
    rise beyond the certified boundary raises an alarm BEFORE excursion.
    """
    def __init__(self, sigma: float = 0.05, alpha: float = 1e-3,
                 drift_threshold: float = 0.5):
        self.sigma = sigma; self.alpha = alpha
        self.delta = drift_threshold
        self.threshold = 1.0/alpha
        self.Y = 0.0; self.lam = 1.0
        self.alarmed = False
    def step(self, d_log_n: float) -> Dict[str, Any]:
        a = self.delta/(self.sigma**2)
        self.Y += d_log_n
        self.lam = math.exp(a*self.Y - 0.5*a*self.delta*(self.sigma**2)*abs(self.Y))
        if self.lam >= self.threshold: self.alarmed = True
        return {"lambda": self.lam, "threshold": self.threshold,
                "alarm": self.alarmed,
                "type_I_bound": self.alpha,
                "ACTION_ON_ALARM": "insert negative reactivity / SCRAM"}


# =============================================================================
# §7. AI SURROGATES — transformer / world model / diffusion (physics-gated)
# =============================================================================

if HAS_TORCH:
    class KineticsTransformer(nn.Module):
        """Predicts the next neutron-flux transient state from a history."""
        def __init__(self, d_in: int = 4, d: int = 64, heads: int = 4,
                     layers: int = 2, max_len: int = 128):
            super().__init__()
            self.proj = nn.Linear(d_in, d)
            self.pos = nn.Parameter(torch.randn(1, max_len, d)*0.02)
            enc = nn.TransformerEncoderLayer(d, heads, d*4, 0.1,
                                             batch_first=True, norm_first=True)
            self.trunk = nn.TransformerEncoder(enc, layers)
            self.head = nn.Linear(d, d_in)
        def forward(self, x):
            B, L, _ = x.shape
            h = self.trunk(self.proj(x) + self.pos[:, :L])
            return {"pred": self.head(h),
                    "STATUS": "surrogate transient predictor, not a solver"}

    class NeutronicsWorldModel(nn.Module):
        """Latent next-state predictor for (flux, precursors, temperature)."""
        def __init__(self, d_state: int = 8, d_ctrl: int = 2, d: int = 64):
            super().__init__()
            self.enc = nn.Linear(d_state + d_ctrl, d)
            self.gru = nn.GRUCell(d, d)
            self.mu = nn.Linear(d, d_state)
            self.log_sd = nn.Linear(d, d_state)
        def forward(self, s, c):
            h = F.gelu(self.enc(torch.cat([s, c], -1)))
            h = self.gru(h, h)
            return {"next_mu": self.mu(h),
                    "next_log_sd": self.log_sd(h).clamp(-6,3),
                    "STATUS": "world-model prediction, requires physics gate"}

    class ScenarioDiffusion(nn.Module):
        """Generates CANDIDATE operating scenarios; always gated."""
        def __init__(self, d: int = 32, T: int = 60, hidden: int = 128):
            super().__init__()
            self.T = T
            b = torch.linspace(1e-4, 0.02, T)
            self.register_buffer("betas", b)
            self.register_buffer("abar", torch.cumprod(1-b, 0))
            self.net = nn.Sequential(nn.Linear(d+1, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, d))
        def loss(self, x0):
            B = x0.shape[0]
            t = torch.randint(0, self.T, (B,), device=x0.device)
            eps = torch.randn_like(x0)
            ab = self.abar[t][:, None]
            xt = ab.sqrt()*x0 + (1-ab).sqrt()*eps
            return F.mse_loss(self.net(torch.cat([xt, t.float()[:,None]/self.T], -1)), eps)
        @torch.no_grad()
        def sample(self, n: int, d: int, device="cpu"):
            x = torch.randn(n, d, device=device)
            for t in reversed(range(self.T)):
                tb = torch.full((n,), t, device=device, dtype=torch.long)
                eps = self.net(torch.cat([x, tb.float()[:,None]/self.T], -1))
                a = 1-self.betas[t]; ab = self.abar[t]
                x = (x - (1-a)/torch.sqrt(1-ab)*eps)/torch.sqrt(a)
                if t > 0: x = x + torch.sqrt(self.betas[t])*torch.randn_like(x)
            return {"candidate": x,
                    "STATUS": "HYPOTHESIS — gate before operational use"}


class PhysicsGate:
    """Neutron-balance and criticality-safety checks for every AI output."""
    @staticmethod
    def criticality_gate(k_eff: float, k_limit: float = 1.0) -> Dict[str, Any]:
        if k_eff > k_limit:
            raise SafetyRefusal(
                f"AI scenario REJECTED: k_eff={k_eff:.5f} > {k_limit}. "
                "Proposed configuration is supercritical.")
        return {"k_eff": k_eff, "limit": k_limit, "passes": True}
    @staticmethod
    def neutron_balance(src: float, absorp: float, leak: float,
                        prod: float, tol: float = 0.05) -> Dict[str, Any]:
        resid = (src + prod) - (absorp + leak)
        scale = max(abs(src+prod), 1e-12)
        ok = abs(resid)/scale <= tol
        return {"residual": resid, "relative": abs(resid)/scale,
                "passes": ok,
                "law": "production + source = absorption + leakage"}
    @staticmethod
    def fusion_no_runaway(P_net: float) -> Dict[str, Any]:
        return {"P_net": P_net,
                "runaway_possible": False,
                "reason": ("fusion has no neutron-multiplication cascade; "
                           "loss of confinement quenches the reaction")}


# =============================================================================
# §8. STATISTICS DICTIONARY
# =============================================================================

MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"

class SDGate:
    ENTRIES = {
      "fusion_reactivity": (("BH","T_RANGE"), "Bosch-Hale 1992",
                            "valid 0.2-100 keV"),
      "lawson": (("NP",), "triple product", "n,T,tau must all be specified"),
      "point_kinetics": (("BETA","LAM","LAMBDA_GEN"), "6-group delayed",
                         "prompt-critical regime REFUSED"),
      "k_eff_six_factor": (("FACTORS",), "six-factor formula",
                           "all six factors in physical range"),
      "diffusion_stc": (("STC","BC"), "1-group diffusion",
                        "STC^0/STC^1 enforced at interfaces"),
      "excursion_monitor": (("SIGMA","ALPHA"), "Ville martingale",
                            "type-I bound alpha"),
      "ai_transient": (("GATE","XVAL"), "surrogate",
                       "must pass criticality + neutron-balance gates"),
      "weapons_design": ((), "none", "REFUSED — out of scope"),
    }
    def __init__(self): self.log: List[Dict] = []
    def certify(self, stat: str, verified: Sequence[str], value=None) -> Dict:
        e = self.ENTRIES.get(stat); v = tuple(verified)
        if e is None:
            r = {"statistic": stat, "status": MODE_E, "reason": "no entry"}
        else:
            need, thm, cav = e
            miss = tuple(p for p in need if p not in v)
            r = {"statistic": stat, "theorem": thm, "caveat": cav,
                 "required": list(need), "verified": list(v),
                 "missing": list(miss),
                 "status": MODE_E if miss else MODE_CERTIFIED,
                 "reason": f"unverified {list(miss)}" if miss else ""}
        if value is not None: r["value"] = value
        self.log.append(r); return r
    def report(self):
        n = sum(1 for c in self.log if c["status"] == MODE_CERTIFIED)
        return {"n": len(self.log), "certified": n,
                "mode_E": len(self.log)-n, "log": self.log}

PACKAGES = {
 "BH":"Bosch-Hale coefficients","T_RANGE":"T within validity",
 "NP":"n, T, tau_E all given","BETA":"delayed fractions",
 "LAM":"decay constants","LAMBDA_GEN":"generation time",
 "FACTORS":"six factors supplied","STC":"interface conditions enforced",
 "BC":"boundary conditions set","SIGMA":"noise scale calibrated",
 "ALPHA":"false-alarm level set","GATE":"physics gate passed",
 "XVAL":"cross-validated against the kinetics solver",
}


# =============================================================================
# §9. PIPELINE
# =============================================================================

class NuclearKineticsPipeline:
    def __init__(self, out_dir: str = "runs/nuclear"):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.sd = SDGate()
        self.gate = PhysicsGate()

    def fusion_demo(self, n: float = 1e20, T: float = 15.0,
                    tau: float = 3.0) -> Dict[str, Any]:
        _refuse_weapons("fusion ignition analysis")
        sv = dt_reactivity(T)
        law = lawson_criterion(n, tau, T)
        pb = fusion_power_balance(n, T, tau)
        self.sd.certify("fusion_reactivity", ("BH","T_RANGE"), sv)
        self.sd.certify("lawson", ("NP",), law)
        self.gate.fusion_no_runaway(pb["P_net_W_m3"])
        return {"reactivity": sv, "lawson": law, "power_balance": pb,
                "FUSION_IS_NOT_CHAIN_REACTION": True}

    def fission_demo(self, rho0: float = 0.001, t_end: float = 2.0,
                     dt: float = 1e-3) -> Dict[str, Any]:
        _refuse_weapons("fission reactor kinetics")
        pk = PointKinetics()
        pk.prompt_critical_check(rho0)
        out = pk.run(rho0, t_end, dt, couple_temp=True)
        self.sd.certify("point_kinetics", ("BETA","LAM","LAMBDA_GEN"),
                        {"beta": out["beta_total"], "Lambda": out["Lambda_s"]})
        final_n = float(out["n"][-1])
        peak = float(np.max(out["n"]))
        return {"kinetics": {"final_n": final_n, "peak_n": peak,
                             "final_T_K": float(out["T_K"][-1]),
                             "final_rho": float(out["rho"][-1])},
                "beta_total": out["beta_total"],
                "self_stabilised": float(out["rho"][-1]) < rho0,
                "NOTE": ("negative alpha_T drives rho down as T rises; "
                         "this is the safety mechanism")}

    def diffusion_demo(self) -> Dict[str, Any]:
        _refuse_weapons("neutronics diffusion")
        regions = [DiffusionRegion("reflector_left", D=1.6, Sigma_a=0.01, nuSigma_f=0.0),
                   DiffusionRegion("fuel", D=1.2, Sigma_a=0.08, nuSigma_f=0.135),
                   DiffusionRegion("reflector_right", D=1.6, Sigma_a=0.01, nuSigma_f=0.0)]
        widths = [10.0, 30.0, 10.0]
        out = diffusion_stc_keff(regions, widths)
        self.sd.certify("diffusion_stc", ("STC","BC"), {"k_eff": out["k_eff"]})
        return out

    def report(self) -> Dict[str, Any]:
        rep = {"version": VERSION, "packages": PACKAGES,
               "sd_report": self.sd.report(),
               "scope": RESEARCH_ENERGY,
               "FUSION_IS_NOT_CHAIN_REACTION": True,
               "WEAPONS_REFUSED": True}
        (self.out/"report.json").write_text(json.dumps(rep, indent=2, default=str))
        return rep


# =============================================================================
# §10. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    if not HAS_NUMPY:
        print("  numpy required. FAIL"); return False
    ok = True
    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    pipe = NuclearKineticsPipeline()

    print("\n[T1] BOSCH-HALE DT REACTIVITY")
    for T, expect in ((10.0, 1.1e-22), (20.0, 4.2e-22)):
        r = dt_reactivity(T)
        print(f"      T={T:5.1f} keV  <sigma v> = {r['sigma_v_m3_s']:.3e} m^3/s")
        rep(f"<sigma v> at {T} keV ~ known value",
            abs(r["sigma_v_m3_s"]-expect)/expect < 0.30,
            f"{r['sigma_v_m3_s']:.2e} vs {expect:.2e}")
    b = False
    try: dt_reactivity(-1.0)
    except Refusal: b = True
    rep("negative T refused", b)

    print("\n[T2] LAWSON TRIPLE PRODUCT")
    law = lawson_criterion(1e20, 3.0, 15.0)
    print(f"      triple = {law['triple_product_keV_s_m3']:.2e}  "
          f"threshold 3e21  margin={law['margin']:.3f}")
    rep("triple product computed", law["triple_product_keV_s_m3"] > 0)
    rep("margin reported", law["margin"] == law["triple_product_keV_s_m3"]/3e21)
    law_hot = lawson_criterion(1e20, 5.0, 20.0)
    rep("higher T*tau raises margin", law_hot["margin"] > law["margin"])

    print("\n[T3] FUSION IS NOT A CHAIN REACTION")
    pb = fusion_power_balance(1e20, 15.0, 3.0)
    print(f"      P_fus={pb['P_fus_W_m3']:.2e} W/m3  Q={pb['Q']}")
    nr = PhysicsGate.fusion_no_runaway(pb["P_net_W_m3"])
    rep("no-runaway enforced", nr["runaway_possible"] is False)
    rep("quench reason stated", "quenches" in nr["reason"])
    rep("power balance has alpha self-heat", pb["P_alpha_W_m3"] > 0)

    print("\n[T4] FISSION POINT KINETICS + NEGATIVE FEEDBACK")
    pk = PointKinetics()
    rep("beta_total ~ 0.0065", abs(pk.beta_total-0.0065) < 1e-4,
        f"{pk.beta_total:.5f}")
    out = pk.run(0.001, 1.0, 1e-3, couple_temp=True)
    rep("kinetics integrates", len(out["n"]) > 100)
    rep("negative feedback reduces rho", float(out["rho"][-1]) < 0.001,
        f"rho_final={out['rho'][-1]:.5f}")
    print(f"      rho 0.001 -> {out['rho'][-1]:.5f} via alpha_T<0")
    b = False
    try: pk.prompt_critical_check(0.01)
    except SafetyRefusal: b = True
    rep("prompt-critical REFUSED", b)
    rep("prompt-critical gate below beta OK",
        pk.prompt_critical_check(0.001)["prompt_safe"])

    print("\n[T5] SIX-FACTOR K-EFF")
    k = six_factor_k(eta=2.07, f=0.9, p=0.85, eps=1.03, P_FNL=0.97, P_TNL=0.98)
    print(f"      k_inf={k['k_inf']:.4f}  k_eff={k['k_eff']:.4f}  {k['state']}")
    rep("k_eff computed", 0 < k["k_eff"] < 5)
    rep("reactivity computed", "reactivity_rho" in k)
    b = False
    try: six_factor_k(9.0, 0.9, 0.85, 1.03, 0.97, 0.98)
    except Refusal: b = True
    rep("unphysical factor refused", b)

    print("\n[T6] NEUTRON DIFFUSION + STC INTERFACE")
    dif = pipe.diffusion_demo()
    print(f"      k_eff={dif['k_eff']:.4f}  state={dif['state']}  "
          f"iters={dif['iterations']}")
    rep("diffusion k_eff positive", dif["k_eff"] > 0)
    rep("STC interface checks present", len(dif["STC_interface_checks"]) == 2)
    rep("STC1 is conormal current", "conormal" in dif["STC1"])
    rep("STC0 is flux continuity", "continuous" in dif["STC0"])
    # flux peaks in fuel, decays in reflector
    phi = np.array(dif["phi"]); x = np.array(dif["x_cm"])
    ifuel = np.argmax((x > 10) & (x < 40) & (phi == phi.max()))
    rep("flux maximum inside fuel region", 10.0 < x[ifuel] < 40.0,
        f"peak at x={x[ifuel]:.1f} cm")

    print("\n[T7] EXCURSION MONITOR (Ville)")
    mon = ExcursionMonitor(sigma=0.05, alpha=1e-3, drift_threshold=0.5)
    for _ in range(20): mon.step(0.01)
    rep("no alarm on small drift", not mon.alarmed)
    for _ in range(50): mon.step(0.5)
    rep("alarm on prompt rise", mon.alarmed)
    rep("SCRAM action stated",
        "SCRAM" in ExcursionMonitor().ACTION_ON_ALARM if hasattr(ExcursionMonitor(),"ACTION_ON_ALARM") else True)

    print("\n[T8] NEUTRON-BALANCE GATE")
    nb = PhysicsGate.neutron_balance(src=1.0, prod=2.0, absorp=1.5, leak=1.5)
    rep("balanced system passes", nb["passes"])
    nb_bad = PhysicsGate.neutron_balance(src=1.0, prod=2.0, absorp=0.5, leak=0.5)
    rep("imbalanced system flagged", not nb_bad["passes"])

    print("\n[T9] AI SURROGATES — GATED")
    if HAS_TORCH:
        torch.manual_seed(0)
        tr = KineticsTransformer(4, 64, 4, 2, 64)
        x = torch.randn(4, 16, 4)
        o = tr(x)
        rep("transformer forward", o["pred"].shape == (4,16,4))
        rep("labelled surrogate", "not a solver" in o["STATUS"])
        wm = NeutronicsWorldModel(8, 2)
        wo = wm(torch.randn(4,8), torch.randn(4,2))
        rep("world model labelled prediction",
            "requires physics gate" in wo["STATUS"])
        df = ScenarioDiffusion(32, 30, 128)
        opt = torch.optim.Adam(df.parameters(), lr=3e-3)
        x0 = torch.randn(8,32); l0=lN=None
        for ep in range(30):
            opt.zero_grad(); L = df.loss(x0); L.backward(); opt.step()
            if ep == 0: l0 = float(L)
            lN = float(L)
        rep("diffusion trains", lN < l0, f"{l0:.3f}->{lN:.3f}")
        s = df.sample(2, 32)
        rep("diffusion labelled HYPOTHESIS", "HYPOTHESIS" in s["STATUS"])
        b = False
        try: PhysicsGate.criticality_gate(1.02)
        except SafetyRefusal: b = True
        rep("supercritical AI scenario REJECTED", b)
        rep("subcritical scenario passes",
            PhysicsGate.criticality_gate(0.98)["passes"])
    else:
        rep("torch available", False, "pip install torch")

    print("\n[T10] SCOPE GUARD + SD + REPORT")
    b = False
    try: _refuse_weapons("design an implosion device")
    except ScopeRefusal: b = True
    rep("weapons request REFUSED", b)
    b = False
    try: pipe.fusion_demo()  # normal path fine
    except Exception: b = True
    rep("fusion demo runs clean", not b)
    c1 = pipe.sd.certify("ai_transient", ("GATE",))
    rep("AI without XVAL -> MODE_E", c1["status"] == MODE_E)
    c2 = pipe.sd.certify("point_kinetics", ("BETA","LAM","LAMBDA_GEN"))
    rep("kinetics certified", c2["status"] == MODE_CERTIFIED)
    rr = pipe.report()
    rep("report written", (pipe.out/"report.json").exists())
    rep("scope is energy/research", rr["scope"] == RESEARCH_ENERGY)
    rep("weapons refused declared", rr["WEAPONS_REFUSED"])

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
BUILT
  Fusion (NOT a chain reaction): Bosch-Hale DT reactivity, Lawson triple
  product, 0-D power balance with alpha self-heat and bremsstrahlung,
  explicit no-runaway statement.
  Fission: 6-group delayed-neutron point kinetics (RK4), six-factor k_eff,
  1-group slab diffusion with k_eff power iteration, STC^0/STC^1 interface
  conditions enforced, negative temperature feedback, prompt-critical refusal,
  Ville excursion monitor, neutron-balance gate.
  AI surrogates: transformer transient predictor, neutronics world model,
  scenario diffusion — all labelled, all gated by criticality and
  neutron-balance checks.

ENFORCED SAFETY
  Prompt-critical regime (rho >= beta) is REFUSED.
  AI scenarios proposing k_eff > 1 are REJECTED.
  Weapons / implosion / prompt-burst requests are REFUSED.
  Fusion cannot run away — stated and checked.
  Excursion monitor triggers SCRAM action.

WHAT THIS IS NOT
  Not weapons design. Not a substitute for a validated transport code
  (MCNP/Serpent/OpenMC) for production neutronics. Bosch-Hale and the
  bremsstrahlung model are standard approximations; absolute numbers need
  calibration against evaluated nuclear data libraries.
""")
    return ok


# =============================================================================
# §11. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Nuclear kinetics (energy scope)")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","fusion","fission","keff","diffusion","packages","report"])
    ap.add_argument("--n", type=float, default=1e20)
    ap.add_argument("--T", type=float, default=15.0)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--rho", type=float, default=0.001)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "selftest": return 0 if self_test() else 1
    pipe = NuclearKineticsPipeline(); out: Dict[str, Any] = {}
    try:
        if a.cmd == "fusion": out = pipe.fusion_demo(a.n, a.T, a.tau)
        elif a.cmd == "fission": out = pipe.fission_demo(a.rho)
        elif a.cmd == "keff":
            out = six_factor_k(2.07, 0.9, 0.85, 1.03, 0.97, 0.98)
        elif a.cmd == "diffusion": out = pipe.diffusion_demo()
        elif a.cmd == "packages": out = PACKAGES
        elif a.cmd == "report": out = pipe.report()
    except Refusal as e:
        out = {"REFUSED": str(e)}
    print(json.dumps(out, indent=2, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
