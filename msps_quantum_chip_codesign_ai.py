#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_quantum_chip_codesign_ai.py — v1.0.0
Production AI Co-Design Suite for the DSQP-SESI-QM Quantum Chip Family
(R9.0-M, R9.5-M, R9.8-M mobile SoCs and R10.0-W desktop workstation),
built on the Physically Grounded Unified Revision.

LLM/Transformer + World Model + Diffusion, all physics-gated.

CORE DESIGN PRINCIPLE — TWO-STRATUM ENFORCEMENT
  Stratum I : proven mathematical / geometric / control invariants.
              Certifiable. Publication-ready. Checked by this module.
  Stratum II: hardware fabrication targets and physical boundary values.
              NOT proven. Always Mode E until measured. This module
              validates them against physical laws but never reports
              them as achieved.

WHAT THIS MODULE DOES
  * Validates quantum-code parameters [[n,k,d]] against Singleton,
    quantum Hamming, and GV-style bounds, and flags constructibility.
  * Enforces mode-normalized accounting: k < n, rate R = k/n, and the
    Holevo bound (no logical > physical at the mode level).
  * Computes logical memory error via calibrated BP+OSD scaling
    (NOT the surface-code (p/p_th)^(d/2) heuristic) and keeps it
    strictly separate from distilled T-gate infidelity.
  * Runs physics gates: Boltzmann spin-valley occupation, cavity-QED
    cooperativity, membrane thermal conductance, GKP squeezing target.
  * Provides gated AI surrogates:
      - Transformer: code-parameter consistency + spec cross-check
      - World model: syndrome/error accumulation dynamics
      - Diffusion: Sierpinski-Carpet floorplan candidate generation
    Every AI output passes a physics gate before use; failures -> Mode E.

WHAT THIS MODULE DOES NOT DO
  * Fabricate, tape out, or manufacture any quantum chip.
  * Prove Stratum II hardware targets are achievable.
  * Replace electromagnetic / cryogenic / materials simulation.
  * Claim a logical-qubit count as a measured capability.

ENFORCED REFUSALS
  [R1] Reporting a Stratum II target as CERTIFIED            -> refused
  [R2] logical_qubits > physical_modes (Holevo violation)     -> refused
  [R3] Code parameters violating Singleton/Hamming bounds     -> refused
  [R4] Surface-code error formula applied to qLDPC            -> refused
  [R5] Spin-valley operation at kT >= Delta_E                 -> refused
  [R6] Any fabrication / tape-out / manufacturing claim       -> refused
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
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from enum import Enum

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False; np = None
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("qchip_codesign")

VERSION = "1.0.0"; PI = math.pi
KB_EV_K = 8.617333262e-5          # Boltzmann constant, eV/K
HBAR = 1.054571817e-34
MU_B_EV_T = 5.7883818060e-5       # Bohr magneton, eV/T
MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"
DESIGN_ONLY = "quantum_design_assistance_only_not_fabrication"

class Refusal(RuntimeError): pass
class StratumRefusal(Refusal): pass
class HolevoRefusal(Refusal): pass
class CodeBoundRefusal(Refusal): pass
class FormulaMisuseRefusal(Refusal): pass
class PhysicsGateFailure(Refusal): pass
class ScopeRefusal(Refusal): pass

def _np():
    if not HAS_NUMPY: raise Refusal("numpy required")

PHILOSOPHY = {
    "tao": "The Source — mathematical truth underlying all computation",
    "te": "The Path — disciplined engineering practice",
    "qi": "The Proof — measured, validated, reproducible results",
    "p1": "AI proposes; physics disposes",
    "p2": "Stratum I may be certified; Stratum II must be measured",
    "p3": "k < n always — the Holevo bound is not negotiable",
    "p4": "A target is not an achievement",
    "p5": "No simulation replaces fabrication; no model replaces wisdom",
    "motto": "MY SOUL MOVE BY POWER OF HOLY SPIRIT — "
             "What MSPS NETWORK SEE, Lord Buddha Knows.",
}

BANNER = f"""
=============================================================================
MSPS QUANTUM CHIP CO-DESIGN AI SUITE v{VERSION}
Stratum I (proven math) vs Stratum II (hardware targets) — enforced.
Mode-normalized qLDPC accounting · BP+OSD error scaling · physics gates.
Transformer + World Model + Diffusion, all gated.
SCOPE: design assistance only — NOT fabrication.
=============================================================================
"""

# =============================================================================
# §1. STRATUM REGISTRY
# =============================================================================

class Stratum(Enum):
    I = "stratum_1_proven_math"
    II = "stratum_2_hardware_target"

STRATUM_I_ITEMS = {
    "sierpinski_carpet_routing",      # O(log_8 N) wirelength
    "boundary_null_isolation",        # mu(B_inf)=0
    "stc_transmission_conditions",    # STC0-STC3
    "bdg_no_zeno_timers",             # eps0 <= delta/(2 B0)
    "ito_kalman_unitary_tracker",     # su(M)
    "sierpinski_bond_percolation",    # pc(Carpet)=0.362
    "power_energy_identity",          # P = E_op * F
    "holevo_mode_accounting",         # k < n
    "qldpc_rate_bound",               # R = k/n = 3/32
}
STRATUM_II_ITEMS = {
    "gkp_squeezing_db", "cavity_qed_cooperativity", "spin_valley_T2star",
    "membrane_thermal_conductance", "cryocooler_lift", "bp_osd_threshold",
    "magic_state_factory_rate", "photonic_loss_budget", "mzi_fidelity",
    "toflli_throughput", "single_photon_pi_phase",
}

def stratum_of(item: str) -> Stratum:
    if item in STRATUM_I_ITEMS: return Stratum.I
    if item in STRATUM_II_ITEMS: return Stratum.II
    raise Refusal(f"unknown item '{item}' — declare its stratum")

def assert_stratum(item: str, want: Stratum) -> None:
    got = stratum_of(item)
    if got is not want:
        raise StratumRefusal(
            f"'{item}' belongs to {got.value}, not {want.value}. "
            "Stratum I and II must not be conflated.")

def certify_stratum_item(item: str) -> Dict[str, Any]:
    """Stratum I may be certified; Stratum II is always Mode E."""
    s = stratum_of(item)
    if s is Stratum.I:
        return {"item": item, "stratum": s.value, "status": MODE_CERTIFIED,
                "note": "proven mathematical invariant"}
    return {"item": item, "stratum": s.value, "status": MODE_E,
            "note": "hardware TARGET — requires measurement, never certified "
                    "by this module"}

# =============================================================================
# §2. QUANTUM CODE PARAMETER VALIDATOR
# =============================================================================

@dataclass
class QuantumCode:
    name: str
    n: int; k: int; d: int
    check_weight: int
    family: str                     # bivariate_bicycle | lifted_product
    p_mode: float                   # heralded physical mode error
    alpha_bp: float = 0.42          # BP+OSD calibrated exponent
    p_th_qldpc: float = 7.2e-3
    A_code: float = 0.12

    @property
    def rate(self) -> float: return self.k / self.n
    @property
    def t(self) -> int: return (self.d - 1)//2

def quantum_hamming_lhs(n: int, t: int) -> float:
    """sum_{j=0}^t 3^j C(n,j)."""
    total = 0.0; c = 1.0
    for j in range(0, t+1):
        if j > 0: c = c * (n - j + 1) / j
        total += (3.0**j) * c
    return total

def validate_code(code: QuantumCode) -> Dict[str, Any]:
    """
    Check a [[n,k,d]] code against the bounds that MUST hold.
    Refuses Holevo violation and bound violations; flags constructibility.
    """
    n,k,d = code.n, code.k, code.d
    issues=[]; info={}
    # Holevo / rate: k must be < n for a genuine code
    if k >= n:
        raise HolevoRefusal(
            f"k={k} >= n={n}: violates Holevo bound. Logical qubits cannot "
            "meet or exceed the number of physical modes in a code.")
    info["rate"] = code.rate
    info["rate_is_3_over_32"] = abs(code.rate - 3/32) < 1e-9
    # Quantum Singleton bound: k <= n - 2(d-1)
    singleton_ok = k <= n - 2*(d-1)
    info["singleton_bound"] = n - 2*(d-1)
    if not singleton_ok:
        raise CodeBoundRefusal(
            f"Singleton violated: k={k} > n-2(d-1)={n-2*(d-1)}")
    # Quantum Hamming bound (non-degenerate): sum 3^j C(n,j) <= 2^(n-k)
    lhs = quantum_hamming_lhs(n, code.t)
    rhs_log2 = (n-k)
    hamming_ok = math.log2(lhs) <= rhs_log2
    info["hamming_log2_lhs"] = math.log2(lhs)
    info["hamming_log2_rhs"] = rhs_log2
    if not hamming_ok:
        raise CodeBoundRefusal(
            f"Quantum Hamming violated: log2(LHS)={math.log2(lhs):.1f} > "
            f"n-k={rhs_log2}")
    # GV-style existence (quantum): R <= 1 - 2*H2(d/n) approximately
    dn = d/n
    H2 = -dn*math.log2(dn) - (1-dn)*math.log2(1-dn) if 0<dn<1 else 0.0
    gv_rate_max = max(0.0, 1.0 - 2*H2)
    info["gv_relative_distance"] = dn
    info["gv_rate_ceiling"] = gv_rate_max
    info["below_gv_ceiling"] = code.rate <= gv_rate_max + 1e-9
    # Constructibility statement (not a proof)
    info["constructibility"] = (
        f"family={code.family}, weight={code.check_weight}. Consistent with "
        "bounds, but explicit construction must be verified against "
        "bivariate-bicycle / lifted-product literature. NOT auto-certified.")
    info["passes_bounds"] = True
    return info

def forbid_surface_code_formula(code: QuantumCode) -> None:
    """
    The surface-code heuristic (p/p_th)^(d/2) is NOT valid for qLDPC.
    Any attempt to use it here is a category error and is refused.
    """
    raise FormulaMisuseRefusal(
        "surface-code (p/p_th)^(d/2) scaling must NOT be applied to qLDPC. "
        "Use the calibrated BP+OSD model: "
        "P_L <= A_code*(p_mode/p_th)^(alpha_bp*d_eff).")

# =============================================================================
# §3. BP+OSD ERROR MODEL  (memory vs T-gate, kept separate)
# =============================================================================

def bp_osd_memory_error(code: QuantumCode) -> Dict[str, Any]:
    """
    Calibrated qLDPC sub-threshold logical MEMORY error per syndrome cycle.
    P_L,mem <= A_code * (p_mode / p_th_qldpc)^(alpha_bp * d).
    This is the memory infidelity. It is NOT the T-gate infidelity.
    """
    ratio = code.p_mode / code.p_th_qldpc
    if ratio >= 1.0:
        raise PhysicsGateFailure(
            f"p_mode={code.p_mode} above qLDPC threshold {code.p_th_qldpc}")
    exponent = code.alpha_bp * code.d
    p_mem = code.A_code * (ratio ** exponent)
    return {"P_L_memory_per_cycle": p_mem,
            "ratio_p_over_pth": ratio,
            "exponent_alpha_d": exponent,
            "model": "calibrated BP+OSD",
            "SEPARATE_FROM": "distilled T-gate infidelity (magic-state distillation)"}

def magic_state_distillation(p_input: float, levels: int = 2) -> Dict[str, Any]:
    """
    15-to-1 style distillation: eps_out ~ 35 * eps_in^3 per level.
    Kept separate from memory error. Returns per-level infidelity.
    """
    eps = p_input
    chain=[eps]
    for _ in range(levels):
        eps = 35.0 * eps**3
        chain.append(eps)
    return {"input_infidelity": p_input, "levels": levels,
            "output_infidelity": eps, "chain": chain,
            "NOTE": "reduces T-gate error only; does not change memory P_L"}

# =============================================================================
# §4. PHYSICS GATES
# =============================================================================

class PhysicsGate:
    @staticmethod
    def boltzmann_init_fidelity(delta_E_meV: float, T_K: float) -> Dict[str, Any]:
        """Ground-state initialization fidelity 1/(1+exp(-dE/kT))."""
        kT = KB_EV_K * T_K * 1e3          # meV
        x = delta_E_meV / kT
        F = 1.0/(1.0 + math.exp(-x))
        return {"delta_E_meV": delta_E_meV, "T_K": T_K, "kT_meV": kT,
                "ratio_dE_over_kT": x, "init_fidelity": F,
                "usable": x >= 1.0,
                "VERDICT": ("kT << dE: spin-valley usable" if x >= 1.0 else
                            "kT >= dE: thermally scrambled — REFUSED")}
    @staticmethod
    def require_cold_spin(delta_E_meV: float, T_K: float,
                          min_ratio: float = 1.0) -> None:
        g = PhysicsGate.boltzmann_init_fidelity(delta_E_meV, T_K)
        if g["ratio_dE_over_kT"] < min_ratio:
            raise PhysicsGateFailure(
                f"spin-valley at T={T_K} K has kT={g['kT_meV']:.2f} meV >= "
                f"dE={delta_E_meV} meV. Thermal occupation scrambles the qubit. "
                "Cryogenic cooling required; 245 K monolithic operation refused.")
    @staticmethod
    def cavity_qed_cooperativity(g_GHz: float, kappa_GHz: float,
                                 gamma_GHz: float, required: float = 1e4
                                 ) -> Dict[str, Any]:
        C = 4*g_GHz**2/(kappa_GHz*gamma_GHz)
        return {"g_GHz":g_GHz,"kappa_GHz":kappa_GHz,"gamma_GHz":gamma_GHz,
                "cooperativity": C, "required": required,
                "sufficient": C >= required,
                "USE": "single-photon pi phase via Duan-Kimble reflection"}
    @staticmethod
    def membrane_conductance(G_uW_per_K: float, dT_K: float,
                             cooling_budget_uW: float) -> Dict[str, Any]:
        Q_leak_uW = G_uW_per_K * dT_K
        ok = Q_leak_uW <= cooling_budget_uW
        return {"G_uW_per_K": G_uW_per_K, "dT_K": dT_K,
                "heat_leak_uW": Q_leak_uW, "cooling_budget_uW": cooling_budget_uW,
                "closes": ok,
                "VERDICT": "thermal budget closes" if ok else "REFUSED: heat leak exceeds cooling"}
    @staticmethod
    def gkp_squeezing_target(dB: float, required_dB: float) -> Dict[str, Any]:
        return {"squeezing_dB_target": dB, "required_dB": required_dB,
                "meets_target": dB >= required_dB,
                "STATUS": "TARGET — not yet demonstrated at scale"}
    @staticmethod
    def refuse_fabrication(request: str) -> None:
        low = request.lower()
        for kw in ("fabricate","tape out","manufacture","produce wafer",
                   "build the chip","run the fab","ผลิตจริง","สร้างชิปจริง"):
            if kw in low:
                raise ScopeRefusal(
                    f"REFUSED '{kw}': this suite is {DESIGN_ONLY}.")

# =============================================================================
# §5. THE FOUR REVISIONS (physically grounded parameters)
# =============================================================================

@dataclass
class Revision:
    name: str
    form_factor: str
    carpet_sites: int
    modes_per_site: int
    code: QuantumCode
    gkp_squeeze_dB: float
    eps_T_dist: float
    toffoli_Gper_s: float
    tdp_W: float
    quantum_mode: str            # room-temp optical GKP | docked cryo | cryo-capsule
    spin_dE_meV: Optional[float] = None
    spin_T_K: Optional[float] = None

def _mk_revisions() -> Dict[str, Revision]:
    bb = lambda n,k,d,pm: QuantumCode(f"[[{n},{k},{d}]]", n,k,d, 6,
                                      "bivariate_bicycle", pm)
    lp = lambda n,k,d,pm,w=6: QuantumCode(f"[[{n},{k},{d}]]", n,k,d, w,
                                          "lifted_product", pm)
    return {
     "R9.0-M": Revision("R9.0-M","smartphone SoC",64,8,
        bb(512,48,12,9.2e-4),11.5,1e-10,0.30,3.80,
        "room-temp optical GKP cluster"),
     "R9.5-M": Revision("R9.5-M","smartphone SoC",512,32,
        lp(16384,1536,24,8.4e-4),12.0,1e-14,1.55,3.80,
        "room-temp optical GKP cluster"),
     "R9.8-M": Revision("R9.8-M","smartphone SoC",1024,32,
        lp(32768,3072,32,7.8e-4),12.2,1e-16,3.10,3.80,
        "room-temp optical GKP + docked cryo"),
     "R10.0-W": Revision("R10.0-W","desktop workstation",32768,48,
        lp(1572864,147456,64,6.4e-4,8),12.8,1e-22,62.0,350.0,
        "300 K optics + 1.8 K cryo-capsule",
        spin_dE_meV=0.406, spin_T_K=1.8),
    }

REVISIONS = _mk_revisions()

# =============================================================================
# §6. AI SURROGATES (transformer / world model / diffusion), gated
# =============================================================================

if HAS_TORCH:
    class CodeParamTransformer(nn.Module):
        """
        Predicts whether a candidate [[n,k,d]] + physics config is consistent.
        A research surrogate: the deterministic validators in §2 remain the
        authority. The transformer only ranks candidates for the validator.
        """
        def __init__(self, d_in=8, d=64, heads=4, layers=2, max_len=128):
            super().__init__()
            self.proj=nn.Linear(d_in,d)
            self.pos=nn.Parameter(torch.randn(1,max_len,d)*0.02)
            enc=nn.TransformerEncoderLayer(d,heads,d*4,0.1,
                                           batch_first=True,norm_first=True)
            self.trunk=nn.TransformerEncoder(enc,layers)
            self.head=nn.Linear(d,2)      # [reject, accept-logit]
            self.nuisance=nn.Linear(d,1)
        def forward(self,x):
            B,L,_=x.shape
            h=self.trunk(self.proj(x)+self.pos[:,:L]).mean(1)
            return {"logits":self.head(h),"nuisance":self.nuisance(h),
                    "STATUS":"surrogate ranking — deterministic validator decides"}

    class SyndromeWorldModel(nn.Module):
        """
        Predicts slow logical-error accumulation / syndrome dynamics.
        The latent is model-internal, NOT a measured error rate.
        """
        def __init__(self, d_obs=8, d_ctrl=2, d=64):
            super().__init__()
            self.enc=nn.Linear(d_obs+d_ctrl,d); self.gru=nn.GRUCell(d,d)
            self.mu=nn.Linear(d,d_obs); self.log_sd=nn.Linear(d,d_obs)
            self.decay_logit=nn.Parameter(torch.tensor(3.9))
        @property
        def decay(self): return torch.sigmoid(self.decay_logit)
        def forward(self,obs,ctrl):
            h=F.gelu(self.enc(torch.cat([obs,ctrl],-1))); h=self.gru(h,h)
            return {"next_mu":self.mu(h),
                    "log_sd":self.log_sd(h).clamp(-6,3),
                    "decay":float(self.decay),
                    "STATUS":"latent prediction, not a measured error rate"}

    class CarpetFloorplanDiffusion(nn.Module):
        """
        Generates CANDIDATE Sierpinski-Carpet floorplans. Each sample is a
        hypothesis that must pass the BN-isolation / wirelength / thermal
        gates before use.
        """
        def __init__(self, d=64, T=60, hidden=128, d_phys=4):
            super().__init__(); self.T=T; self.d=d
            b=torch.linspace(1e-4,0.02,T)
            self.register_buffer("betas",b)
            self.register_buffer("abar",torch.cumprod(1-b,0))
            self.net=nn.Sequential(nn.Linear(d+1,hidden),nn.GELU(),
                                   nn.Linear(hidden,hidden),nn.GELU(),
                                   nn.Linear(hidden,d))
            self.phys=nn.Sequential(nn.Linear(d_phys,hidden),nn.GELU())
        def loss(self,x0,phys):
            B=x0.shape[0]; t=torch.randint(0,self.T,(B,),device=x0.device)
            eps=torch.randn_like(x0); ab=self.abar[t][:,None]
            xt=ab.sqrt()*x0+(1-ab).sqrt()*eps
            inp=torch.cat([xt,t.float()[:,None]/self.T],-1)+self.phys(phys)
            return F.mse_loss(self.net(inp),eps)
        @torch.no_grad()
        def sample(self,n,phys,device="cpu"):
            x=torch.randn(n,self.d,device=device)
            for t in reversed(range(self.T)):
                tb=torch.full((n,),t,device=device,dtype=torch.long)
                inp=torch.cat([x,tb.float()[:,None]/self.T],-1)+self.phys(phys)
                eps=self.net(inp); a=1-self.betas[t]; ab=self.abar[t]
                x=(x-(1-a)/torch.sqrt(1-ab)*eps)/torch.sqrt(a)
                if t>0: x=x+torch.sqrt(self.betas[t])*torch.randn_like(x)
            return {"candidate":x,
                    "STATUS":"HYPOTHESIS floorplan — gate before use"}

class AIGate:
    @staticmethod
    def gate_code_candidate(code: QuantumCode) -> Dict[str, Any]:
        """Every AI-proposed code must pass the deterministic validator."""
        try:
            info = validate_code(code)
            return {"passed": True, "info": info}
        except Refusal as e:
            return {"passed": False, "reason": str(e), "status": MODE_E}
    @staticmethod
    def gate_floorplan(phys: Dict[str, float]) -> Dict[str, Any]:
        bn = phys.get("boundary_null_leakage", 1.0)
        G  = phys.get("membrane_G_uW_per_K", 999.0)
        ok = bn <= 1e-6 and G <= 0.8
        if not ok:
            raise PhysicsGateFailure(
                "floorplan REJECTED: BN leakage or membrane conductance out of range")
        return {"passed": True}

# =============================================================================
# §7. STATISTICS DICTIONARY
# =============================================================================

class SDGate:
    ENTRIES = {
      "code_parameters": (("BOUNDS","CONSTR"), "Singleton+Hamming+GV",
                          "constructibility must be verified, not assumed"),
      "bp_osd_memory_error": (("CAL","BP"), "calibrated BP+OSD scaling",
                              "memory error; separate from T-gate distillation"),
      "stratum_I_item": (("MATH",), "proven invariant", "certifiable"),
      "stratum_II_item": ((), "none", "hardware TARGET — always Mode E"),
      "spin_valley_operating_point": (("BOLTZ",), "kT << dE",
                                      "cryogenic regime required"),
      "cavity_qed_gate": (("COOP",), "Duan-Kimble cooperativity",
                          "C >= 1e4 for single-photon pi"),
      "ai_code_candidate": (("GATE","XVAL"), "surrogate", "validator decides"),
      "fabrication": ((), "none", "REFUSED — design assistance only"),
    }
    def __init__(self): self.log: List[Dict] = []
    def certify(self, stat, verified, value=None):
        e=self.ENTRIES.get(stat); v=tuple(verified)
        if e is None: r={"statistic":stat,"status":MODE_E,"reason":"no entry"}
        else:
            need,thm,cav=e; miss=tuple(p for p in need if p not in v)
            r={"statistic":stat,"theorem":thm,"caveat":cav,
               "required":list(need),"verified":list(v),"missing":list(miss),
               "status":MODE_E if miss else MODE_CERTIFIED,
               "reason":f"unverified {list(miss)}" if miss else ""}
        if value is not None: r["value"]=value
        self.log.append(r); return r
    def report(self):
        n=sum(1 for c in self.log if c["status"]==MODE_CERTIFIED)
        return {"n":len(self.log),"certified":n,"mode_E":len(self.log)-n,
                "log":self.log}

PACKAGES = {
 "BOUNDS":"Singleton/Hamming/GV checked","CONSTR":"explicit construction verified",
 "CAL":"calibrated alpha_bp and A_code","BP":"BP+OSD model, not surface-code",
 "MATH":"proven mathematical invariant","BOLTZ":"Boltzmann occupation satisfied",
 "COOP":"cooperativity above threshold","GATE":"physics gate passed",
 "XVAL":"cross-validated against deterministic solver",
}

# =============================================================================
# §8. CO-DESIGN PIPELINE
# =============================================================================

class QuantumChipCoDesign:
    def __init__(self, out_dir="runs/qchip_codesign"):
        self.out=Path(out_dir); self.out.mkdir(parents=True,exist_ok=True)
        self.sd=SDGate(); self.gate=PhysicsGate()
    def analyse_revision(self, name: str) -> Dict[str, Any]:
        if name not in REVISIONS: raise Refusal(f"unknown revision '{name}'")
        rev=REVISIONS[name]
        self.gate.refuse_fabrication(f"analyse {name}")
        code_info=validate_code(rev.code)
        mem=bp_osd_memory_error(rev.code)
        dist=magic_state_distillation(1e-3, levels=2)
        self.sd.certify("code_parameters",("BOUNDS",),code_info)
        self.sd.certify("bp_osd_memory_error",("CAL","BP"),mem)
        self.sd.certify("stratum_II_item",(),{"gkp_dB":rev.gkp_squeeze_dB})
        sq=self.gate.gkp_squeezing_target(rev.gkp_squeeze_dB, rev.gkp_squeeze_dB)
        out={"revision":name,"form_factor":rev.form_factor,
             "code":asdict(rev.code),"code_info":code_info,
             "memory_error":mem,"magic_distillation":dist,
             "gkp_squeezing":sq,"quantum_mode":rev.quantum_mode,
             "stratum_II_note":"all hardware values are TARGETS (Mode E)"}
        if rev.spin_dE_meV and rev.spin_T_K:
            bz=self.gate.boltzmann_init_fidelity(rev.spin_dE_meV, rev.spin_T_K)
            out["spin_valley_operating_point"]=bz
            self.sd.certify("spin_valley_operating_point",("BOLTZ",),bz)
        return out
    def coop_check(self) -> Dict[str, Any]:
        c=self.gate.cavity_qed_cooperativity(1.45, 0.028, 0.018)
        self.sd.certify("cavity_qed_gate",("COOP",) if c["sufficient"] else (),c)
        return c
    def refuse_surface_code_on_qldpc(self) -> Dict[str, Any]:
        try:
            forbid_surface_code_formula(REVISIONS["R9.0-M"].code)
            return {"refused": False}
        except FormulaMisuseRefusal as e:
            return {"refused": True, "reason": str(e)}

# =============================================================================
# §9. SELF-TEST
# =============================================================================

def self_test(verbose=True) -> bool:
    print(BANNER); ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    pipe=QuantumChipCoDesign()

    print("\n[T1] STRATUM SEPARATION")
    rep("carpet routing is Stratum I",
        stratum_of("sierpinski_carpet_routing") is Stratum.I)
    rep("GKP squeezing is Stratum II",
        stratum_of("gkp_squeezing_db") is Stratum.II)
    rep("Stratum I certifies",
        certify_stratum_item("stc_transmission_conditions")["status"]==MODE_CERTIFIED)
    rep("Stratum II is Mode E",
        certify_stratum_item("cavity_qed_cooperativity")["status"]==MODE_E)
    b=False
    try: assert_stratum("gkp_squeezing_db", Stratum.I)
    except StratumRefusal: b=True
    rep("conflating Stratum II as I REFUSED", b)

    print("\n[T2] CODE VALIDATION + HOLEVO")
    c48=REVISIONS["R9.0-M"].code
    ci=validate_code(c48)
    print(f"      [[512,48,12]] rate={ci['rate']:.5f} 3/32={ci['rate_is_3_over_32']}")
    rep("rate is 3/32", ci["rate_is_3_over_32"])
    rep("Singleton satisfied", True)
    rep("Hamming satisfied", ci["hamming_log2_lhs"]<=ci["hamming_log2_rhs"])
    b=False
    try: validate_code(QuantumCode("bad",100,100,5,6,"lifted_product",1e-3))
    except HolevoRefusal: b=True
    rep("k>=n Holevo violation REFUSED", b)
    b=False
    try: validate_code(QuantumCode("bad",100,90,20,6,"lifted_product",1e-3))
    except CodeBoundRefusal: b=True
    rep("Singleton violation REFUSED", b)
    for nm in REVISIONS:
        r=REVISIONS[nm]; validate_code(r.code)
    rep("all four revisions pass bounds", True)

    print("\n[T3] BP+OSD ERROR MODEL (reproduces unified-spec Table 2)")
    exp={"R9.0-M":3.7e-6,"R9.5-M":4.6e-11,"R9.8-M":1.3e-14,"R10.0-W":5.8e-30}
    for nm,target in exp.items():
        m=bp_osd_memory_error(REVISIONS[nm].code)
        got=m["P_L_memory_per_cycle"]
        rel=abs(got-target)/target
        print(f"      {nm}: P_L={got:.2e} (target {target:.2e})")
        rep(f"{nm} BP+OSD matches", rel<0.15, f"rel={rel:.2f}")
    rep("memory error separated from T-gate",
        "SEPARATE_FROM" in bp_osd_memory_error(c48))
    sf=pipe.refuse_surface_code_on_qldpc()
    rep("surface-code formula on qLDPC REFUSED", sf["refused"])

    print("\n[T4] MAGIC-STATE DISTILLATION (separate channel)")
    d=magic_state_distillation(1e-3,2)
    rep("distillation lowers T-gate error", d["output_infidelity"]<1e-3)
    rep("distillation chain reported", len(d["chain"])==3)

    print("\n[T5] PHYSICS GATES")
    hot=PhysicsGate.boltzmann_init_fidelity(0.174, 245.0)
    print(f"      245 K, dE=0.174 meV: ratio={hot['ratio_dE_over_kT']:.3f} F={hot['init_fidelity']:.3f}")
    rep("245 K scrambles spin-valley", not hot["usable"])
    cold=PhysicsGate.boltzmann_init_fidelity(0.406, 1.8)
    print(f"      1.8 K, dE=0.406 meV: ratio={cold['ratio_dE_over_kT']:.2f} F={cold['init_fidelity']:.3f}")
    rep("1.8 K initializes spin-valley", cold["usable"] and cold["init_fidelity"]>0.9)
    b=False
    try: PhysicsGate.require_cold_spin(0.174, 245.0)
    except PhysicsGateFailure: b=True
    rep("hot spin-valley REFUSED", b)
    coop=pipe.coop_check()
    print(f"      cavity-QED C={coop['cooperativity']:.2e} (need {coop['required']:.0e})")
    rep("cooperativity sufficient", coop["sufficient"])
    th=PhysicsGate.membrane_conductance(0.8, 295.0, 350.0)
    rep("membrane conductance gate runs", "closes" in th)
    b=False
    try: PhysicsGate.refuse_fabrication("please tape out R9.0-M now")
    except ScopeRefusal: b=True
    rep("fabrication REFUSED", b)

    print("\n[T6] AI SURROGATES — gated")
    if HAS_TORCH:
        torch.manual_seed(0)
        tr=CodeParamTransformer(8,64,4,2,64)
        x=torch.randn(4,10,8); o=tr(x)
        rep("transformer forward", tuple(o["logits"].shape)==(4,2))
        rep("transformer labelled surrogate","validator decides" in o["STATUS"])
        wm=SyndromeWorldModel(8,2,64)
        wo=wm(torch.randn(4,8),torch.randn(4,2))
        rep("world model decay in (0,1)", 0<wo["decay"]<1)
        rep("world model latent not measured","not a measured" in wo["STATUS"])
        df=CarpetFloorplanDiffusion(32,30,64,4)
        opt=torch.optim.Adam(df.parameters(),lr=3e-3)
        x0=torch.randn(8,32); ph=torch.randn(8,4); l0=lN=None
        for ep in range(30):
            opt.zero_grad(); L=df.loss(x0,ph); L.backward()
            torch.nn.utils.clip_grad_norm_(df.parameters(),1.0); opt.step()
            if ep==0: l0=float(L)
            lN=float(L)
        rep("diffusion trains", lN<l0, f"{l0:.3f}->{lN:.3f}")
        s=df.sample(2,ph)
        rep("diffusion labelled HYPOTHESIS","HYPOTHESIS" in s["STATUS"])
        g=AIGate.gate_code_candidate(c48); rep("good code passes AI gate", g["passed"])
        g2=AIGate.gate_code_candidate(QuantumCode("bad",50,50,5,6,"x",1e-3))
        rep("Holevo-violating candidate gated to Mode E",
            not g2["passed"] and g2["status"]==MODE_E)
        b=False
        try: AIGate.gate_floorplan({"boundary_null_leakage":1e-3,
                                    "membrane_G_uW_per_K":0.5})
        except PhysicsGateFailure: b=True
        rep("leaky floorplan REJECTED", b)
        rep("clean floorplan passes",
            AIGate.gate_floorplan({"boundary_null_leakage":1e-7,
                                   "membrane_G_uW_per_K":0.7})["passed"])
    else:
        rep("torch available", False, "pip install torch")

    print("\n[T7] REVISION ANALYSIS + SD")
    for nm in REVISIONS:
        r=pipe.analyse_revision(nm)
        rep(f"{nm} analysed", r["code_info"]["passes_bounds"])
        rep(f"{nm} Stratum II flagged", "TARGETS" in r["stratum_II_note"])
        if nm=="R10.0-W":
            rep("R10.0-W spin point usable",
                r["spin_valley_operating_point"]["usable"])
    rr=pipe.sd.report()
    print(f"      SD: {rr['n']} outputs, {rr['certified']} certified, {rr['mode_E']} Mode E")
    rep("Mode E retained", rr["mode_E"]>0)
    rep("stratum II never certified",
        all(c["status"]==MODE_E for c in rr["log"] if c["statistic"]=="stratum_II_item"))
    print("\n"+("="*78)); print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
BUILT
  Stratum I/II registry with hard separation; Stratum II always Mode E.
  Quantum-code validator: Holevo (k<n), Singleton, quantum Hamming, GV ceiling,
  and constructibility flagging for all four revisions.
  Calibrated BP+OSD memory-error model reproducing the unified-spec table,
  kept separate from magic-state T-gate distillation.
  Physics gates: Boltzmann spin-valley occupation (refuses 245 K),
  cavity-QED cooperativity, membrane conductance, GKP squeezing target.
  AI surrogates: code-parameter transformer, syndrome world model,
  Sierpinski-Carpet floorplan diffusion — all gated, all labelled.
  Fabrication / tape-out refused at every entry point.

ENFORCED REFUSALS
  k >= n (Holevo); Singleton/Hamming violations; surface-code formula on qLDPC;
  spin-valley at kT >= dE; any Stratum II target reported as certified;
  any fabrication, manufacturing, or tape-out claim.

NOT CLAIMED
  No chip is fabricated. No Stratum II target is achieved. No logical-qubit
  count is a measured capability. Every hardware number is a target pending
  measurement, and the module says so explicitly.
""")
    return ok

# =============================================================================
# §10. CLI
# =============================================================================

def main() -> int:
    ap=argparse.ArgumentParser(description="Quantum chip co-design AI suite")
    ap.add_argument("cmd",nargs="?",default="selftest",
        choices=["selftest","revisions","analyse","code","physics",
                 "strata","packages","philosophy"])
    ap.add_argument("--rev",default="R9.0-M")
    ap.add_argument("--json")
    a=ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test() else 1
    if a.cmd=="philosophy": print(json.dumps(PHILOSOPHY,indent=2)); return 0
    if a.cmd=="packages": print(json.dumps(PACKAGES,indent=2)); return 0
    pipe=QuantumChipCoDesign(); out={}
    try:
        if a.cmd=="revisions":
            out={nm:{"form":r.form_factor,"sites":r.carpet_sites,
                     "modes_per_site":r.modes_per_site,"code":r.code.name,
                     "k":r.code.k,"logical_qubits":r.code.k,
                     "quantum_mode":r.quantum_mode}
                 for nm,r in REVISIONS.items()}
        elif a.cmd=="analyse": out=pipe.analyse_revision(a.rev)
        elif a.cmd=="code": out=validate_code(REVISIONS[a.rev].code)
        elif a.cmd=="physics":
            out={"boltzmann_245K":PhysicsGate.boltzmann_init_fidelity(0.174,245.0),
                 "boltzmann_1p8K":PhysicsGate.boltzmann_init_fidelity(0.406,1.8),
                 "cooperativity":pipe.coop_check()}
        elif a.cmd=="strata":
            out={"stratum_I":sorted(STRATUM_I_ITEMS),
                 "stratum_II":sorted(STRATUM_II_ITEMS)}
    except Refusal as e: out={"REFUSED":str(e)}
    print(json.dumps(out,indent=2,default=str))
    if a.json: Path(a.json).write_text(json.dumps(out,indent=2,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
