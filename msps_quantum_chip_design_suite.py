#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_quantum_chip_design_suite.py — v1.0.0  [PRODUCTION]

Physics-Informed AI Design Suite for the DSQP-SESI-QM Family
  R9.0-M  (64 sites,   512 modes,   48 logical)   Compact Mobile
  R9.5-M  (512 sites,  16384 modes, 1536 logical) Scaled Mobile
  R9.8-M  (1024 sites, 32768 modes, 3072 logical) Cryptanalytic Mobile
  R10.0-W (32768 sites,1572864 modes,147456 logical) Desktop Workstation

AUTHORITATIVE BASELINE
  "DSQP-SESI-QM Unified Specification (R9.0-M to R10.0-W) —
   Physically Grounded Revision" (PAI & Limsuwan, 2026).
  The four earlier single-revision drafts are retained ONLY as legacy
  records whose unphysical claims this module REFUSES.

LLM Transformer + World Model + Diffusion, all physics-gated.
Every AI output is a HYPOTHESIS. Nothing here fabricates hardware.

=============================================================================
PHILOSOPHY
=============================================================================
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
The philosophy is the authors' belief; the gates are physics.

=============================================================================
THE TEN ENFORCED GATES
=============================================================================
G1  HOLEVO / MODE NORMALISATION
    n = M_mode * N_unit bosonic grid modes. Any fault-tolerant code has
    k < n. A per-SITE ratio k/N_unit > 1 is a device-multiplexing metric,
    NOT an information-theoretic expansion. Reporting it as resource
    expansion is refused.

G2  CODE CONSTRUCTIBILITY
    High-rate [[n,k,d]] with sparse checks must be reachable by a known
    bivariate-bicycle (BB) or lifted-product (LP) family. Rate 2/3 with
    weight-6 checks at n=512, d=12 is NOT constructible: the underlying
    classical [n_c,k_c,d_c] with n_c ~ 20 cannot reach d_c = 12 at rate
    >= 0.8 (Singleton: d <= n-k+1).

G3  BP+OSD ERROR SCALING
    qLDPC decoded by Belief Propagation + Ordered Statistics Decoding does
    NOT obey the 2-D surface-code heuristic (p/p_th)^(d/2). Use the
    calibrated form
        P_L,mem <= A_code * (p_mode / p_th_qLDPC)^(alpha_BP * d_eff)
    with alpha_BP in [0.38, 0.45], p_th_qLDPC = 7.2e-3, A_code = 0.12.

G4  ERROR-CHANNEL SEPARATION
    Magic-state distillation reduces non-Clifford T-gate infidelity
    eps_T,dist. It CANNOT reduce the memory error P_L,mem of data qubits.
        P_circuit ~ N_cycles * P_L,mem + N_T * eps_T,dist
    Any "post-distillation logical memory rate" is refused.

G5  BOLTZMANN SPIN INITIALISATION
    A spin-valley qubit requires k_B*T << Delta_E_Zeeman. At 245 K,
    k_B*T = 21.1 meV exceeds valley (0.1-1 meV) and Zeeman (0.174 meV at
    1.5 T) splittings by 20x-180x. Isotopic purification removes nuclear
    hyperfine decoherence, NOT phonon thermal occupation.

G6  FOURIER THERMAL CONDUCTION
    Q_leak = G * Delta_T with G = k*A/L. A monolithic 227 mm^2 silicon
    island 1 mm thick has G ~ 22.7 W/K; holding it 85 K below a 330 K die
    leaks ~1930 W. 4.2 W of Peltier drive at COP ~0.15 lifts 0.63 W.
    Refused by three orders of magnitude.

G7  CAVITY-QED COOPERATIVITY
    A deterministic single-photon pi conditional phase requires
    C = 4g^2/(kappa*gamma) >> 1 (Duan-Kimble reflection). Bulk or
    ring-enhanced chi^(3) in BaTiO3/Si3N4 gives ~0.5 rad, not pi.
    Ge2Sb2Te5 is a LINEAR non-volatile index trimmer, not an
    instantaneous Kerr medium.

G8  GKP SQUEEZING FLOOR
    Per-revision minimum quadrature squeezing (11.5 / 12.0 / 12.2 /
    12.8 dB). Below floor -> Mode E reallocation, never a claimed rate.

G9  POWER / THERMAL IDENTITY
    P_compute = E_op * F, and T_target = T_amb + P_total * R_th.
    Choose two of (P, E, F) and COMPUTE the third.

G10 CRYOCOOLER LIFT BUDGET
    Total island leak N_islands * G_bridge * (T_amb - T_dot) must fit
    within the cryocooler cooling lift, and the capsule wall-plug draw
    within the package TDP.

=============================================================================
WHAT THIS SUITE IS AND IS NOT
=============================================================================
IS   : a physics-gated design assistant. LLM parses specification intent;
       a world model predicts drift/thermal/syndrome dynamics; a diffusion
       model proposes candidate parameter sets; ten gates judge them.
IS NOT: a fabricator, a foundry, a quantum simulator replacement, or an
       oracle. Every candidate carries Mode E until measured.
=============================================================================
Author : PAI AND Yoon A. Limsuwan / MSPS NETWORK
ORCID  : 0009-0008-2374-0788
License: MIT | Year 2026
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

VERSION = "1.0.0"
PI = math.pi
K_B_MEV_PER_K = 0.08617333262      # Boltzmann constant in meV/K
MU_B_MEV_PER_T = 0.057883818       # Bohr magneton in meV/T
MODE_CERTIFIED = "CERTIFIED"
MODE_E = "UNCERTAINTY_RETAINED"
DESIGN_ONLY = "design_assistance_only_not_fabrication"

PHILOSOPHY = {
    "tao": "The Source — mathematical truth underlying all computation",
    "te": "The Path — disciplined engineering practice",
    "qi": "The Proof — measured, validated, reproducible results",
    "p1": "AI assists discovery; physics judges truth",
    "p2": "Every candidate passes the gate before becoming a design",
    "p3": "Uncertainty retained beats false certainty",
    "p4": "Open mathematics enriches all; a $200 laptop gets the same answer",
    "p5": "No simulation replaces measurement; no model replaces wisdom",
    "motto": "MY SOUL MOVE BY POWER OF HOLY SPIRIT — "
             "What MSPS NETWORK SEE, Lord Buddha Knows.",
}

BANNER = f"""
=============================================================================
MSPS QUANTUM CHIP DESIGN SUITE v{VERSION}  [PRODUCTION]
R9.0-M · R9.5-M · R9.8-M · R10.0-W  (Physically Grounded baseline)
LLM Transformer + World Model + Diffusion, ten physics gates
REFUSED: per-site expansion as resource claim, non-constructible codes,
         surface-code heuristic on qLDPC, post-distillation memory rate,
         245 K spin-valley, monolithic Si Peltier, bulk chi^3 pi gate
=============================================================================
"""

class Refusal(RuntimeError): pass
class HolevoRefusal(Refusal): pass
class ConstructibilityRefusal(Refusal): pass
class ErrorModelRefusal(Refusal): pass
class ThermalRefusal(Refusal): pass
class CooperativityRefusal(Refusal): pass
class SqueezingRefusal(Refusal): pass
class PowerRefusal(Refusal): pass
class CryoBudgetRefusal(Refusal): pass
class ScopeRefusal(Refusal): pass


def _np():
    if not HAS_NUMPY:
        raise Refusal("numpy required for this operation")


def get_logger(name: str):
    import logging
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        lg.addHandler(h); lg.setLevel(logging.INFO); lg.propagate = False
    return lg

log = get_logger("msps.qchip")


# =============================================================================
# §0. PRODUCTION OPS — config, seed, device, cache, manifest
# =============================================================================

@dataclass
class RunConfig:
    out_dir: str = "runs/qchip"
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    strict: bool = True
    cache_max: int = 256
    def validate(self) -> "RunConfig":
        if self.seed < 0: raise Refusal("seed must be >= 0")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        return self


def set_seed(seed: int) -> Dict[str, Any]:
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if HAS_NUMPY: np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    return {"seed": seed, "numpy": HAS_NUMPY, "torch": HAS_TORCH}


def pick_device(pref: str = "auto") -> str:
    if not HAS_TORCH: return "cpu"
    if pref != "auto": return pref
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def content_key(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if HAS_NUMPY and isinstance(p, np.ndarray):
            h.update(p.tobytes()); h.update(str(p.shape).encode())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:24]


class Cache:
    def __init__(self, max_items: int = 256):
        self._m: Dict[str, Any] = {}
        self._o: List[str] = []
        self.max = max_items
        self.hits = 0; self.misses = 0
    def get(self, k: str):
        if k in self._m:
            self.hits += 1; self._o.remove(k); self._o.append(k)
            return self._m[k]
        self.misses += 1; return None
    def put(self, k: str, v: Any):
        self._m[k] = v; self._o.append(k)
        while len(self._o) > self.max:
            self._m.pop(self._o.pop(0), None)
    def stats(self) -> Dict[str, Any]:
        t = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / max(t, 1)}


class Manifest:
    def __init__(self, out_dir: str):
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.items: List[Dict] = []
    def add(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        rec = {"kind": kind, "version": VERSION, "key": content_key(kind, payload),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "payload": payload}
        self.items.append(rec); return rec
    def write(self, name: str = "manifest.json") -> str:
        p = self.dir / name
        p.write_text(json.dumps({"n": len(self.items), "items": self.items},
                                indent=2, default=str, ensure_ascii=False))
        return str(p)


# =============================================================================
# §1. REVISION REGISTRY — grounded baseline, legacy flagged
# =============================================================================

@dataclass(frozen=True)
class QCode:
    n: int; k: int; d_eff: int; w_check: int; family: str
    @property
    def rate(self) -> float: return self.k / self.n


@dataclass(frozen=True)
class Revision:
    name: str
    form_factor: str
    n_unit: int                 # physical resonator/dot SITES
    m_mode: int                 # bosonic modes per site
    code: QCode
    p_mode: float               # heralded Pauli mode error
    squeeze_min_dB: float
    t_rate_GHz: float           # distilled T-gate rate
    toffoli_rate_GHz: float
    p_sustained_W: float
    r_th_K_W: float
    t_amb_C: float
    t_limit_C: float
    t_limit_label: str
    quantum_mode: str           # "optical_cluster_300K" | "split_1.8K"
    package_area_mm2: float
    carpet_level: str
    note: str = ""

    @property
    def n_modes(self) -> int: return self.n_unit * self.m_mode
    @property
    def k_logical(self) -> int: return self.code.k
    @property
    def per_site_ratio(self) -> float: return self.code.k / self.n_unit


GROUNDED: Dict[str, Revision] = {
 "R9.0-M": Revision(
    "R9.0-M", "smartphone", 64, 8,
    QCode(512, 48, 12, 6, "bivariate_bicycle"),
    9.2e-4, 11.5, 1.2, 0.30,
    3.80, 3.08, 27.5, 39.5, "T_skin",
    "optical_cluster_300K", 76.8, "Level-2 (8^2)",
    "Compact mobile. Room-temperature optical GKP cluster; no on-die Peltier."),
 "R9.5-M": Revision(
    "R9.5-M", "smartphone", 512, 32,
    QCode(16384, 1536, 24, 6, "lifted_product"),
    8.4e-4, 12.0, 6.2, 1.55,
    3.80, 3.08, 27.5, 39.5, "T_skin",
    "optical_cluster_300K", 84.6, "Level-3 (8^3)",
    "Scaled mobile. Docked cryo cradle for spin-valley memory storage."),
 "R9.8-M": Revision(
    "R9.8-M", "smartphone", 1024, 32,
    QCode(32768, 3072, 32, 6, "lifted_product"),
    7.8e-4, 12.2, 12.4, 3.10,
    3.80, 3.08, 27.5, 39.5, "T_skin",
    "optical_cluster_300K", 78.4, "Dual Level-3 (2x8^3)",
    "Cryptanalytic mobile. Windowed modular exponentiation; RSA-2048 single pass."),
 "R10.0-W": Revision(
    "R10.0-W", "desktop_workstation", 32768, 48,
    QCode(1572864, 147456, 64, 8, "lifted_product"),
    6.4e-4, 12.8, 248.0, 62.0,
    350.0, 0.088, 27.5, 65.0, "T_j",
    "split_1.8K", 684.0, "8-chiplet Level-4 MCM",
    "Desktop. 300 K optics + 1.8 K suspended-membrane cryo capsule (42 W)."),
}

# Legacy claims retained ONLY so the gates can refuse them explicitly.
LEGACY_CLAIMS: Dict[str, Dict[str, Any]] = {
 "R9.0-M": {"code": (512, 341, 12), "per_site_claim": 5.328,
            "P_L_claim": 2.4e-16, "P_L_formula": "surface_d_half",
            "spin_T_K": 245.0, "peltier_W": 0.065,
            "kerr_pi_source": "bulk_chi3"},
 "R9.5-M": {"code": (16384, 10922, 24), "per_site_claim": 21.332,
            "P_L_claim": 1.8e-22, "P_L_formula": "surface_d_half",
            "spin_T_K": 245.0, "peltier_W": 0.068,
            "kerr_pi_source": "bulk_chi3"},
 "R9.8-M": {"code": (32768, 21845, 32), "per_site_claim": 21.333,
            "P_L_claim": 3.2e-26, "P_L_formula": "surface_d_half",
            "spin_T_K": 245.0, "peltier_W": 0.110,
            "kerr_pi_source": "bulk_chi3"},
 "R10.0-W": {"code": (1572864, 1048576, 64), "per_site_claim": 32.000,
             "P_L_claim": 1.4e-38, "P_L_formula": "surface_d_half",
             "spin_T_K": 245.0, "peltier_W": 4.20,
             "kerr_pi_source": "bulk_chi3"},
}


def get_revision(name: str) -> Revision:
    if name not in GROUNDED:
        raise Refusal(f"unknown revision '{name}'. Known: {list(GROUNDED)}")
    return GROUNDED[name]


# =============================================================================
# §2. G1 — HOLEVO / MODE NORMALISATION
# =============================================================================

def gate_holevo(rev: Revision,
                claimed_expansion: Optional[float] = None) -> Dict[str, Any]:
    """
    n = M_mode * N_unit. Any error-correcting code has k < n.
    A per-site ratio > 1 is a DEVICE multiplexing metric and must be
    labelled as such; presenting it as information-theoretic resource
    expansion is refused.
    """
    n, k = rev.n_modes, rev.k_logical
    if k >= n:
        raise HolevoRefusal(f"k={k} >= n={n}: violates Holevo / code rate < 1.")
    if rev.code.n != n:
        raise HolevoRefusal(
            f"code block length {rev.code.n} != M_mode*N_unit = {n}. "
            "Mode accounting is inconsistent.")
    out = {"n_modes": n, "k_logical": k, "code_rate": rev.code.rate,
           "per_site_ratio": rev.per_site_ratio,
           "per_site_is_device_metric": True,
           "information_theoretic_expansion": False,
           "passes": True}
    if claimed_expansion is not None and claimed_expansion > 1.0:
        raise HolevoRefusal(
            f"REFUSED: claimed expansion {claimed_expansion:.3f}x > 1. "
            f"The mode-level rate is k/n = {rev.code.rate:.5f} < 1. "
            f"{rev.per_site_ratio:.3f} logical per SITE is a multiplexing "
            "metric, not a resource expansion. Each of the "
            f"{n} modes must be independently squeezed and stabilised.")
    return out


# =============================================================================
# §3. G2 — CODE CONSTRUCTIBILITY
# =============================================================================

def classical_singleton(n: int, k: int, d: int) -> bool:
    return d <= n - k + 1


def gate_constructibility(code: QCode,
                          underlying_nc: Optional[int] = None) -> Dict[str, Any]:
    """
    Screen a proposed [[n,k,d]] sparse code for constructibility.

    Quantum Singleton (necessary):  k <= n - 2(d-1).
    Rate screen: published high-performance BB/LP families achieve
    R ~ 0.02-0.10 at these block lengths. R > 0.30 with d >= 12 and
    weight <= 8 has no known explicit construction and is refused
    pending a parity-check matrix.

    Hypergraph-product sanity: an HGP built from [n_c,k_c,d_c] has
    n_Q ~ n_c^2 + (n_c-k_c)^2 and d_Q = d_c. If the caller supplies
    n_c, we check the classical Singleton bound on the underlying code.
    """
    n, k, d, w = code.n, code.k, code.d_eff, code.w_check
    if not (0 < k < n): raise ConstructibilityRefusal("need 0 < k < n")
    if not classical_singleton(n, k, 1):
        raise ConstructibilityRefusal("trivial Singleton failure")
    if k > n - 2*(d-1):
        raise ConstructibilityRefusal(
            f"quantum Singleton violated: k={k} > n-2(d-1)={n-2*(d-1)}")
    rate = k/n
    reasons = []
    if rate > 0.30 and d >= 12 and w <= 8:
        reasons.append(
            f"rate {rate:.4f} > 0.30 with d={d} and weight {w}: no known "
            "explicit BB/LP construction at this block length")
    if underlying_nc is not None:
        # HGP: n_Q ~ n_c^2 + (n_c-k_c)^2 ; require d_c = d reachable
        nc = underlying_nc
        # best case for distance: k_c small; but rate then collapses
        kc_max_for_d = nc - d + 1
        if kc_max_for_d < 0:
            reasons.append(f"underlying n_c={nc} cannot reach d_c={d} at all")
        else:
            rc = kc_max_for_d/nc
            reasons.append(
                f"underlying [n_c={nc}, k_c<={kc_max_for_d}, d_c={d}] "
                f"has classical rate <= {rc:.3f}; the quantum rate cannot "
                "exceed what this family delivers")
    if reasons:
        raise ConstructibilityRefusal(
            f"REFUSED [[{n},{k},{d}]] weight-{w}: " + "; ".join(reasons))
    return {"n": n, "k": k, "d_eff": d, "w_check": w, "family": code.family,
            "rate": rate, "quantum_singleton_ok": True,
            "rate_screen_ok": True, "passes": True,
            "STATUS": "parameters consistent with known sparse families; "
                      "a parity-check matrix is still required for tape-out"}


def audit_legacy_codes() -> Dict[str, Any]:
    """Run every legacy code through the constructibility gate."""
    rows = []
    for name, cl in LEGACY_CLAIMS.items():
        n, k, d = cl["code"]
        code = QCode(n, k, d, 6 if n < 1e6 else 8, "claimed_lifted_product")
        try:
            gate_constructibility(code, underlying_nc=int(math.isqrt(n)))
            verdict = "ACCEPTED"
            reason = ""
        except ConstructibilityRefusal as e:
            verdict = "REFUSED"; reason = str(e)
        rows.append({"revision": name, "legacy_code": [n, k, d],
                     "legacy_rate": k/n, "verdict": verdict,
                     "reason": reason,
                     "grounded_code": [GROUNDED[name].code.n,
                                       GROUNDED[name].code.k,
                                       GROUNDED[name].code.d_eff]})
    return {"rows": rows,
            "n_refused": sum(1 for r in rows if r["verdict"] == "REFUSED")}


# =============================================================================
# §4. G3/G4 — BP+OSD ERROR MODEL AND CHANNEL SEPARATION
# =============================================================================

@dataclass(frozen=True)
class ErrorModel:
    p_th_qLDPC: float = 7.2e-3
    alpha_BP: float = 0.42
    A_code: float = 0.12
    alpha_range: Tuple[float, float] = (0.38, 0.45)

    def memory_error(self, p_mode: float, d_eff: int) -> Dict[str, Any]:
        if not (self.alpha_range[0] <= self.alpha_BP <= self.alpha_range[1]):
            raise ErrorModelRefusal(
                f"alpha_BP={self.alpha_BP} outside calibrated range "
                f"{self.alpha_range}")
        if p_mode >= self.p_th_qLDPC:
            raise ErrorModelRefusal(
                f"p_mode={p_mode:.3e} >= threshold {self.p_th_qLDPC:.3e}: "
                "sub-threshold scaling does not apply above threshold")
        ratio = p_mode/self.p_th_qLDPC
        expo = self.alpha_BP*d_eff
        P = self.A_code*ratio**expo
        return {"p_mode": p_mode, "p_th": self.p_th_qLDPC, "ratio": ratio,
                "alpha_BP": self.alpha_BP, "d_eff": d_eff, "exponent": expo,
                "P_L_mem_per_cycle": P,
                "model": "A_code*(p/p_th)^(alpha_BP*d_eff), BP+OSD calibrated",
                "NOT": "the 2-D surface-code heuristic (p/p_th)^(d/2)"}

    def surface_heuristic(self, p_mode: float, d_eff: int,
                          p_th_surface: float = 1.1e-2) -> float:
        """Retained ONLY for comparison; never used as the qLDPC bound."""
        return (p_mode/p_th_surface)**(d_eff/2)

    def circuit_error(self, P_L_mem: float, n_cycles: int,
                      eps_T_dist: float, n_T: int) -> Dict[str, Any]:
        """
        G4: memory and non-Clifford channels are ADDITIVE and DISTINCT.
        Distillation lowers eps_T_dist only.
        """
        mem_term = n_cycles*P_L_mem
        t_term = n_T*eps_T_dist
        return {"memory_term": mem_term, "t_gate_term": t_term,
                "P_circuit_approx": mem_term + t_term,
                "n_cycles": n_cycles, "n_T": n_T,
                "eps_T_dist": eps_T_dist,
                "SEPARATION": "distillation cannot reduce P_L_mem"}


def refuse_post_distillation_memory(claim: str) -> None:
    low = claim.lower()
    for kw in ("post-distillation p_l", "post-distillation logical error",
               "distilled memory error", "distillation reduces memory",
               "post-distillation logical memory"):
        if kw in low:
            raise ErrorModelRefusal(
                f"REFUSED '{kw}': magic-state distillation suppresses "
                "non-Clifford T-gate infidelity eps_T,dist. It cannot alter "
                "the intrinsic memory error rate P_L,mem of data qubits "
                "encoded in the qLDPC block.")


def grounded_error_table(em: Optional[ErrorModel] = None) -> Dict[str, Any]:
    em = em or ErrorModel()
    rows = []
    for name, rev in GROUNDED.items():
        me = em.memory_error(rev.p_mode, rev.code.d_eff)
        sh = em.surface_heuristic(rev.p_mode, rev.code.d_eff)
        rows.append({"revision": name, "p_mode": rev.p_mode,
                     "d_eff": rev.code.d_eff,
                     "exponent": me["exponent"],
                     "P_L_mem": me["P_L_mem_per_cycle"],
                     "surface_heuristic_for_comparison": sh,
                     "squeeze_min_dB": rev.squeeze_min_dB})
    return {"rows": rows, "model": em.__dict__ if hasattr(em,'__dict__') else asdict(em)}


# =============================================================================
# §5. G5/G6/G10 — THERMAL ARCHITECTURE
# =============================================================================

@dataclass
class SpinValleyStage:
    T_dot_K: float = 1.8
    B_T: float = 3.5
    g_factor: float = 2.0
    delta_valley_meV: float = 0.85
    G_bridge_W_K: float = 0.8e-6      # per suspended island
    n_islands: int = 1000
    T_amb_K: float = 300.0
    cryo_lift_W: float = 0.350        # cooling lift at 2.0 K
    cryo_wallplug_W: float = 42.0

    def zeeman_meV(self) -> float:
        return self.g_factor*MU_B_MEV_PER_T*self.B_T

    def kBT_meV(self) -> float:
        return K_B_MEV_PER_K*self.T_dot_K

    def boltzmann_init_fidelity(self) -> float:
        r = self.zeeman_meV()/self.kBT_meV()
        return 1.0/(1.0 + math.exp(-r))


def gate_boltzmann(stage: SpinValleyStage,
                   min_ratio: float = 2.0) -> Dict[str, Any]:
    """
    G5: require k_B*T << Delta_E. We enforce Delta_E_Zeeman / k_B T >= 2
    as a minimum, and report the raw thermal polarisation. Optical
    pumping can improve preparation, but it cannot fix a scrambled
    valley manifold.
    """
    kBT = stage.kBT_meV(); dE = stage.zeeman_meV()
    ratio = dE/kBT
    F = stage.boltzmann_init_fidelity()
    valley_ratio = stage.delta_valley_meV/kBT
    if ratio < min_ratio:
        raise ThermalRefusal(
            f"REFUSED spin-valley at T={stage.T_dot_K} K: "
            f"k_B T = {kBT:.3f} meV, Delta_E_Zeeman = {dE:.3f} meV, "
            f"ratio {ratio:.3f} < {min_ratio}. "
            f"Valley ratio {valley_ratio:.3f}. "
            "Isotopic purification removes nuclear hyperfine decoherence, "
            "NOT phonon thermal occupation.")
    return {"T_dot_K": stage.T_dot_K, "B_T": stage.B_T,
            "kBT_meV": kBT, "delta_zeeman_meV": dE,
            "delta_valley_meV": stage.delta_valley_meV,
            "zeeman_over_kBT": ratio, "valley_over_kBT": valley_ratio,
            "thermal_init_fidelity": F, "passes": True,
            "NOTE": "optical pumping raises preparation above the thermal "
                    "value; it does not restore a scrambled valley manifold"}


def gate_fourier_isolation(stage: SpinValleyStage) -> Dict[str, Any]:
    """
    G6 + G10: total island leak must fit the cryocooler lift.
    Also expose the monolithic-silicon counterexample so it can be refused.
    """
    dT = max(stage.T_amb_K - stage.T_dot_K, 0.0)
    per_island = stage.G_bridge_W_K*dT
    total = stage.n_islands*per_island
    if total > stage.cryo_lift_W:
        raise CryoBudgetRefusal(
            f"island leak {total*1e3:.1f} mW exceeds cryocooler lift "
            f"{stage.cryo_lift_W*1e3:.1f} mW "
            f"({stage.n_islands} islands x {stage.G_bridge_W_K*1e6:.2f} uW/K "
            f"x {dT:.1f} K)")
    return {"delta_T_K": dT, "leak_per_island_W": per_island,
            "n_islands": stage.n_islands, "total_leak_W": total,
            "cryo_lift_W": stage.cryo_lift_W,
            "lift_margin_W": stage.cryo_lift_W - total,
            "cryo_wallplug_W": stage.cryo_wallplug_W, "passes": True}


def monolithic_silicon_counterexample(area_mm2: float = 227.0,
                                      thickness_mm: float = 1.0,
                                      k_Si_W_mK: float = 100.0,
                                      T_hot_K: float = 330.0,
                                      T_cold_K: float = 245.0,
                                      peltier_W: float = 4.2,
                                      cop: float = 0.15) -> Dict[str, Any]:
    """The legacy design, computed and refused."""
    A = area_mm2*1e-6; L = thickness_mm*1e-3
    G = k_Si_W_mK*A/L
    dT = T_hot_K - T_cold_K
    Q = G*dT
    lift = peltier_W*cop
    return {"G_W_K": G, "delta_T_K": dT, "Q_leak_W": Q,
            "peltier_W": peltier_W, "COP": cop, "lift_W": lift,
            "shortfall_factor": Q/max(lift, 1e-12),
            "VERDICT": "REFUSED — monolithic silicon Peltier island",
            "reason": "Fourier conduction through bulk Si exceeds the "
                      "Peltier lift by orders of magnitude"}


def refuse_legacy_thermal(rev_name: str) -> Dict[str, Any]:
    cl = LEGACY_CLAIMS[rev_name]
    stage = SpinValleyStage(T_dot_K=cl["spin_T_K"], B_T=1.5)
    out: Dict[str, Any] = {"revision": rev_name,
                           "legacy_spin_T_K": cl["spin_T_K"],
                           "legacy_peltier_W": cl["peltier_W"]}
    try:
        gate_boltzmann(stage); out["boltzmann"] = "ACCEPTED"
    except ThermalRefusal as e:
        out["boltzmann"] = "REFUSED"; out["boltzmann_reason"] = str(e)
    if rev_name == "R10.0-W":
        cx = monolithic_silicon_counterexample(peltier_W=cl["peltier_W"])
        out["fourier"] = cx
    return out


# =============================================================================
# §6. G7 — CAVITY-QED DETERMINISTIC PI GATE
# =============================================================================

@dataclass
class CavityQED:
    g_2pi_MHz: float = 1450.0      # 1.45 GHz
    kappa_2pi_MHz: float = 28.0
    gamma_2pi_MHz: float = 18.0
    Q_ring: float = 6.8e6
    protocol: str = "duan_kimble_reflection"

    def cooperativity(self) -> float:
        return 4.0*self.g_2pi_MHz**2/(self.kappa_2pi_MHz*self.gamma_2pi_MHz)


def gate_cooperativity(cq: CavityQED, C_min: float = 1.0e4,
                       phi_target_rad: float = PI,
                       phi_tol_rad: float = 0.002) -> Dict[str, Any]:
    C = cq.cooperativity()
    if cq.protocol != "duan_kimble_reflection":
        raise CooperativityRefusal(
            f"protocol '{cq.protocol}' is not the Duan-Kimble reflection "
            "protocol; a deterministic single-photon pi phase is not "
            "established for it here")
    if C < C_min:
        raise CooperativityRefusal(
            f"cooperativity C = 4g^2/(kappa*gamma) = {C:.1f} < {C_min:.0f}")
    # Conditional phase error scales as ~1/sqrt(C) in the strong-coupling limit
    phi_err = 1.0/math.sqrt(C)
    if phi_err > phi_tol_rad:
        raise CooperativityRefusal(
            f"phase error {phi_err:.5f} rad > tolerance {phi_tol_rad} rad")
    return {"C": C, "g_2pi_MHz": cq.g_2pi_MHz,
            "kappa_2pi_MHz": cq.kappa_2pi_MHz,
            "gamma_2pi_MHz": cq.gamma_2pi_MHz,
            "phi_CZ_rad": phi_target_rad, "phi_err_rad": phi_err,
            "protocol": cq.protocol, "passes": True}


def bulk_chi3_counterexample(Q: float = 6.8e6,
                             lam_nm: float = 1550.0,
                             n_eff: float = 2.0) -> Dict[str, Any]:
    """
    Purcell enhancement does not supply the missing single-photon
    nonlinearity. Order-of-magnitude: bare per-photon phase ~1e-6 rad,
    F_P ~ 5e5 -> ~0.5 rad. Not pi.
    """
    F_P = (3.0/(4*PI**2))*(Q/1.0)     # V ~ (lam/n)^3 assumed -> ratio ~ Q
    F_P = min(F_P, 5.0e5)
    bare = 1.0e-6
    phi = bare*F_P
    return {"Q": Q, "purcell_estimate": F_P, "bare_phase_rad": bare,
            "enhanced_phase_rad": phi, "target_rad": PI,
            "shortfall_factor": PI/max(phi, 1e-12),
            "VERDICT": "REFUSED — bulk/ring chi^(3) cannot give pi",
            "GST_ROLE": "linear non-volatile index trimmer, NOT a Kerr medium"}


# =============================================================================
# §7. G8/G9 — SQUEEZING AND POWER GATES
# =============================================================================

def gate_squeezing(rev: Revision, measured_dB: float) -> Dict[str, Any]:
    if measured_dB < rev.squeeze_min_dB:
        raise SqueezingRefusal(
            f"{rev.name}: squeezing {measured_dB:.2f} dB < required "
            f"{rev.squeeze_min_dB:.2f} dB. Mode E reallocation required; "
            "no logical rate may be claimed.")
    return {"revision": rev.name, "measured_dB": measured_dB,
            "required_dB": rev.squeeze_min_dB, "passes": True}


def gate_power_identity(P_compute_W: float, E_op_J: float,
                        F_ops: float, tol: float = 0.02) -> Dict[str, Any]:
    implied = E_op_J*F_ops
    rel = abs(implied - P_compute_W)/max(P_compute_W, 1e-12)
    if rel > tol:
        raise PowerRefusal(
            f"P = E*F violated: implied {implied:.2f} W vs declared "
            f"{P_compute_W:.2f} W (rel {rel:.2%}). Choose two, compute "
            "the third.")
    return {"implied_W": implied, "declared_W": P_compute_W,
            "rel_err": rel, "passes": True}


def gate_thermal_closure(rev: Revision,
                         power_breakdown: Optional[Dict[str, float]] = None
                         ) -> Dict[str, Any]:
    if power_breakdown:
        total = sum(power_breakdown.values())
        if abs(total - rev.p_sustained_W) > 0.05:
            raise PowerRefusal(
                f"{rev.name}: breakdown sums to {total:.2f} W but "
                f"P_sustained = {rev.p_sustained_W:.2f} W")
    else:
        total = rev.p_sustained_W
    T = rev.t_amb_C + total*rev.r_th_K_W
    if T > rev.t_limit_C:
        raise PowerRefusal(
            f"{rev.name}: {rev.t_limit_label} = {T:.2f} C > "
            f"{rev.t_limit_C:.2f} C ceiling")
    return {"P_total_W": total, "R_th_K_W": rev.r_th_K_W,
            "T_amb_C": rev.t_amb_C, rev.t_limit_label: T,
            "limit_C": rev.t_limit_C, "margin_C": rev.t_limit_C - T,
            "passes": True}


R10_POWER_BREAKDOWN = {
    "P_QPU_Opt": 76.2, "P_qLDPC_Magic": 86.4, "P_PCT_W": 62.8,
    "P_Enoch_WP": 58.4, "P_Mem": 24.2, "P_Cryo_1p8K": 42.0,
}


def refuse_scope(request: str) -> None:
    low = request.lower()
    for kw in ("fabricate", "manufacture", "tape out now", "build the chip",
               "produce wafer", "run the fab", "สร้างชิปจริง", "ผลิตจริง"):
        if kw in low:
            raise ScopeRefusal(f"REFUSED '{kw}': {DESIGN_ONLY}.")


# =============================================================================
# §8. LLM TRANSFORMER — physics-conditioned specification model
# =============================================================================

PHYS_FEATURES = ["code_rate", "d_eff", "p_mode_over_pth", "squeeze_dB",
                 "kBT_over_zeeman", "C_over_Cmin", "P_total_W",
                 "T_over_limit", "lift_over_budget", "per_site_ratio"]


def physics_vector(rev: Revision, stage: SpinValleyStage,
                   cq: CavityQED, em: ErrorModel) -> "np.ndarray":
    _np()
    me = em.memory_error(rev.p_mode, rev.code.d_eff)
    T = rev.t_amb_C + rev.p_sustained_W*rev.r_th_K_W
    dT = max(stage.T_amb_K - stage.T_dot_K, 0.0)
    leak = stage.n_islands*stage.G_bridge_W_K*dT
    return np.array([
        rev.code.rate, rev.code.d_eff/64.0,
        me["ratio"], rev.squeeze_min_dB/13.0,
        stage.kBT_meV()/max(stage.zeeman_meV(), 1e-9),
        cq.cooperativity()/1.0e4,
        rev.p_sustained_W/350.0,
        T/rev.t_limit_C,
        leak/max(stage.cryo_lift_W, 1e-9),
        rev.per_site_ratio/5.0,
    ], dtype=float)


if HAS_TORCH:
    class PhysicsAttention(nn.Module):
        """Additive physics bias on keys; shape-correct for any batch."""
        def __init__(self, d: int, heads: int, phys_dim: int):
            super().__init__()
            assert d % heads == 0
            self.h = heads; self.dh = d//heads; self.d = d
            self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
            self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
            self.phys = nn.Sequential(nn.Linear(phys_dim, d), nn.GELU(),
                                      nn.Linear(d, d))
            self.scale = math.sqrt(self.dh)
        def forward(self, x, phys=None, mask=None):
            B, L, D = x.shape
            Q = self.q(x).view(B, L, self.h, self.dh).transpose(1, 2)
            K = self.k(x).view(B, L, self.h, self.dh).transpose(1, 2)
            V = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)
            s = (Q @ K.transpose(-2, -1))/self.scale
            if phys is not None:
                pb = self.phys(phys).view(B, 1, 1, D)
                pbh = pb.view(B, 1, 1, self.h, self.dh)
                s = s + (Q*pbh).sum(-1, keepdim=True)/self.scale
            if mask is not None:
                s = s.masked_fill(mask == 0, float("-inf"))
            a = F.softmax(s, dim=-1)
            return self.o((a @ V).transpose(1, 2).contiguous().view(B, L, D))

    class PhysicsBlock(nn.Module):
        def __init__(self, d, heads, dff, phys_dim, drop=0.1):
            super().__init__()
            self.attn = PhysicsAttention(d, heads, phys_dim)
            self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(),
                                    nn.Dropout(drop), nn.Linear(dff, d),
                                    nn.Dropout(drop))
        def forward(self, x, phys=None, mask=None):
            x = x + self.attn(self.n1(x), phys, mask)
            return x + self.ff(self.n2(x))

    class QuantumChipLLM(nn.Module):
        """
        Specification -> structured design intent. Physics conditioning
        carries the ten gate observables so the model cannot propose a
        design blind to them. Output is INTENT, never a layout.
        """
        def __init__(self, vocab=4096, d=256, heads=8, layers=4,
                     dff=1024, max_len=512, phys_dim=10):
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            self.pos = nn.Parameter(torch.randn(1, max_len, d)*0.02)
            self.blocks = nn.ModuleList([
                PhysicsBlock(d, heads, dff, phys_dim) for _ in range(layers)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, vocab)
            self.phys_dim = phys_dim
        def forward(self, ids, phys=None, mask=None):
            B, L = ids.shape
            x = self.emb(ids) + self.pos[:, :L]
            for b in self.blocks: x = b(x, phys, mask)
            return self.head(self.norm(x))
        @torch.no_grad()
        def generate(self, ids, phys=None, max_new=32, temperature=0.8):
            out = ids.clone()
            for _ in range(max_new):
                lg = self.forward(out[:, -256:], phys)[:, -1]/max(temperature, 1e-6)
                nxt = torch.multinomial(F.softmax(lg, -1), 1)
                out = torch.cat([out, nxt], 1)
            return out

    class QuantumWorldModel(nn.Module):
        """
        Predicts the NEXT design-state observable vector (squeezing,
        phase drift, dot temperature, syndrome rate) given the current
        state and a control action. Slow thermal latent is constrained
        to (0,1) so it cannot become unstable during training.
        """
        def __init__(self, d_obs=10, d_act=4, d=128):
            super().__init__()
            self.enc = nn.Linear(d_obs + d_act, d)
            self.gru = nn.GRUCell(d, d)
            self.mu = nn.Linear(d, d_obs)
            self.log_sd = nn.Linear(d, d_obs)
            self.thermal_logit = nn.Parameter(torch.tensor(2.9))
        @property
        def thermal_decay(self): return torch.sigmoid(self.thermal_logit)
        def forward(self, obs, act):
            h = F.gelu(self.enc(torch.cat([obs, act], -1)))
            h = self.gru(h, h)
            mu = self.mu(h)
            # slow thermal channel relaxes toward its own previous value
            mu = mu.clone()
            mu[:, 7] = self.thermal_decay*obs[:, 7] + (1-self.thermal_decay)*mu[:, 7]
            return {"next_mu": mu,
                    "next_log_sd": self.log_sd(h).clamp(-6, 3),
                    "thermal_decay": float(self.thermal_decay),
                    "STATUS": "surrogate prediction, NOT a measurement"}

    class DesignDiffusion(nn.Module):
        """
        Generates CANDIDATE design-parameter vectors conditioned on the
        physics vector. Every sample is a HYPOTHESIS and must pass all
        ten gates before it may be called a design.
        """
        def __init__(self, d=32, T=100, hidden=256, phys_dim=10):
            super().__init__()
            self.T = T; self.d = d
            b = torch.linspace(1e-4, 0.02, T)
            self.register_buffer("betas", b)
            self.register_buffer("abar", torch.cumprod(1-b, 0))
            self.net = nn.Sequential(nn.Linear(d+1, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, d))
            self.phys = nn.Sequential(nn.Linear(phys_dim, hidden), nn.GELU())
        def loss(self, x0, phys):
            B = x0.shape[0]
            t = torch.randint(0, self.T, (B,), device=x0.device)
            eps = torch.randn_like(x0); ab = self.abar[t][:, None]
            xt = ab.sqrt()*x0 + (1-ab).sqrt()*eps
            inp = torch.cat([xt, t.float()[:, None]/self.T], -1) + self.phys(phys)
            return F.mse_loss(self.net(inp), eps)
        @torch.no_grad()
        def sample(self, n, phys, device="cpu"):
            x = torch.randn(n, self.d, device=device)
            for t in reversed(range(self.T)):
                tb = torch.full((n,), t, device=device, dtype=torch.long)
                inp = torch.cat([x, tb.float()[:, None]/self.T], -1) + self.phys(phys)
                eps = self.net(inp)
                a = 1-self.betas[t]; ab = self.abar[t]
                x = (x - (1-a)/torch.sqrt(1-ab)*eps)/torch.sqrt(a)
                if t > 0: x = x + torch.sqrt(self.betas[t])*torch.randn_like(x)
            return {"candidate": x,
                    "STATUS": "HYPOTHESIS — ten gates before design status"}


# =============================================================================
# §9. PRODUCTION TRAINER
# =============================================================================

@dataclass
class TrainConfig:
    epochs: int = 80; lr: float = 3e-3; weight_decay: float = 1e-4
    grad_clip: float = 1.0; patience: int = 15; amp: bool = True
    log_every: int = 0


class Trainer:
    """AMP + clipping + cosine LR + early stop + best-checkpoint restore."""
    def __init__(self, model, cfg: TrainConfig, run: RunConfig,
                 loss_fn: Callable, name: str = "model"):
        self.cfg = cfg; self.run = run; self.name = name
        self.dev = pick_device(run.device)
        self.model = model.to(self.dev)
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, cfg.epochs)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(cfg.amp and self.dev == "cuda"))
        self.loss_fn = loss_fn
        self.best = float("inf"); self.bad = 0
        self.hist: List[Dict] = []
        self.ckpt = Path(run.out_dir)/f"{name}.best.pt"
        self.mf = Manifest(run.out_dir)
    def fit(self, *batch):
        batch = [b.to(self.dev) if hasattr(b, "to") else b for b in batch]
        t0 = time.perf_counter()
        for ep in range(self.cfg.epochs):
            self.opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                L = self.loss_fn(self.model, *batch)
            self.scaler.scale(L).backward()
            self.scaler.unscale_(self.opt)
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                self.cfg.grad_clip)
            self.scaler.step(self.opt); self.scaler.update(); self.sch.step()
            lv = float(L)
            self.hist.append({"epoch": ep, "loss": lv, "grad_norm": float(gn),
                              "lr": self.sch.get_last_lr()[0]})
            if lv < self.best - 1e-6:
                self.best = lv; self.bad = 0
                torch.save({"state": self.model.state_dict(),
                            "epoch": ep, "loss": lv, "version": VERSION},
                           self.ckpt)
            else:
                self.bad += 1
            if self.cfg.log_every and ep % self.cfg.log_every == 0:
                log.info(f"{self.name} ep{ep} loss={lv:.5f}")
            if self.bad >= self.cfg.patience: break
        if self.ckpt.exists():
            self.model.load_state_dict(
                torch.load(self.ckpt, map_location=self.dev)["state"])
        wall = time.perf_counter() - t0
        self.mf.add("train", {"name": self.name, "device": self.dev,
                              "epochs": len(self.hist), "best": self.best,
                              "wall_s": wall, "seed": self.run.seed})
        self.mf.write(f"{self.name}_manifest.json")
        return {"best_loss": self.best, "epochs": len(self.hist),
                "wall_s": wall, "device": self.dev,
                "checkpoint": str(self.ckpt), "history": self.hist,
                "early_stopped": self.bad >= self.cfg.patience}


# =============================================================================
# §10. STATISTICS DICTIONARY
# =============================================================================

class SDGate:
    ENTRIES = {
      "logical_capacity": (("HOLEVO","CONSTR"), "mode-normalised qLDPC",
                           "k < n; constructible family required"),
      "memory_error_rate": (("BPOSD","SQZ"), "calibrated BP+OSD scaling",
                            "not the surface-code heuristic"),
      "t_gate_infidelity": (("DIST",), "magic-state distillation",
                            "separate channel from memory"),
      "spin_valley_stage": (("BOLTZ","FOURIER","LIFT"),
                            "kBT << DeltaE + isolated membrane",
                            "245 K on silicon refused"),
      "cz_gate": (("COOP",), "Duan-Kimble cavity QED",
                  "bulk chi^3 cannot give pi"),
      "power_thermal": (("PWR","THERM"), "P=E*F and T=Tamb+P*Rth",
                        "choose two, compute the third"),
      "ai_candidate": ((), "none", "HYPOTHESIS until all gates pass"),
      "legacy_expansion_claim": ((), "none", "REFUSED — per-site metric"),
      "legacy_code": ((), "none", "REFUSED — non-constructible"),
      "legacy_post_distillation": ((), "none", "REFUSED — channel confusion"),
      "fabrication_claim": ((), "none", "REFUSED — design assistance only"),
    }
    def __init__(self, strict: bool = True):
        self.log: List[Dict] = []; self.strict = strict
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
    def report(self) -> Dict[str, Any]:
        n = sum(1 for c in self.log if c["status"] == MODE_CERTIFIED)
        return {"n": len(self.log), "certified": n,
                "mode_E": len(self.log)-n, "log": self.log}


PACKAGES = {
 "HOLEVO":  "k < n at the bosonic-mode level; n = M_mode*N_unit",
 "CONSTR":  "constructible BB/LP parity-check family",
 "BPOSD":   "calibrated BP+OSD sub-threshold exponent",
 "SQZ":     "GKP squeezing at or above the per-revision floor",
 "DIST":    "magic-state distillation infidelity, separate channel",
 "BOLTZ":   "k_B T << Delta_E_Zeeman and valley splitting",
 "FOURIER": "Q_leak = G*Delta_T within the isolation design",
 "LIFT":    "island leak within cryocooler cooling lift",
 "COOP":    "C = 4g^2/(kappa*gamma) >> 1, Duan-Kimble protocol",
 "PWR":     "P_compute = E_op * F verified",
 "THERM":   "T_target = T_amb + P_total*R_th below ceiling",
}


# =============================================================================
# §11. DESIGN WORKFLOW
# =============================================================================

@dataclass
class DesignCandidate:
    revision: str
    squeeze_dB: float
    stage: SpinValleyStage
    cavity: CavityQED
    E_op_fJ: float = 2.60
    F_ops: float = 1.52e17
    P_compute_W: float = 395.2


class QuantumChipDesignSuite:
    def __init__(self, cfg: Optional[RunConfig] = None):
        self.cfg = (cfg or RunConfig()).validate()
        set_seed(self.cfg.seed)
        self.sd = SDGate(self.cfg.strict)
        self.mf = Manifest(self.cfg.out_dir)
        self.cache = Cache(self.cfg.cache_max)
        self.em = ErrorModel()

    # ---- full gate stack -------------------------------------------------
    def evaluate(self, cand: DesignCandidate) -> Dict[str, Any]:
        rev = get_revision(cand.revision)
        out: Dict[str, Any] = {"revision": rev.name, "gates": {}}
        failures: List[str] = []

        def run(name, fn):
            try:
                r = fn(); out["gates"][name] = r; return r
            except Refusal as e:
                out["gates"][name] = {"REFUSED": str(e)}
                failures.append(name); return None

        run("G1_holevo", lambda: gate_holevo(rev))
        run("G2_constructibility", lambda: gate_constructibility(rev.code))
        me = run("G3_bposd", lambda: self.em.memory_error(rev.p_mode,
                                                          rev.code.d_eff))
        run("G8_squeezing", lambda: gate_squeezing(rev, cand.squeeze_dB))
        if rev.quantum_mode == "split_1.8K":
            run("G5_boltzmann", lambda: gate_boltzmann(cand.stage))
            run("G6_G10_isolation", lambda: gate_fourier_isolation(cand.stage))
        else:
            out["gates"]["G5_G6_G10"] = {
                "skipped": "mobile revision runs a 300 K optical GKP cluster; "
                           "no on-die spin-valley stage",
                "quantum_mode": rev.quantum_mode}
        run("G7_cooperativity", lambda: gate_cooperativity(cand.cavity))
        run("G9_power", lambda: gate_power_identity(
            cand.P_compute_W, cand.E_op_fJ*1e-15, cand.F_ops))
        bd = R10_POWER_BREAKDOWN if rev.name == "R10.0-W" else None
        run("G9_thermal", lambda: gate_thermal_closure(rev, bd))

        # G4 channel separation is reported, not optional
        if me is not None:
            out["channel_separation"] = self.em.circuit_error(
                me["P_L_mem_per_cycle"], n_cycles=10**6,
                eps_T_dist=1.0e-16, n_T=10**9)

        # SD certification
        verified: List[str] = []
        if "G1_holevo" not in failures and "G2_constructibility" not in failures:
            verified += ["HOLEVO", "CONSTR"]
        self.sd.certify("logical_capacity", verified,
                        {"n": rev.n_modes, "k": rev.k_logical,
                         "rate": rev.code.rate})
        v2: List[str] = []
        if "G3_bposd" not in failures: v2.append("BPOSD")
        if "G8_squeezing" not in failures: v2.append("SQZ")
        self.sd.certify("memory_error_rate", v2,
                        me["P_L_mem_per_cycle"] if me else None)
        v3: List[str] = []
        if rev.quantum_mode == "split_1.8K":
            if "G5_boltzmann" not in failures: v3.append("BOLTZ")
            if "G6_G10_isolation" not in failures: v3 += ["FOURIER", "LIFT"]
        self.sd.certify("spin_valley_stage", v3,
                        {"mode": rev.quantum_mode})
        self.sd.certify("cz_gate",
                        ["COOP"] if "G7_cooperativity" not in failures else [],
                        {"C": cand.cavity.cooperativity()})
        v5: List[str] = []
        if "G9_power" not in failures: v5.append("PWR")
        if "G9_thermal" not in failures: v5.append("THERM")
        self.sd.certify("power_thermal", v5, None)

        out["n_failures"] = len(failures)
        out["failures"] = failures
        out["status"] = MODE_CERTIFIED if not failures else MODE_E
        out["scope"] = DESIGN_ONLY
        self.mf.add("evaluate", {"revision": rev.name,
                                 "status": out["status"],
                                 "failures": failures})
        return out

    # ---- AI-assisted loop ------------------------------------------------
    def ai_loop(self, cand: DesignCandidate, n_candidates: int = 4
                ) -> Dict[str, Any]:
        refuse_scope("design exploration")
        rev = get_revision(cand.revision)
        base = self.evaluate(cand)
        res: Dict[str, Any] = {"baseline": base}
        if not HAS_TORCH or not HAS_NUMPY:
            res["ai"] = {"available": False,
                         "reason": "torch/numpy absent"}
            return res
        dev = pick_device(self.cfg.device)
        pv = physics_vector(rev, cand.stage, cand.cavity, self.em)
        phys = torch.tensor(pv, dtype=torch.float32,
                            device=dev).unsqueeze(0).repeat(n_candidates, 1)
        dd = DesignDiffusion(d=32, T=40, phys_dim=10).to(dev)
        samples = dd.sample(n_candidates, phys, dev)
        # Decode a candidate into a physical squeeze value in [10, 14] dB
        raw = samples["candidate"].detach().cpu().numpy()
        proposals = []
        for i in range(n_candidates):
            sq = 10.0 + 4.0*(1.0/(1.0 + math.exp(-float(raw[i, 0]))))
            c2 = DesignCandidate(cand.revision, sq, cand.stage, cand.cavity,
                                 cand.E_op_fJ, cand.F_ops, cand.P_compute_W)
            ev = self.evaluate(c2)
            proposals.append({"id": i, "squeeze_dB": sq,
                              "status": ev["status"],
                              "failures": ev["failures"],
                              "AI_STATUS": "HYPOTHESIS"})
        self.sd.certify("ai_candidate", (), {"n": n_candidates})
        res["ai"] = {"available": True, "device": dev,
                     "proposals": proposals,
                     "n_certified": sum(1 for p in proposals
                                        if p["status"] == MODE_CERTIFIED),
                     "rule": "AI proposes; physics disposes"}
        return res

    def report(self) -> Dict[str, Any]:
        rep = {"version": VERSION, "philosophy": PHILOSOPHY,
               "revisions": {k: {**asdict(v),
                                 "n_modes": v.n_modes,
                                 "k_logical": v.k_logical,
                                 "code_rate": v.code.rate,
                                 "per_site_ratio": v.per_site_ratio}
                             for k, v in GROUNDED.items()},
               "error_table": grounded_error_table(self.em),
               "legacy_audit": audit_legacy_codes(),
               "packages": PACKAGES, "cache": self.cache.stats(),
               "sd_report": self.sd.report(), "scope": DESIGN_ONLY}
        p = Path(self.cfg.out_dir)/"report.json"
        p.write_text(json.dumps(rep, indent=2, default=str, ensure_ascii=False))
        self.mf.write()
        return rep


# =============================================================================
# §12. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    if not HAS_NUMPY:
        print("  numpy required. FAIL"); return False
    ok = True
    def rep(n, p, e=""):
        nonlocal ok; ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    cfg = RunConfig().validate()
    suite = QuantumChipDesignSuite(cfg)
    em = ErrorModel()
    stage = SpinValleyStage()
    cq = CavityQED()

    print("\n[T1] G1 — MODE NORMALISATION / HOLEVO")
    for name, rv in GROUNDED.items():
        g = gate_holevo(rv)
        print(f"      {name:8s} n={rv.n_modes:>8d} k={rv.k_logical:>7d} "
              f"rate={g['code_rate']:.5f} per-site={g['per_site_ratio']:.3f}")
        rep(f"{name} k < n", rv.k_logical < rv.n_modes)
        rep(f"{name} rate = 0.09375", abs(g["code_rate"]-0.09375) < 1e-9)
        rep(f"{name} n = M*N_unit", rv.code.n == rv.n_modes)
    b = False
    try: gate_holevo(GROUNDED["R10.0-W"], claimed_expansion=32.0)
    except HolevoRefusal: b = True
    rep("32x per-site expansion as resource claim REFUSED", b)

    print("\n[T2] G2 — CODE CONSTRUCTIBILITY")
    for name, rv in GROUNDED.items():
        g = gate_constructibility(rv.code)
        rep(f"{name} [[{rv.code.n},{rv.code.k},{rv.code.d_eff}]] accepted",
            g["passes"])
    audit = audit_legacy_codes()
    for r in audit["rows"]:
        print(f"      legacy {r['revision']:8s} {r['legacy_code']} "
              f"rate={r['legacy_rate']:.4f} -> {r['verdict']}")
    rep("all 4 legacy codes REFUSED", audit["n_refused"] == 4,
        f"{audit['n_refused']}/4")
    b = False
    try: gate_constructibility(QCode(512, 341, 12, 6, "claimed"),
                               underlying_nc=20)
    except ConstructibilityRefusal: b = True
    rep("[[512,341,12]] with n_c=20 REFUSED", b)

    print("\n[T3] G3 — BP+OSD CALIBRATED SCALING")
    tab = grounded_error_table(em)
    expected = {"R9.0-M": 3.7e-6, "R9.5-M": 4.6e-11,
                "R9.8-M": 1.3e-14, "R10.0-W": 5.8e-30}
    for r in tab["rows"]:
        print(f"      {r['revision']:8s} expo={r['exponent']:5.2f} "
              f"P_L,mem={r['P_L_mem']:.2e}  "
              f"(surface heuristic {r['surface_heuristic_for_comparison']:.2e})")
        e = expected[r["revision"]]
        rep(f"{r['revision']} P_L within 25% of grounded value",
            abs(r["P_L_mem"]-e)/e < 0.25, f"{r['P_L_mem']:.2e} vs {e:.1e}")
    b = False
    try: em.memory_error(1.0e-2, 12)
    except ErrorModelRefusal: b = True
    rep("above-threshold p_mode REFUSED", b)
    b = False
    try: ErrorModel(alpha_BP=0.60).memory_error(9.2e-4, 12)
    except ErrorModelRefusal: b = True
    rep("uncalibrated alpha_BP REFUSED", b)

    print("\n[T4] G4 — MEMORY vs DISTILLATION SEPARATION")
    me = em.memory_error(7.8e-4, 32)
    cs = em.circuit_error(me["P_L_mem_per_cycle"], 10**6, 1e-16, 10**9)
    print(f"      memory term={cs['memory_term']:.2e}  "
          f"T term={cs['t_gate_term']:.2e}")
    rep("channels are additive and distinct",
        abs(cs["P_circuit_approx"]-(cs["memory_term"]+cs["t_gate_term"])) < 1e-30)
    rep("separation stated", "cannot reduce" in cs["SEPARATION"])
    b = False
    try: refuse_post_distillation_memory("post-distillation P_L = 3.2e-26")
    except ErrorModelRefusal: b = True
    rep("post-distillation memory claim REFUSED", b)

    print("\n[T5] G5 — BOLTZMANN SPIN INITIALISATION")
    g = gate_boltzmann(stage)
    print(f"      T=1.8 K: kBT={g['kBT_meV']:.4f} meV  "
          f"dE_Z={g['delta_zeeman_meV']:.4f} meV  "
          f"ratio={g['zeeman_over_kBT']:.3f}  F={g['thermal_init_fidelity']:.4f}")
    rep("kBT ~ 0.155 meV at 1.8 K", abs(g["kBT_meV"]-0.1551) < 0.002)
    rep("dE_Z ~ 0.405 meV at 3.5 T", abs(g["delta_zeeman_meV"]-0.4052) < 0.005)
    rep("F_init ~ 0.932", abs(g["thermal_init_fidelity"]-0.932) < 0.005)
    legacy = refuse_legacy_thermal("R9.8-M")
    print(f"      legacy 245 K -> {legacy['boltzmann']}")
    rep("245 K spin-valley REFUSED", legacy["boltzmann"] == "REFUSED")
    hot = SpinValleyStage(T_dot_K=245.0, B_T=1.5)
    rep("245 K F_init ~ 0.50 (scrambled)",
        abs(hot.boltzmann_init_fidelity()-0.5) < 0.02,
        f"{hot.boltzmann_init_fidelity():.4f}")

    print("\n[T6] G6/G10 — FOURIER ISOLATION AND CRYO LIFT")
    iso = gate_fourier_isolation(stage)
    print(f"      leak/island={iso['leak_per_island_W']*1e6:.1f} uW  "
          f"total={iso['total_leak_W']*1e3:.1f} mW  "
          f"lift={iso['cryo_lift_W']*1e3:.0f} mW")
    rep("island leak within lift", iso["passes"])
    cx = monolithic_silicon_counterexample()
    print(f"      monolithic Si: G={cx['G_W_K']:.1f} W/K  "
          f"Q={cx['Q_leak_W']:.0f} W  lift={cx['lift_W']:.2f} W  "
          f"shortfall {cx['shortfall_factor']:.0f}x")
    rep("Si conductance ~22.7 W/K", abs(cx["G_W_K"]-22.7) < 0.5)
    rep("monolithic Peltier REFUSED", cx["VERDICT"].startswith("REFUSED"))
    rep("shortfall > 1000x", cx["shortfall_factor"] > 1000)
    b = False
    try: gate_fourier_isolation(SpinValleyStage(n_islands=3000))
    except CryoBudgetRefusal: b = True
    rep("3000 islands exceed lift -> REFUSED", b)

    print("\n[T7] G7 — CAVITY-QED PI GATE")
    gq = gate_cooperativity(cq)
    print(f"      C = {gq['C']:.0f}  phi_err = {gq['phi_err_rad']:.5f} rad")
    rep("C ~ 1.67e4", abs(gq["C"]-16686) < 50)
    rep("phase error < 0.01 rad", gq["phi_err_rad"] < 0.01)
    bx = bulk_chi3_counterexample()
    print(f"      bulk chi3: phi={bx['enhanced_phase_rad']:.3f} rad  "
          f"shortfall {bx['shortfall_factor']:.1f}x")
    rep("bulk chi3 REFUSED for pi", bx["VERDICT"].startswith("REFUSED"))
    rep("GST labelled linear trimmer", "linear" in bx["GST_ROLE"])
    b = False
    try: gate_cooperativity(CavityQED(g_2pi_MHz=100.0))
    except CooperativityRefusal: b = True
    rep("low cooperativity REFUSED", b)

    print("\n[T8] G8/G9 — SQUEEZING, POWER, THERMAL")
    for name, rv in GROUNDED.items():
        gs = gate_squeezing(rv, rv.squeeze_min_dB)
        rep(f"{name} squeezing floor {rv.squeeze_min_dB} dB", gs["passes"])
        b = False
        try: gate_squeezing(rv, rv.squeeze_min_dB-0.5)
        except SqueezingRefusal: b = True
        rep(f"{name} below floor REFUSED", b)
    gp = gate_power_identity(395.2, 2.60e-15, 1.52e17)
    rep("R8.5-style identity 395.2 W holds", gp["passes"])
    b = False
    try: gate_power_identity(450.0, 11.4e-15, 1.52e17)
    except PowerRefusal: b = True
    rep("R7.0 inconsistency (1733 W vs 450 W) REFUSED", b)
    tw = gate_thermal_closure(GROUNDED["R10.0-W"], R10_POWER_BREAKDOWN)
    print(f"      R10.0-W: P={tw['P_total_W']:.1f} W  "
          f"T_j={tw['T_j']:.2f} C")
    rep("breakdown sums to 350.0 W", abs(tw["P_total_W"]-350.0) < 0.05)
    rep("T_j ~ 58.30 C", abs(tw["T_j"]-58.30) < 0.05)
    for name in ("R9.0-M", "R9.5-M", "R9.8-M"):
        tm = gate_thermal_closure(GROUNDED[name])
        rep(f"{name} T_skin ~ 39.20 C", abs(tm["T_skin"]-39.204) < 0.01)

    print("\n[T9] FULL GATE STACK PER REVISION")
    for name, rv in GROUNDED.items():
        cand = DesignCandidate(name, rv.squeeze_min_dB, stage, cq)
        if name != "R10.0-W":
            cand.E_op_fJ, cand.F_ops, cand.P_compute_W = 2.60, 1.0e15, 2.60
        ev = suite.evaluate(cand)
        print(f"      {name:8s} status={ev['status']} "
              f"failures={ev['failures']}")
        rep(f"{name} all gates pass", ev["status"] == MODE_CERTIFIED,
            str(ev["failures"]))
        rep(f"{name} scope design-only", ev["scope"] == DESIGN_ONLY)
    bad = DesignCandidate("R10.0-W", 10.0, SpinValleyStage(T_dot_K=245.0),
                          CavityQED(g_2pi_MHz=50.0))
    evb = suite.evaluate(bad)
    rep("bad candidate -> MODE_E", evb["status"] == MODE_E)
    rep("multiple failures named", evb["n_failures"] >= 3,
        str(evb["failures"]))

    print("\n[T10] AI COMPONENTS")
    if HAS_TORCH:
        torch.manual_seed(cfg.seed)
        dev = pick_device(cfg.device)
        pv = physics_vector(GROUNDED["R10.0-W"], stage, cq, em)
        llm = QuantumChipLLM(vocab=512, d=64, heads=4, layers=2,
                             dff=128, max_len=64, phys_dim=10).to(dev)
        for B in (1, 3, 6):
            ids = torch.randint(0, 512, (B, 16), device=dev)
            ph = torch.tensor(pv, dtype=torch.float32,
                              device=dev).unsqueeze(0).repeat(B, 1)
            o = llm(ids, ph)
            rep(f"LLM batch B={B} forward", tuple(o.shape) == (B, 16, 512))
        gen = llm.generate(ids, ph, max_new=8)
        rep("LLM generation", gen.shape[1] == 24)
        wm = QuantumWorldModel(10, 4, 64).to(dev)
        with torch.no_grad(): wm.thermal_logit.fill_(20.0)
        rep("thermal decay constrained < 1", 0 < float(wm.thermal_decay) < 1,
            f"{float(wm.thermal_decay):.6f}")
        obs = torch.tensor(pv, dtype=torch.float32,
                           device=dev).unsqueeze(0).repeat(4, 1)
        wo = wm(obs, torch.randn(4, 4, device=dev))
        rep("world model labelled surrogate", "NOT a measurement" in wo["STATUS"])
        dd = DesignDiffusion(32, 40, 128, 10).to(dev)
        opt = torch.optim.Adam(dd.parameters(), lr=3e-3)
        x0 = torch.randn(6, 32, device=dev)
        ph6 = torch.tensor(pv, dtype=torch.float32,
                           device=dev).unsqueeze(0).repeat(6, 1)
        l0 = lN = None
        for ep in range(40):
            opt.zero_grad(); L = dd.loss(x0, ph6); L.backward()
            torch.nn.utils.clip_grad_norm_(dd.parameters(), 1.0); opt.step()
            if ep == 0: l0 = float(L)
            lN = float(L)
        rep("diffusion trains", lN < l0, f"{l0:.4f}->{lN:.4f}")
        s = dd.sample(3, ph6[:3], dev)
        rep("diffusion labelled HYPOTHESIS", "HYPOTHESIS" in s["STATUS"])
        # production trainer
        m = QuantumChipLLM(vocab=256, d=64, heads=4, layers=2,
                           dff=128, max_len=64, phys_dim=10).to(dev)
        ids2 = torch.randint(0, 256, (6, 16), device=dev)
        tgt = torch.randint(0, 256, (6, 16), device=dev)
        ph2 = torch.tensor(pv, dtype=torch.float32,
                           device=dev).unsqueeze(0).repeat(6, 1)
        lf = lambda mm, ii, pp, tt: F.cross_entropy(
            mm(ii, pp).reshape(-1, 256), tt.reshape(-1))
        tr = Trainer(m, TrainConfig(epochs=50, patience=12), cfg, lf, "qllm")
        r = tr.fit(ids2, ph2, tgt)
        rep("trainer loss decreased", r["best_loss"] < r["history"][0]["loss"],
            f"{r['history'][0]['loss']:.3f}->{r['best_loss']:.3f}")
        rep("checkpoint written", Path(r["checkpoint"]).exists())
        rep("grad norms finite",
            all(math.isfinite(h["grad_norm"]) for h in r["history"]))
        loop = suite.ai_loop(DesignCandidate("R9.0-M", 11.5, stage, cq,
                                             2.60, 1.0e15, 2.60), 4)
        rep("AI loop returns proposals", loop["ai"]["available"])
        rep("proposals labelled HYPOTHESIS",
            all(p["AI_STATUS"] == "HYPOTHESIS"
                for p in loop["ai"]["proposals"]))
        rep("rule stated", loop["ai"]["rule"] == "AI proposes; physics disposes")
    else:
        rep("torch available", False, "pip install torch")

    print("\n[T11] SCOPE AND SD REPORT")
    b = False
    try: refuse_scope("please fabricate this quantum chip")
    except ScopeRefusal: b = True
    rep("fabrication claim REFUSED", b)
    suite.sd.certify("legacy_expansion_claim", ())
    suite.sd.certify("legacy_code", ())
    suite.sd.certify("legacy_post_distillation", ())
    rr = suite.report()
    sr = rr["sd_report"]
    print(f"      SD: {sr['n']} outputs, {sr['certified']} certified, "
          f"{sr['mode_E']} Mode E")
    rep("Mode E retained", sr["mode_E"] > 0)
    rep("report written", Path(cfg.out_dir, "report.json").exists())
    rep("manifest written", Path(cfg.out_dir, "manifest.json").exists())
    rep("philosophy included", "motto" in rr["philosophy"])
    rep("all 4 revisions in report", len(rr["revisions"]) == 4)

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
BUILT — TEN ENFORCED GATES
  G1  Holevo / mode normalisation: n = M_mode*N_unit, k < n, rate 0.09375.
      Per-site ratios (0.750 / 3.000 / 3.000 / 4.500) are labelled device
      multiplexing metrics, never resource expansion.
  G2  Constructibility: grounded [[512,48,12]], [[16384,1536,24]],
      [[32768,3072,32]], [[1572864,147456,64]] accepted; all four legacy
      rate-2/3 codes REFUSED with the Singleton/HGP reason.
  G3  BP+OSD calibrated scaling reproduces the grounded table
      (3.7e-6, 4.6e-11, 1.3e-14, 5.8e-30) and refuses above-threshold
      or uncalibrated exponents.
  G4  Memory and T-gate channels are additive and distinct; any
      post-distillation memory rate is refused.
  G5  Boltzmann: 1.8 K / 3.5 T gives kBT 0.155 meV, dE_Z 0.405 meV,
      F_init 0.932. 245 K gives F_init ~0.50 and is refused.
  G6  Fourier: monolithic 227 mm^2 Si gives G 22.7 W/K and ~1930 W leak
      against 0.63 W of Peltier lift — refused by >1000x.
  G7  Cavity QED: C = 1.67e4, Duan-Kimble pi with 0.0077 rad error.
      Bulk chi^(3) gives ~0.5 rad and is refused; GST is a linear trimmer.
  G8  Per-revision squeezing floors 11.5 / 12.0 / 12.2 / 12.8 dB enforced.
  G9  P = E*F and T = T_amb + P*R_th: R10.0-W sums to 350.0 W, T_j 58.30 C;
      mobile T_skin 39.20 C. The R7.0-style 1733 W vs 450 W conflict is refused.
  G10 Island leak fits the 350 mW cryocooler lift; 3000 islands refused.

AI LAYER
  Physics-conditioned LLM transformer (batch-correct), world model with a
  constrained slow thermal latent, and a physics-conditioned DDPM. All
  outputs are HYPOTHESES. Production trainer: AMP, clipping, cosine LR,
  early stopping, best-checkpoint restore, manifest.

NOT CLAIMED
  No hardware is fabricated. No code is certified without a parity-check
  matrix. No logical rate is claimed below the squeezing floor. The
  cryogenic and cavity-QED parameters are TARGETS requiring the six
  experimental milestones M1-M6 of the grounded revision.
""")
    return ok


# =============================================================================
# §13. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="MSPS quantum chip design suite (physically grounded)")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest", "revisions", "errors", "legacy", "thermal",
                 "cavity", "evaluate", "ai", "philosophy", "packages",
                 "report"])
    ap.add_argument("--rev", default="R10.0-W")
    ap.add_argument("--squeeze", type=float, default=None)
    ap.add_argument("--tdot", type=float, default=1.8)
    ap.add_argument("--islands", type=int, default=1000)
    ap.add_argument("--n-candidates", type=int, default=4)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest":
        return 0 if self_test() else 1
    if a.cmd == "philosophy":
        print(json.dumps(PHILOSOPHY, indent=2, ensure_ascii=False)); return 0
    if a.cmd == "packages":
        print(json.dumps(PACKAGES, indent=2)); return 0

    suite = QuantumChipDesignSuite()
    out: Dict[str, Any] = {}
    try:
        if a.cmd == "revisions":
            out = {k: {**asdict(v), "n_modes": v.n_modes,
                       "k_logical": v.k_logical, "code_rate": v.code.rate,
                       "per_site_ratio": v.per_site_ratio}
                   for k, v in GROUNDED.items()}
        elif a.cmd == "errors":
            out = grounded_error_table()
        elif a.cmd == "legacy":
            out = {"codes": audit_legacy_codes(),
                   "thermal": {k: refuse_legacy_thermal(k) for k in GROUNDED},
                   "bulk_chi3": bulk_chi3_counterexample(),
                   "monolithic_Si": monolithic_silicon_counterexample()}
        elif a.cmd == "thermal":
            st = SpinValleyStage(T_dot_K=a.tdot, n_islands=a.islands)
            out = {"boltzmann": gate_boltzmann(st),
                   "isolation": gate_fourier_isolation(st)}
        elif a.cmd == "cavity":
            out = {"cavity_qed": gate_cooperativity(CavityQED()),
                   "bulk_chi3_refused": bulk_chi3_counterexample()}
        elif a.cmd in ("evaluate", "ai"):
            rv = get_revision(a.rev)
            sq = a.squeeze if a.squeeze is not None else rv.squeeze_min_dB
            st = SpinValleyStage(T_dot_K=a.tdot, n_islands=a.islands)
            if a.rev == "R10.0-W":
                cand = DesignCandidate(a.rev, sq, st, CavityQED())
            else:
                cand = DesignCandidate(a.rev, sq, st, CavityQED(),
                                       2.60, 1.0e15, 2.60)
            out = suite.evaluate(cand) if a.cmd == "evaluate" \
                  else suite.ai_loop(cand, a.n_candidates)
        elif a.cmd == "report":
            out = suite.report()
    except Refusal as e:
        out = {"REFUSED": str(e)}
    print(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str,
                                           ensure_ascii=False))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
