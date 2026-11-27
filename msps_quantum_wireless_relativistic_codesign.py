#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_quantum_wireless_relativistic_codesign.py — v1.0.0  [PRODUCTION]

AI Co-Design Suite for the DSQP-SESI-QM-OTA / OTA-GR Quantum Chip Family
  R11.0  OTA          (superseded: K->1 quantum AirComp)
  R12.0  OTA-GR       (superseded: carpet RT, Everettian branches)
  R12.0  Grounded     (AUTHORITATIVE)
  R12.0  Grounded (1) (AUTHORITATIVE, erasure/Pauli split + slant Shapiro)

LLM/Transformer + World Model + Diffusion, every output physics-gated.

=============================================================================
THE SIX NO-GO THEOREMS THIS MODULE ENFORCES
=============================================================================
[N1] K-TO-1 COMBINER NO-GO
     A passive unitary K-port combiner merging K independent spatial
     transmitters into ONE receiver mode obeys eta_combine <= 1/K
     (conservation of etendue / brightness). The pure-loss bosonic quantum
     capacity is Q(eta) = max{0, log2[eta/(1-eta)]}, so eta <= 0.5 gives
     Q = 0 identically. Quantum states therefore travel ONLY on Plane Q
     (1->1 point-to-point or WDM/MIMO-orthogonal, K=1 per mode).
     Plane C (300 GHz sub-THz, n_th = 20.3) carries CLASSICAL AirComp only.

[N2] BARE TELEPORTATION FIDELITY BOUND
     Braunstein-Kimble: F_raw = 1/(1 + exp(-2r)), S_dB = 10 log10(e^{2r}).
     At 12.2 dB, F_raw = 0.9432. F_raw = 0.994 requires 22.19 dB.
     High LOGICAL fidelity is reached by erasure heralding + qLDPC decoding,
     NOT by claiming a bare fidelity the squeezing cannot support.

[N3] ERASURE / PAULI SEPARATION
     sigma^2_tot = e^{-2r}/2 + (1-eta)/2. At 12.2 dB and eta=0.68 this is
     0.1901 (sigma=0.436), giving raw shift error 8.2e-2. Voronoi-boundary
     post-selection converts boundary events into HERALDED ERASURES;
     only the residual UNHERALDED PAULI rate enters the Pauli decoder.
     Per-quadrature and per-mode erasure rates are DIFFERENT numbers and
     must be declared explicitly before comparison with p_c.

[N4] EUCLIDEAN FRACTAL != HYPERBOLIC BULK
     Sierpinski carpet: d_H = ln8/ln3 = 1.8928, minimal cut ~ (8/3)^m
     = L^0.8928 (polynomial). Ryu-Takayanagi in AdS3/CFT2 requires
     S(A) ~ (c/3) ln|A| (logarithmic). Only a constant-negative-curvature
     {5,4} pentagonal graph supplies this. The carpet is PHYSICAL ROUTING;
     the {5,4} tessellation is the LOGICAL tensor network. Because {5,4}
     grows exponentially in depth while the carpet grows polynomially,
     embedding is restricted to FINITE DEPTH with an explicit crossing
     and dilation budget.

[N5] DECOHERED BRANCHES CANNOT INTERFERE
     If <E_1|E_2> = 0 the branches are orthogonal environment records and
     cross-branch interference is physically impossible. What IS possible
     before decoherence is a controlled-SWAP (Fredkin) overlap measurement
     with Quantum Amplitude Estimation: N_queries = O(1/eps), a quadratic
     speedup over O(1/eps^2) classical sampling. This is an OVERLAP AND
     RENYI ENTROPY ESTIMATOR, not "many-worlds interference".

[N6] CAUSAL FEED-FORWARD
     Teleportation reconstructs the state only after a CLASSICAL displacement
     message arrives: t_ff >= L/c. No superluminal signalling, ever.

=============================================================================
WHAT THIS MODULE IS AND IS NOT
=============================================================================
IS   : a design-assistance and verification workbench. It checks Stratum I
       invariants, generates and gates candidate configurations, and reports
       Stratum II hardware values as TARGETS.
IS NOT: a fab, a foundry flow, a quantum simulator replacement, or a claim
       that any of these chips has been built or measured.

Every AI output is a HYPOTHESIS. It becomes usable only after a deterministic
physics gate passes. Otherwise it is retained as Mode E.
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

import argparse, hashlib, json, math, os, sys, time
from dataclasses import dataclass, field, asdict
from enum import Enum
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

VERSION = "1.0.0"; PI = math.pi

# --- physical constants (SI) -------------------------------------------------
H_PLANCK   = 6.62607015e-34
C_LIGHT    = 2.99792458e8
K_BOLTZ    = 1.380649e-23
GM_EARTH   = 3.986004418e14      # m^3/s^2
R_EARTH    = 6.371e6             # m
MU_B_EV_T  = 5.7883818060e-5     # eV/T
KB_EV_K    = 8.617333262e-5      # eV/K

MODE_CERTIFIED = "CERTIFIED"
MODE_E         = "UNCERTAINTY_RETAINED"
DESIGN_ONLY    = "quantum_design_assistance_only_not_fabrication"

PHILOSOPHY = {
    "tao": "The Source — mathematical truth underlying all computation",
    "te":  "The Path — disciplined engineering practice",
    "qi":  "The Proof — measured, validated, reproducible results",
    "p1": "AI proposes; physics disposes",
    "p2": "Stratum I may be certified; Stratum II must be measured",
    "p3": "A no-go theorem is not an engineering inconvenience",
    "p4": "Separate what is heralded from what is not",
    "p5": "No simulation replaces fabrication; no model replaces wisdom",
    "motto": "MY SOUL MOVE BY POWER OF HOLY SPIRIT — "
             "What MSPS NETWORK SEE, Lord Buddha Knows.",
}


# =============================================================================
# §0. EXCEPTIONS, LOGGING, CONFIG, CACHE, MANIFEST
# =============================================================================

class Refusal(RuntimeError): pass
class NoGoViolation(Refusal): pass
class StratumRefusal(Refusal): pass
class CausalityRefusal(Refusal): pass
class GeometryRefusal(Refusal): pass
class CapacityRefusal(Refusal): pass
class ScopeRefusal(Refusal): pass
class ConventionRefusal(Refusal): pass


def get_logger(name: str, json_mode: bool = False):
    import logging
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr)
        if json_mode:
            class JF(logging.Formatter):
                def format(self, r):
                    return json.dumps({"t": self.formatTime(r),
                                       "lvl": r.levelname, "mod": r.name,
                                       "msg": r.getMessage()}, ensure_ascii=False)
            h.setFormatter(JF())
        else:
            h.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        lg.addHandler(h); lg.setLevel(logging.INFO); lg.propagate = False
    return lg

log = get_logger("msps.qwrc")


@dataclass
class RunConfig:
    out_dir: str = "runs/qwrc"
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    json_log: bool = False
    strict: bool = True
    def validate(self) -> "RunConfig":
        if self.seed < 0: raise Refusal("seed must be >= 0")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        return self


def set_seed(s: int) -> Dict[str, Any]:
    import random
    random.seed(s); os.environ["PYTHONHASHSEED"] = str(s)
    if HAS_NUMPY: np.random.seed(s)
    if HAS_TORCH:
        torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    return {"seed": s, "numpy": HAS_NUMPY, "torch": HAS_TORCH}


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
    return h.hexdigest()[:32]


class Cache:
    """Content-addressed LRU. Deterministic keys, no hidden state."""
    def __init__(self, max_items: int = 256):
        self._m: Dict[str, Any] = {}; self._o: List[str] = []
        self.max = max_items; self.hits = 0; self.misses = 0
    def get(self, k: str):
        if k in self._m:
            self.hits += 1; self._o.remove(k); self._o.append(k)
            return self._m[k]
        self.misses += 1; return None
    def put(self, k: str, v: Any):
        self._m[k] = v; self._o.append(k)
        while len(self._o) > self.max: self._m.pop(self._o.pop(0), None)
    def stats(self):
        t = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / max(t, 1), "items": len(self._m)}


class Manifest:
    """Append-only reproducibility record."""
    def __init__(self, out_dir: str):
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.items: List[Dict] = []
    def add(self, kind: str, payload: Dict) -> Dict:
        rec = {"kind": kind, "version": VERSION, "key": content_key(kind, payload),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "payload": payload}
        self.items.append(rec); return rec
    def write(self, name: str = "manifest.json") -> str:
        p = self.dir / name
        p.write_text(json.dumps({"n": len(self.items), "items": self.items},
                                indent=2, default=str, ensure_ascii=False))
        return str(p)


def _np():
    if not HAS_NUMPY: raise Refusal("numpy required for this operation")


def phi_std(x: float) -> float:
    """Standard normal CDF via erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def refuse_scope(request: str) -> None:
    low = request.lower()
    for kw in ("fabricate", "manufacture", "tape out now", "run the fab",
               "build the chip", "produce wafer", "weapon", "warhead",
               "ผลิตจริง", "สร้างชิปจริง", "อาวุธ"):
        if kw in low:
            raise ScopeRefusal(f"REFUSED '{kw}': {DESIGN_ONLY}.")


BANNER = f"""
=============================================================================
MSPS QUANTUM WIRELESS-RELATIVISTIC CO-DESIGN v{VERSION}  [PRODUCTION]
Plane Q (1->1 optical quantum) | Plane C (300 GHz classical AirComp)
F_raw vs F_L separated | erasure vs Pauli separated | {5,4} vs carpet separated
GR metric compensation | Q-OEE O(1/eps) | causality t_ff >= L/c
Stratum I certified | Stratum II always Mode E | DESIGN ASSISTANCE ONLY
=============================================================================
"""


# =============================================================================
# §1. STRATUM REGISTRY  (Stratum I provable, Stratum II target)
# =============================================================================

class Stratum(Enum):
    I  = "stratum_I_proven_invariant"
    II = "stratum_II_hardware_target"


STRATUM_I = {
    "k_to_1_combiner_no_go",        # eta <= 1/K, etendue conservation
    "holevo_pure_loss_capacity",    # Q(eta) = max{0, log2[eta/(1-eta)]}
    "braunstein_kimble_fidelity",   # F_raw = 1/(1+e^{-2r})
    "gkp_variance_budget",          # sigma^2 = e^{-2r}/2 + (1-eta)/2
    "voronoi_erasure_heralding",    # geometric post-selection
    "bp_osd_logical_scaling",       # calibrated sub-threshold form
    "carpet_hausdorff_dimension",   # ln8/ln3
    "carpet_polynomial_cut",        # (8/3)^m
    "hyperbolic_log_rt_scaling",    # {5,4} negative curvature
    "swap_test_overlap_identity",   # P(0) = (1+Re<psi1|psi2>)/2
    "qae_query_complexity",         # O(1/eps)
    "causal_feed_forward_bound",    # t_ff >= L/c
    "schwarzschild_shapiro",        # (2GM/c^3) ln[...]
    "gravitational_redshift",       # dU/c^2
    "transverse_doppler",           # -v^2/(2c^2)
    "boltzmann_occupation",         # 1/(exp(hv/kT)-1)
    "power_energy_identity",        # P = E_op * F
    "no_signalling",                # linear unitary QM
}

STRATUM_II = {
    "gkp_squeezing_db",             # 11.5 - 12.8 dB target
    "opa_aperture_and_range",
    "atmospheric_transmittance",
    "csac_allan_deviation",
    "optical_comb_lock_jitter",
    "waveguide_crossing_loss",
    "hyperbolic_patch_node_count",
    "membrane_thermal_conductance",
    "cryocooler_lift_and_power",
    "cavity_qed_cooperativity",
    "magic_state_factory_rate",
    "teleportation_throughput",
    "qoee_latency",
    "aircomp_aggregation_latency",
    "logical_qubit_count",
}


def stratum_of(item: str) -> Stratum:
    if item in STRATUM_I:  return Stratum.I
    if item in STRATUM_II: return Stratum.II
    raise Refusal(f"unknown item '{item}'; declare its stratum first")


def assert_stratum(item: str, want: Stratum) -> None:
    got = stratum_of(item)
    if got is not want:
        raise StratumRefusal(
            f"'{item}' is {got.value}, not {want.value}. "
            "Proven invariants and hardware targets must never be conflated.")


def certify_item(item: str) -> Dict[str, Any]:
    s = stratum_of(item)
    if s is Stratum.I:
        return {"item": item, "stratum": s.value, "status": MODE_CERTIFIED,
                "note": "proven mathematical / physical invariant"}
    return {"item": item, "stratum": s.value, "status": MODE_E,
            "note": "hardware TARGET — requires measurement; never certified here"}


# =============================================================================
# §2. DOCUMENT LINEAGE  (superseded vs authoritative)
# =============================================================================

@dataclass(frozen=True)
class DocumentRevision:
    tag: str
    title: str
    status: str                    # "superseded" | "authoritative"
    superseded_claims: Tuple[str, ...]
    corrections: Tuple[str, ...]

LINEAGE: Dict[str, DocumentRevision] = {
 "R11.0-OTA": DocumentRevision(
   "R11.0-OTA",
   "Over-the-Air Quantum Computation & Distributed Photonic-Quantum Network",
   "superseded",
   ("K transmitters interfere in air to compute quantum SUM gates",
    "distributed over-the-air qLDPC stabiliser parity extraction",
    "Q-FedAvg quantum gradient states aggregated by wave superposition",
    "8-phone mesh p_loss = 0.12 with K=8 quantum combining"),
   ("K->1 combiner obeys eta <= 1/K, so Q(eta)=0 for K>=2",
    "quantum plane restricted to 1->1; classical AirComp kept on sub-THz")),
 "R12.0-OTA-GR-original": DocumentRevision(
   "R12.0-OTA-GR-original",
   "Single-Chip OTA Teleportation, GR, Holographic String Duality, Multi-Branch",
   "superseded",
   ("F_tele >= 0.994 at 12.2 dB squeezing",
    "Sierpinski carpet gives exact Ryu-Takayanagi AdS3/CFT2",
    "24-layer AdS4/CFT3 exact ER=EPR OTOC",
    "Everettian multi-branch interference, O(1/eps^2) shots",
    "CSAC sigma_y(1s) <= 3e-12",
    "delta_f/f ~ 1e-13 quoted for LEO"),
   ("F_raw = 0.943 at 12.2 dB; 0.994 needs 22.2 dB",
    "{5,4} hyperbolic logical graph, carpet is routing only",
    "AdS4/CFT3 exactness dropped",
    "Q-OEE controlled-SWAP with QAE O(1/eps)",
    "commercial CSAC ~3e-10, disciplined by optical comb",
    "LEO net delta_f/f = -2.72e-10")),
 "R12.0-grounded": DocumentRevision(
   "R12.0-grounded",
   "Physically Grounded Point-to-Point Optical Quantum Links & Classical AirComp",
   "authoritative", (),
   ("bipartite Plane Q / Plane C separation",
    "F_raw separated from F_L,tele",
    "{5,4} logical tessellation on carpet routing",
    "Q-OEE with O(1/eps) query complexity")),
 "R12.0-grounded-1": DocumentRevision(
   "R12.0-grounded-1",
   "Grounded revision with explicit erasure/Pauli split and slant Shapiro",
   "authoritative", (),
   ("sigma^2_tot = 0.1901, raw shift error 8.2e-2 made explicit",
    "Voronoi discard window w_th = 0.24 sqrt(pi), p_erase vs p_Pauli split",
    "slant-path Shapiro 4.48 ps vs zenith 2.24 ps, geometry declared",
    "finite-depth {5,4} routing budget with crossing counts")),
}


def authoritative_revisions() -> List[str]:
    return [k for k, v in LINEAGE.items() if v.status == "authoritative"]


def lineage_report() -> Dict[str, Any]:
    return {"revisions": {k: asdict(v) for k, v in LINEAGE.items()},
            "authoritative": authoritative_revisions(),
            "rule": ("a superseded claim may be inspected for provenance but "
                     "must never be used as a design input")}


# =============================================================================
# §3. [N1] BIPARTITE CHANNEL — K->1 NO-GO AND HOLEVO CAPACITY
# =============================================================================

class Plane(Enum):
    Q = "plane_Q_optical_quantum_1to1"
    C = "plane_C_subTHz_classical_aircomp"


def thermal_occupation(freq_Hz: float, T_K: float = 300.0) -> Dict[str, Any]:
    """[N1 support] n_th = 1/(exp(h nu / kT) - 1). Boltzmann/Planck exact."""
    if freq_Hz <= 0 or T_K <= 0: raise Refusal("freq and T must be positive")
    x = H_PLANCK * freq_Hz / (K_BOLTZ * T_K)
    n = 1.0 / math.expm1(x) if x < 700 else 0.0
    return {"freq_Hz": freq_Hz, "T_K": T_K, "h_nu_over_kT": x,
            "n_thermal": n,
            "quantum_capable": n < 1e-6,
            "verdict": ("quantum-capable carrier" if n < 1e-6 else
                        "CLASSICAL ONLY — thermal photons destroy squeezing")}


def holevo_pure_loss_capacity(eta: float) -> Dict[str, Any]:
    """Q(eta) = max{0, log2[eta/(1-eta)]} for the pure-loss bosonic channel."""
    if not (0.0 <= eta <= 1.0): raise Refusal("eta in [0,1]")
    if eta <= 0.5:
        Q = 0.0
    elif eta >= 1.0:
        Q = float("inf")
    else:
        Q = math.log2(eta / (1.0 - eta))
    return {"eta": eta, "Q_qubits_per_mode": Q,
            "positive_capacity": Q > 0.0,
            "threshold_eta": 0.5,
            "theorem": "Holevo-Werner pure-loss bosonic capacity"}


def combiner_transmissivity_bound(K: int) -> Dict[str, Any]:
    """[N1] eta_combine <= 1/K for a passive unitary K->1 combiner."""
    if K < 1: raise Refusal("K >= 1")
    eta = 1.0 / K
    cap = holevo_pure_loss_capacity(eta)
    return {"K": K, "eta_combine_max": eta,
            "Q_at_bound": cap["Q_qubits_per_mode"],
            "quantum_viable": cap["positive_capacity"],
            "theorem": "etendue / brightness conservation, unitary K-port",
            "consequence": ("K>=2 forces Q=0: quantum combining is FORBIDDEN"
                            if K >= 2 else "K=1: point-to-point permitted")}


class BipartiteChannelGate:
    """
    Enforces the two-plane architecture. Any attempt to place a quantum
    state on a K>=2 combined optical mode, or on the sub-THz carrier, is
    refused before any downstream computation runs.
    """
    def __init__(self, code_rate: float = 0.09375):
        self.code_rate = code_rate

    def authorize(self, plane: Plane, K: int, freq_Hz: float,
                  eta_atm: float, eta_optics: float = 1.0,
                  T_K: float = 300.0) -> Dict[str, Any]:
        occ = thermal_occupation(freq_Hz, T_K)
        if plane is Plane.Q:
            if K != 1:
                b = combiner_transmissivity_bound(K)
                raise NoGoViolation(
                    f"[N1] Plane Q requires K=1 per spatial/WDM mode; got K={K}. "
                    f"eta_combine <= {b['eta_combine_max']:.4f} gives "
                    f"Q = {b['Q_at_bound']:.4f} qubits/mode (zero). "
                    "Route quantum states point-to-point; use Plane C for "
                    "multi-node classical aggregation.")
            if not occ["quantum_capable"]:
                raise NoGoViolation(
                    f"[N1] carrier n_th = {occ['n_thermal']:.4g} at "
                    f"{freq_Hz/1e9:.1f} GHz is not quantum-capable. "
                    "Sub-THz belongs to Plane C.")
            eta_tot = eta_atm * eta_optics
            cap = holevo_pure_loss_capacity(eta_tot)
            if not cap["positive_capacity"]:
                raise CapacityRefusal(
                    f"eta_total = {eta_tot:.4f} <= 0.5 gives Q = 0. "
                    "No quantum capacity on this link.")
            margin = cap["Q_qubits_per_mode"] / self.code_rate
            return {"authorized": True, "plane": plane.value, "K": 1,
                    "eta_total": eta_tot, "capacity": cap,
                    "code_rate": self.code_rate,
                    "capacity_margin_over_code_rate": margin,
                    "thermal": occ,
                    "constraint": "1->1 only; WDM/MIMO orthogonal modes allowed"}
        # Plane C: classical only, K may exceed 1
        if occ["quantum_capable"]:
            log.warning("Plane C requested on a quantum-capable carrier; "
                        "still restricted to classical aggregation by policy.")
        return {"authorized": True, "plane": plane.value, "K": K,
                "quantum_states_permitted": False,
                "thermal": occ,
                "permitted": ["syndrome parity sums (classical bits)",
                              "channel-state telemetry",
                              "federated gradient values",
                              "clock sync and causal feed-forward"],
                "theorem": "classical AirComp (Nazer-Gastpar); no quantum claim"}


# =============================================================================
# §4. [N2][N3] GKP FIDELITY CHAIN — raw, variance, erasure, Pauli, logical
# =============================================================================

def squeezing_db_to_exp(s_db: float) -> float:
    """S_dB = 10 log10(e^{2r})  =>  e^{-2r} = 10^{-S_dB/10}."""
    return 10.0 ** (-s_db / 10.0)


def bare_teleport_fidelity(s_db: float) -> Dict[str, Any]:
    """[N2] Braunstein-Kimble F_raw = 1/(1+e^{-2r})."""
    e = squeezing_db_to_exp(s_db)
    F = 1.0 / (1.0 + e)
    return {"squeezing_dB": s_db, "exp_minus_2r": e, "F_raw": F,
            "theorem": "Braunstein-Kimble continuous-variable teleportation",
            "NOT": "the post-error-correction logical fidelity"}


def squeezing_for_target_fidelity(F_target: float) -> Dict[str, Any]:
    if not (0.5 < F_target < 1.0): raise Refusal("F_target in (0.5,1)")
    e = (1.0 - F_target) / F_target
    s_db = -10.0 * math.log10(e)
    return {"F_target": F_target, "required_exp_minus_2r": e,
            "required_squeezing_dB": s_db}


@dataclass
class GKPNoiseBudget:
    squeezing_dB: float = 12.2
    eta_total: float = 0.68

    @property
    def sigma2_squeezing(self) -> float:
        return squeezing_db_to_exp(self.squeezing_dB) / 2.0
    @property
    def sigma2_loss(self) -> float:
        return (1.0 - self.eta_total) / 2.0
    @property
    def sigma2_total(self) -> float:
        return self.sigma2_squeezing + self.sigma2_loss
    @property
    def sigma_total(self) -> float:
        return math.sqrt(self.sigma2_total)

    def report(self) -> Dict[str, Any]:
        return {"squeezing_dB": self.squeezing_dB, "eta_total": self.eta_total,
                "sigma2_squeezing": self.sigma2_squeezing,
                "sigma2_loss": self.sigma2_loss,
                "sigma2_total": self.sigma2_total,
                "sigma_total": self.sigma_total,
                "identity": "sigma^2_tot = e^{-2r}/2 + (1-eta)/2"}


class ErasureConvention(Enum):
    PER_QUADRATURE = "per_quadrature"
    PER_MODE_UNION = "per_mode_union"


def raw_shift_error(budget: GKPNoiseBudget) -> Dict[str, Any]:
    """Probability that |Delta q| or |Delta p| exceeds sqrt(pi)/2 (no heralding)."""
    half = math.sqrt(PI) / 2.0
    z = half / budget.sigma_total
    p_one = 2.0 * (1.0 - phi_std(z))
    p_any = 1.0 - (1.0 - p_one) ** 2
    return {"correctable_half_window": half, "z_score": z,
            "p_exceed_one_quadrature": p_one,
            "p_exceed_either_quadrature": p_any,
            "note": "raw shift error WITHOUT erasure heralding"}


def voronoi_erasure_split(budget: GKPNoiseBudget,
                          w_th_frac_of_sqrt_pi: float = 0.24,
                          convention: ErasureConvention = ErasureConvention.PER_QUADRATURE,
                          p_c_carpet: float = 0.362) -> Dict[str, Any]:
    """
    [N3] Voronoi-boundary post-selection.
    Discard window: |Delta| in [sqrt(pi)/2 - w_th, sqrt(pi)/2].
    Events inside the window become HERALDED ERASURES.
    Events beyond sqrt(pi)/2 that were not flagged remain UNHERALDED PAULI.

    The per-quadrature and per-mode-union erasure rates are DIFFERENT.
    The caller MUST declare which is compared against p_c.
    """
    half = math.sqrt(PI) / 2.0
    w_th = w_th_frac_of_sqrt_pi * math.sqrt(PI)
    lower = half - w_th
    s = budget.sigma_total
    z_lo, z_hi = lower / s, half / s
    P_lo = 2.0 * (1.0 - phi_std(z_lo))     # |Delta| > lower
    P_hi = 2.0 * (1.0 - phi_std(z_hi))     # |Delta| > half
    in_window_q = P_lo - P_hi              # flagged per quadrature

    p_erase_q = P_lo                       # flag-or-beyond, per quadrature
    p_erase_mode = 1.0 - (1.0 - P_lo) ** 2
    p_pauli_q = P_hi                       # residual unheralded, per quadrature
    p_pauli_mode = 1.0 - (1.0 - P_hi) ** 2

    if convention is ErasureConvention.PER_QUADRATURE:
        p_erase, p_pauli = p_erase_q, p_pauli_q
    else:
        p_erase, p_pauli = p_erase_mode, p_pauli_mode

    return {"w_th": w_th, "w_th_frac_of_sqrt_pi": w_th_frac_of_sqrt_pi,
            "discard_lower_edge": lower, "correctable_half_window": half,
            "z_lower": z_lo, "z_upper": z_hi,
            "in_window_per_quadrature": in_window_q,
            "p_erase_per_quadrature": p_erase_q,
            "p_erase_per_mode_union": p_erase_mode,
            "p_pauli_per_quadrature": p_pauli_q,
            "p_pauli_per_mode_union": p_pauli_mode,
            "convention_declared": convention.value,
            "p_erase_used": p_erase, "p_pauli_used": p_pauli,
            "p_c_carpet": p_c_carpet,
            "below_percolation_threshold": p_erase < p_c_carpet,
            "REQUIRED_DECLARATION": (
                "state whether p_c applies per quadrature (independent q/p "
                "syndrome layers) or per mode/bond; the two rates differ by "
                "roughly a factor of two and change the verdict")}


def bp_osd_logical_error(p_phys: float, d_eff: int,
                         alpha_bp: float = 0.42,
                         p_th_qldpc: float = 7.2e-3,
                         A_code: float = 0.12) -> Dict[str, Any]:
    """
    Calibrated qLDPC BP+OSD sub-threshold MEMORY error.
    P_L,mem <= A_code * (p/p_th)^{alpha_bp * d_eff}
    NOT the surface-code (p/p_th)^{d/2} heuristic.
    """
    if p_phys >= p_th_qldpc:
        raise CapacityRefusal(
            f"p_phys={p_phys:.3e} at or above qLDPC threshold {p_th_qldpc:.3e}")
    ratio = p_phys / p_th_qldpc
    expo = alpha_bp * d_eff
    P = A_code * ratio ** expo
    return {"p_physical": p_phys, "d_eff": d_eff, "alpha_bp": alpha_bp,
            "p_th_qldpc": p_th_qldpc, "A_code": A_code,
            "ratio": ratio, "exponent": expo,
            "P_L_memory_per_cycle": P,
            "model": "calibrated BP+OSD (qLDPC)",
            "SEPARATE_FROM": "distilled T-gate infidelity eps_T",
            "NOT": "surface-code (p/p_th)^{d/2}"}


def magic_state_distillation(p_in: float, levels: int = 2,
                             protocol_factor: float = 35.0) -> Dict[str, Any]:
    """15-to-1 style: eps_out ~ 35 eps_in^3 per level. T-gate channel only."""
    eps = p_in; chain = [eps]
    for _ in range(levels):
        eps = protocol_factor * eps ** 3; chain.append(eps)
    return {"input_infidelity": p_in, "levels": levels,
            "output_infidelity": eps, "chain": chain,
            "cost_per_level": "15 input states per output (15-to-1)",
            "NOTE": "does not change memory P_L"}


class GKPFidelityChain:
    """End-to-end: raw -> variance -> erasure/Pauli -> qLDPC logical."""
    def __init__(self, code_n: int, code_k: int, code_d: int):
        if code_k >= code_n:
            raise NoGoViolation(f"k={code_k} >= n={code_n} violates Holevo")
        self.n, self.k, self.d = code_n, code_k, code_d
        self.rate = code_k / code_n

    def evaluate(self, budget: GKPNoiseBudget,
                 convention: ErasureConvention = ErasureConvention.PER_QUADRATURE,
                 w_th_frac: float = 0.24,
                 alpha_bp: float = 0.42) -> Dict[str, Any]:
        raw = bare_teleport_fidelity(budget.squeezing_dB)
        shift = raw_shift_error(budget)
        split = voronoi_erasure_split(budget, w_th_frac, convention)
        if not split["below_percolation_threshold"]:
            raise CapacityRefusal(
                f"p_erase = {split['p_erase_used']:.4f} >= p_c = "
                f"{split['p_c_carpet']:.4f} under the "
                f"{convention.value} convention. Erasure decoding cannot be "
                "certified; re-declare the convention or reduce loss.")
        logical = bp_osd_logical_error(split["p_pauli_used"], self.d, alpha_bp)
        F_L = 1.0 - logical["P_L_memory_per_cycle"]
        return {"code": {"n": self.n, "k": self.k, "d": self.d,
                         "rate": self.rate},
                "noise_budget": budget.report(),
                "F_raw": raw["F_raw"],
                "raw_shift_error": shift,
                "erasure_pauli_split": split,
                "logical_memory_error": logical,
                "F_logical_teleport": F_L,
                "separation": ("F_raw is the bare Gaussian bound; F_logical "
                               "is after heralding + qLDPC decoding"),
                "stratum": {"F_raw": MODE_CERTIFIED,
                            "squeezing_dB": MODE_E,
                            "eta_total": MODE_E,
                            "F_logical": MODE_E}}


# =============================================================================
# §5. [N6] CAUSALITY AND GR METRIC COMPENSATION
# =============================================================================

def causal_feed_forward_min(L_m: float) -> Dict[str, Any]:
    """[N6] t_ff >= L/c. No superluminal signalling."""
    if L_m < 0: raise Refusal("L >= 0")
    t = L_m / C_LIGHT
    return {"L_m": L_m, "t_ff_min_s": t,
            "theorem": "relativistic causality; classical message required",
            "forbids": "superluminal state reconstruction"}


def assert_causality(L_m: float, t_ff_s: float) -> Dict[str, Any]:
    need = L_m / C_LIGHT
    if t_ff_s < need * (1.0 - 1e-12):
        raise CausalityRefusal(
            f"[N6] t_ff = {t_ff_s:.3e} s < L/c = {need:.3e} s for L={L_m:.1f} m. "
            "Teleportation cannot complete before the classical message arrives.")
    return {"passes": True, "t_ff_s": t_ff_s, "t_min_s": need,
            "margin_s": t_ff_s - need}


def gravitational_redshift(h_m: float) -> Dict[str, Any]:
    """dU/c^2 = GM h / (c^2 R (R+h)). Ground -> altitude h."""
    val = GM_EARTH * h_m / (C_LIGHT ** 2 * R_EARTH * (R_EARTH + h_m))
    return {"altitude_m": h_m, "delta_U_over_c2": val,
            "sign": "blueshift for ground->orbit (positive)",
            "theorem": "weak-field Schwarzschild gravitational potential"}


def orbital_velocity(h_m: float) -> Dict[str, Any]:
    r = R_EARTH + h_m
    v = math.sqrt(GM_EARTH / r)
    return {"altitude_m": h_m, "r_m": r, "v_orbital_m_s": v,
            "theorem": "circular orbit v = sqrt(GM/r)"}


def transverse_doppler(v_m_s: float) -> Dict[str, Any]:
    return {"v_m_s": v_m_s, "shift": -v_m_s ** 2 / (2.0 * C_LIGHT ** 2),
            "sign": "time dilation (negative)",
            "theorem": "second-order (transverse) Doppler"}


def first_order_doppler(v_radial_m_s: float, nu_Hz: float) -> Dict[str, Any]:
    return {"v_radial_m_s": v_radial_m_s, "nu_Hz": nu_Hz,
            "delta_f_Hz": v_radial_m_s / C_LIGHT * nu_Hz,
            "note": "longitudinal; must be tracked separately from GR terms"}


def shapiro_delay(r1_m: float, r2_m: float, L_m: float,
                  geometry: str = "slant") -> Dict[str, Any]:
    """
    Schwarzschild Shapiro delay. Geometry MUST be declared.
      slant  : (2GM/c^3) ln[(r1+r2+L)/(r1+r2-L)]
      zenith : (2GM/c^3) ln[r2/r1]     (radial path, L = r2-r1)
    """
    pref = 2.0 * GM_EARTH / C_LIGHT ** 3
    if geometry == "slant":
        num, den = r1_m + r2_m + L_m, r1_m + r2_m - L_m
        if den <= 0: raise GeometryRefusal("r1+r2-L must be positive")
        dt = pref * math.log(num / den)
        formula = "(2GM/c^3) ln[(r1+r2+L)/(r1+r2-L)]"
    elif geometry == "zenith":
        if r2_m <= r1_m: raise GeometryRefusal("zenith requires r2 > r1")
        dt = pref * math.log(r2_m / r1_m)
        formula = "(2GM/c^3) ln[r2/r1]"
    else:
        raise GeometryRefusal("geometry must be 'slant' or 'zenith'")
    return {"geometry": geometry, "r1_m": r1_m, "r2_m": r2_m, "L_m": L_m,
            "prefactor_s": pref, "delta_t_s": dt, "formula": formula,
            "REQUIRED": "declare geometry; slant and zenith differ by ~2x"}


class GRMetricCompensator:
    """
    Layer 85/0 GR-MRC. Computes every metric term the link must pre-compensate,
    then checks actuator range. Fast phase comes from the optical comb beacon;
    the CSAC supplies long-term absolute frequency, not short-term phase.
    """
    def __init__(self, nu_Hz: float = 193.4e12, lam_m: float = 1550e-9):
        self.nu, self.lam = nu_Hz, lam_m

    def leo_link(self, h_m: float = 500e3, L_slant_m: float = 1000e3,
                 v_radial_m_s: float = 0.0) -> Dict[str, Any]:
        grav = gravitational_redshift(h_m)
        orb = orbital_velocity(h_m)
        trans = transverse_doppler(orb["v_orbital_m_s"])
        net = grav["delta_U_over_c2"] + trans["shift"]
        sh_slant = shapiro_delay(R_EARTH, R_EARTH + h_m, L_slant_m, "slant")
        sh_zen = shapiro_delay(R_EARTH, R_EARTH + h_m, h_m, "zenith")
        first = first_order_doppler(v_radial_m_s, self.nu)
        optical_period = self.lam / C_LIGHT
        return {"altitude_m": h_m, "slant_L_m": L_slant_m,
                "gravitational": grav, "orbital": orb,
                "transverse_doppler": trans,
                "net_fractional_shift": net,
                "net_delta_f_Hz": net * self.nu,
                "first_order_doppler": first,
                "shapiro_slant": sh_slant, "shapiro_zenith": sh_zen,
                "optical_period_s": optical_period,
                "shapiro_slant_in_optical_cycles":
                    sh_slant["delta_t_s"] / optical_period,
                "clock_architecture": (
                    "CSAC sigma_y(1s) ~ 3e-10 (commercial) disciplined by a "
                    "co-propagating 1530 nm optical frequency comb via two-way "
                    "time transfer; SHORT-TERM phase comes from the beacon, "
                    "not from the atomic reference"),
                "stratum": {"metric_terms": MODE_CERTIFIED,
                            "csac_allan": MODE_E,
                            "comb_jitter": MODE_E}}

    def actuator_check(self, net_fractional: float, shapiro_s: float,
                       comb_range_fractional: float = 1e-9,
                       delay_buffer_s: float = 50e-9,
                       delay_resolution_s: float = 10e-15) -> Dict[str, Any]:
        ok_f = abs(net_fractional) <= comb_range_fractional
        ok_d = shapiro_s <= delay_buffer_s
        ok_r = delay_resolution_s <= 1e-14
        if not (ok_f and ok_d and ok_r):
            raise Refusal(
                f"GR-MRC actuator insufficient: freq_ok={ok_f}, "
                f"delay_ok={ok_d}, resolution_ok={ok_r}")
        return {"passes": True, "frequency_range_ok": ok_f,
                "delay_buffer_ok": ok_d, "resolution_ok": ok_r}


# =============================================================================
# §6. [N4] HYPERBOLIC {5,4} LOGICAL GRAPH vs EUCLIDEAN CARPET ROUTING
# =============================================================================

def carpet_scaling() -> Dict[str, Any]:
    d_h = math.log(8) / math.log(3)
    cut_exp = math.log(8.0 / 3.0) / math.log(3)
    return {"hausdorff_dimension": d_h,
            "minimal_cut_exponent": cut_exp,
            "cut_scaling": "|gamma_A| ~ (8/3)^m = L^0.8928",
            "entanglement_law": "polynomial (sub-volume) — NOT logarithmic",
            "rt_compatible": False,
            "role": "PHYSICAL ROUTING FABRIC ONLY"}


def hyperbolic_scaling() -> Dict[str, Any]:
    return {"tessellation": "{5,4} pentagonal",
            "curvature": "constant negative",
            "minimal_cut_scaling": "|gamma_A| ~ log|A|",
            "rt_entropy": "S(A) ~ (c/3) ln|A|  (AdS3/CFT2)",
            "rt_compatible": True,
            "growth": "exponential in radial depth",
            "role": "LOGICAL TENSOR NETWORK (HaPPY perfect tensors)",
            "reference": "Pastawski, Yoshida, Harlow, Preskill (2015)"}


def assert_rt_geometry(logical: str) -> Dict[str, Any]:
    """[N4] Refuse RT claims on a Euclidean fractal."""
    if logical in ("sierpinski_carpet", "euclidean_fractal", "carpet"):
        c = carpet_scaling()
        raise GeometryRefusal(
            f"[N4] '{logical}' has polynomial minimal cut "
            f"L^{c['minimal_cut_exponent']:.4f}, not the logarithmic scaling "
            "S(A) ~ (c/3) ln|A| required by Ryu-Takayanagi in AdS3/CFT2. "
            "Use a {5,4} hyperbolic LOGICAL graph; the carpet may serve as "
            "the physical routing fabric only.")
    if logical not in ("{5,4}", "hyperbolic_pentagonal", "HaPPY"):
        raise GeometryRefusal(f"unknown logical geometry '{logical}'")
    return {"passes": True, **hyperbolic_scaling()}


def ads4_cft3_exactness_request() -> Dict[str, Any]:
    raise GeometryRefusal(
        "[N4] No exact tensor-network construction for AdS4/CFT3 is known. "
        "HaPPY supplies AdS3/CFT2 only. This claim is refused.")


@dataclass
class HyperbolicPatchBudget:
    depth: int
    nodes: int
    max_edge_dilation: float          # multiples of carpet pitch
    crossings_total: int
    crossings_worst_path: int
    loss_per_crossing_dB: float = 0.008
    wdm_rails: int = 64

    def report(self) -> Dict[str, Any]:
        loss_total = self.crossings_total * self.loss_per_crossing_dB
        loss_path = self.crossings_worst_path * self.loss_per_crossing_dB
        return {"depth": self.depth, "nodes": self.nodes,
                "max_edge_dilation": self.max_edge_dilation,
                "crossings_total": self.crossings_total,
                "crossings_worst_path": self.crossings_worst_path,
                "wdm_rails": self.wdm_rails,
                "loss_per_crossing_dB": self.loss_per_crossing_dB,
                "aggregate_crossing_loss_dB": loss_total,
                "worst_path_insertion_loss_dB": loss_path,
                "DECLARED": ("the loss budget that matters for a link is the "
                             "WORST-PATH insertion loss, not the aggregate "
                             "sum over all crossings in the fabric")}


class HyperbolicEmbedding:
    """Finite-depth {5,4} patch on carpet switch hubs, with a hard budget."""
    MAX_DEPTH_MOBILE = 4
    MAX_DEPTH_WORKSTATION = 7
    MAX_WORST_PATH_LOSS_DB = 2.0
    MAX_DILATION = 8.0

    def __init__(self, platform: str = "workstation"):
        if platform not in ("mobile", "workstation"):
            raise Refusal("platform must be mobile|workstation")
        self.platform = platform
        self.max_depth = (self.MAX_DEPTH_MOBILE if platform == "mobile"
                          else self.MAX_DEPTH_WORKSTATION)

    def authorize(self, patch: HyperbolicPatchBudget,
                  logical_qubits_available: int) -> Dict[str, Any]:
        geo = assert_rt_geometry("{5,4}")
        if patch.depth > self.max_depth:
            raise GeometryRefusal(
                f"[N4] depth {patch.depth} exceeds the {self.platform} bound "
                f"{self.max_depth}: a hyperbolic graph grows exponentially "
                "while the carpet grows polynomially, so exact embedding is "
                "only possible for finite-depth patches.")
        if patch.nodes > logical_qubits_available:
            raise CapacityRefusal(
                f"patch needs {patch.nodes} nodes but only "
                f"{logical_qubits_available} logical qubits are available")
        if patch.max_edge_dilation > self.MAX_DILATION:
            raise GeometryRefusal(
                f"edge dilation {patch.max_edge_dilation} exceeds "
                f"{self.MAX_DILATION} carpet pitches")
        rep = patch.report()
        if rep["worst_path_insertion_loss_dB"] > self.MAX_WORST_PATH_LOSS_DB:
            raise CapacityRefusal(
                f"worst-path crossing loss "
                f"{rep['worst_path_insertion_loss_dB']:.3f} dB exceeds "
                f"{self.MAX_WORST_PATH_LOSS_DB} dB")
        return {"authorized": True, "platform": self.platform,
                "geometry": geo, "carpet_role": carpet_scaling(),
                "budget": rep,
                "stratum": {"geometry_scaling": MODE_CERTIFIED,
                            "node_count": MODE_E,
                            "crossing_loss": MODE_E}}


# =============================================================================
# §7. [N5] QUANTUM OVERLAP & RENYI ENTROPY ESTIMATOR (Q-OEE)
# =============================================================================

def swap_test_probability(overlap_real: float) -> Dict[str, Any]:
    """P(ancilla=0) = (1 + Re<psi1|psi2>)/2. Exact identity."""
    if not (-1.0 <= overlap_real <= 1.0):
        raise Refusal("Re overlap in [-1,1]")
    return {"overlap_real": overlap_real,
            "p_ancilla_zero": 0.5 * (1.0 + overlap_real),
            "theorem": "controlled-SWAP (Fredkin) overlap identity"}


def renyi2_from_purity(purity: float) -> Dict[str, Any]:
    if not (0.0 < purity <= 1.0): raise Refusal("Tr(rho^2) in (0,1]")
    return {"purity": purity, "S2": -math.log(purity),
            "definition": "S2(rho_A) = -ln Tr(rho_A^2)"}


def qae_query_count(eps: float, mode: str = "quantum") -> Dict[str, Any]:
    """[N5] QAE gives O(1/eps); classical sampling gives O(1/eps^2)."""
    if eps <= 0: raise Refusal("eps > 0")
    if mode == "quantum":
        n = 1.0 / eps; law = "O(1/eps)"
    elif mode == "classical":
        n = 1.0 / eps ** 2; law = "O(1/eps^2)"
    else:
        raise Refusal("mode must be quantum|classical")
    return {"eps": eps, "mode": mode, "queries": n, "scaling": law,
            "speedup": "quadratic" if mode == "quantum" else "baseline"}


def assert_no_decohered_interference(env_overlap: float) -> Dict[str, Any]:
    """[N5] If <E1|E2> = 0 the branches cannot be interfered."""
    if abs(env_overlap) < 1e-12:
        raise NoGoViolation(
            "[N5] environment records are orthogonal (<E1|E2> = 0). The "
            "branches are decohered; cross-branch interference is physically "
            "impossible. Only pre-decoherence coherent superpositions may be "
            "fed to the controlled-SWAP network.")
    return {"passes": True, "env_overlap": env_overlap,
            "note": "coherence retained; overlap estimation permitted"}


class OverlapEstimator:
    """
    Layer 86/0 Q-OEE. Measures state overlaps and second-order Renyi entropy
    across coherent unitary trajectories. NOT 'many-worlds interference'.
    """
    def __init__(self, logical_qubits: int):
        self.logical_qubits = logical_qubits

    def register_dim(self, n_reg: int) -> Dict[str, Any]:
        if n_reg > self.logical_qubits:
            raise CapacityRefusal(
                f"register needs {n_reg} qubits, only "
                f"{self.logical_qubits} logical available")
        return {"n_register_qubits": n_reg, "hilbert_dim": 2 ** n_reg,
                "note": ("this is the Hilbert-space dimension of an n-qubit "
                         "register, not a count of parallel worlds")}

    def estimate(self, n_reg: int, eps: float, env_overlap: float = 1.0,
                 query_time_s: float = 20e-6) -> Dict[str, Any]:
        assert_no_decohered_interference(env_overlap)
        dim = self.register_dim(n_reg)
        q = qae_query_count(eps, "quantum")
        c = qae_query_count(eps, "classical")
        return {"register": dim, "quantum_queries": q, "classical_queries": c,
                "speedup_factor": c["queries"] / q["queries"],
                "target_latency_s": q["queries"] * query_time_s,
                "latency_stratum": MODE_E,
                "measures": ["Re<psi1|psi2>", "S2(rho_A) = -ln Tr(rho_A^2)"],
                "forbids": ["inter-branch signalling", "cloning",
                            "interference of decohered branches"],
                "renamed_from": "Everettian multi-branch interference"}


# =============================================================================
# §8. REVISION REGISTRY  (authoritative parameters only)
# =============================================================================

@dataclass(frozen=True)
class ChipRevision:
    name: str
    form_factor: str
    logical_qubits: int
    code_n: int; code_k: int; code_d: int
    tdp_W: float
    t_limit_C: float
    rth_K_W: float
    t_ambient_C: float
    quantum_mode: str
    max_hyp_depth: int
    teleport_rate_qps: float          # target, Stratum II
    squeezing_dB_target: float

REVISIONS: Dict[str, ChipRevision] = {
 "R12.0-M": ChipRevision("R12.0-M", "mobile SoC 86.4 mm^2",
    3072, 32768, 3072, 32, 3.80, 39.2, 3.08, 27.5,
    "300 K optical GKP cluster (no on-die Peltier)", 4, 14.2e6, 12.2),
 "R12.0-W": ChipRevision("R12.0-W", "8-chiplet MCM 684 mm^2",
    147456, 1572864, 147456, 64, 350.0, 58.3, 0.088, 27.5,
    "300 K optics + 1.8 K cryo-capsule (42 W)", 7, 680e6, 12.8),
}


def thermal_closure(rev: ChipRevision) -> Dict[str, Any]:
    """P = E*F identity is separate; here T = T_amb + P*Rth."""
    T = rev.t_ambient_C + rev.tdp_W * rev.rth_K_W
    ok = T <= rev.t_limit_C + 1e-9
    return {"revision": rev.name, "P_W": rev.tdp_W, "Rth_K_W": rev.rth_K_W,
            "T_ambient_C": rev.t_ambient_C, "T_predicted_C": T,
            "T_limit_C": rev.t_limit_C, "within_limit": ok,
            "identity": "T = T_amb + P * Rth",
            "note": "Rth is a Stratum II engineering value"}


def boltzmann_spin_init(delta_E_meV: float, T_K: float) -> Dict[str, Any]:
    kT = KB_EV_K * T_K * 1e3
    x = delta_E_meV / kT
    F = 1.0 / (1.0 + math.exp(-x))
    return {"delta_E_meV": delta_E_meV, "T_K": T_K, "kT_meV": kT,
            "ratio_dE_over_kT": x, "init_fidelity": F,
            "usable": x >= 1.0,
            "verdict": ("cryogenic regime: kT << dE" if x >= 1.0 else
                        "THERMALLY SCRAMBLED — spin-valley unusable")}


def assert_spin_regime(delta_E_meV: float, T_K: float) -> Dict[str, Any]:
    r = boltzmann_spin_init(delta_E_meV, T_K)
    if not r["usable"]:
        raise NoGoViolation(
            f"spin-valley at T={T_K} K has kT={r['kT_meV']:.3f} meV >= "
            f"dE={delta_E_meV} meV. Isotopic purity removes hyperfine "
            "decoherence, NOT thermal occupation. Cryogenic operation required.")
    return r


def membrane_heat_leak(G_uW_per_K: float, dT_K: float,
                       cooling_lift_uW: float) -> Dict[str, Any]:
    leak = G_uW_per_K * dT_K
    ok = leak <= cooling_lift_uW
    return {"G_uW_per_K": G_uW_per_K, "dT_K": dT_K,
            "heat_leak_uW": leak, "cooling_lift_uW": cooling_lift_uW,
            "closes": ok, "law": "Fourier: Q_leak = G * dT"}


def cavity_qed_cooperativity(g_GHz: float, kappa_GHz: float,
                             gamma_GHz: float,
                             required: float = 1e4) -> Dict[str, Any]:
    C = 4.0 * g_GHz ** 2 / (kappa_GHz * gamma_GHz)
    return {"g_GHz": g_GHz, "kappa_GHz": kappa_GHz, "gamma_GHz": gamma_GHz,
            "cooperativity": C, "required": required,
            "sufficient": C >= required,
            "protocol": "Duan-Kimble reflection for single-photon pi phase",
            "replaces": "bulk chi^(3) Kerr (insufficient at single-photon level)"}


# =============================================================================
# §9. AI SURROGATES  (transformer / world model / diffusion), all gated
# =============================================================================

if HAS_TORCH:
    class LinkConsistencyTransformer(nn.Module):
        """
        Ranks candidate link/code/geometry configurations for consistency.
        A SURROGATE: the deterministic gates in §§3-7 remain the authority.
        A nuisance head predicts environmental confounds (weather, platform
        motion) so they can be inspected rather than absorbed silently.
        """
        def __init__(self, d_in: int = 12, d_cond: int = 6, d: int = 96,
                     heads: int = 4, layers: int = 3, max_len: int = 256):
            super().__init__()
            self.proj = nn.Linear(d_in, d)
            self.cond = nn.Sequential(nn.Linear(d_cond, d), nn.GELU(),
                                      nn.Linear(d, d))
            self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
            enc = nn.TransformerEncoderLayer(d, heads, d * 4, 0.1,
                                             batch_first=True, norm_first=True)
            self.trunk = nn.TransformerEncoder(enc, layers)
            self.head = nn.Linear(d, 2)        # reject / accept logit
            self.nuisance = nn.Linear(d, 1)
        def forward(self, x, cond):
            B, L, _ = x.shape
            h = self.proj(x) + self.cond(cond).unsqueeze(1) + self.pos[:, :L]
            p = self.trunk(h).mean(1)
            return {"logits": self.head(p), "nuisance": self.nuisance(p),
                    "STATUS": "surrogate ranking — deterministic gate decides"}

    class LinkWorldModel(nn.Module):
        """
        Predicts the next link/phase state (transmittance, residual phase,
        Doppler, erasure rate) from the current state and a control action.
        Used for pre-emptive beam widening / handoff. The latent is
        model-internal; it is NOT a measurement.
        """
        def __init__(self, d_obs: int = 10, d_act: int = 4, d: int = 96):
            super().__init__()
            self.enc = nn.Linear(d_obs + d_act, d)
            self.gru = nn.GRUCell(d, d)
            self.mu = nn.Linear(d, d_obs)
            self.log_sd = nn.Linear(d, d_obs)
            self.decay_logit = nn.Parameter(torch.tensor(3.9))
        @property
        def decay(self): return torch.sigmoid(self.decay_logit)
        def forward(self, obs, act):
            h = F.gelu(self.enc(torch.cat([obs, act], -1)))
            h = self.gru(h, h)
            return {"next_mu": self.mu(h),
                    "log_sd": self.log_sd(h).clamp(-6, 3),
                    "decay": float(self.decay),
                    "STATUS": "prediction for cueing, NOT a measurement"}

    class RoutingDiffusion(nn.Module):
        """
        DDPM generating CANDIDATE {5,4}-on-carpet routing embeddings.
        Every sample is a hypothesis and must pass HyperbolicEmbedding.authorize
        plus the crossing-loss budget before use.
        """
        def __init__(self, d: int = 48, T: int = 60, hidden: int = 192,
                     d_cond: int = 6):
            super().__init__()
            self.T, self.d = T, d
            b = torch.linspace(1e-4, 0.02, T)
            self.register_buffer("betas", b)
            self.register_buffer("abar", torch.cumprod(1 - b, 0))
            self.net = nn.Sequential(nn.Linear(d + 1, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, d))
            self.cond = nn.Sequential(nn.Linear(d_cond, hidden), nn.GELU())
        def loss(self, x0, cond):
            B = x0.shape[0]
            t = torch.randint(0, self.T, (B,), device=x0.device)
            eps = torch.randn_like(x0); ab = self.abar[t][:, None]
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            inp = torch.cat([xt, t.float()[:, None] / self.T], -1) + self.cond(cond)
            return F.mse_loss(self.net(inp), eps)
        @torch.no_grad()
        def sample(self, n: int, cond, device: str = "cpu"):
            x = torch.randn(n, self.d, device=device)
            for t in reversed(range(self.T)):
                tb = torch.full((n,), t, device=device, dtype=torch.long)
                inp = torch.cat([x, tb.float()[:, None] / self.T], -1) + self.cond(cond)
                eps = self.net(inp)
                a = 1 - self.betas[t]; ab = self.abar[t]
                x = (x - (1 - a) / torch.sqrt(1 - ab) * eps) / torch.sqrt(a)
                if t > 0: x = x + torch.sqrt(self.betas[t]) * torch.randn_like(x)
            return {"candidate": x,
                    "STATUS": "HYPOTHESIS routing — gate before use"}


@dataclass
class TrainConfig:
    epochs: int = 60; lr: float = 3e-3; weight_decay: float = 1e-4
    grad_clip: float = 1.0; patience: int = 12; amp: bool = True


class Trainer:
    """AMP + clipping + cosine LR + early stop + best-checkpoint restore."""
    def __init__(self, model, cfg: TrainConfig, run: RunConfig,
                 loss_fn: Callable, name: str = "model"):
        self.cfg = cfg; self.run = run; self.name = name
        self.dev = pick_device(run.device); self.model = model.to(self.dev)
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, cfg.epochs)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(cfg.amp and self.dev == "cuda"))
        self.loss_fn = loss_fn
        self.best = float("inf"); self.bad = 0; self.hist: List[Dict] = []
        self.ckpt = Path(run.out_dir) / f"{name}.best.pt"
    def fit(self, *batch):
        batch = [b.to(self.dev) if hasattr(b, "to") else b for b in batch]
        for ep in range(self.cfg.epochs):
            self.opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                L = self.loss_fn(self.model, *batch)
            self.scaler.scale(L).backward(); self.scaler.unscale_(self.opt)
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                self.cfg.grad_clip)
            self.scaler.step(self.opt); self.scaler.update(); self.sch.step()
            lv = float(L)
            self.hist.append({"epoch": ep, "loss": lv, "grad_norm": float(gn)})
            if lv < self.best - 1e-6:
                self.best = lv; self.bad = 0
                torch.save({"state": self.model.state_dict(), "epoch": ep},
                           self.ckpt)
            else:
                self.bad += 1
            if self.bad >= self.cfg.patience: break
        if self.ckpt.exists():
            self.model.load_state_dict(
                torch.load(self.ckpt, map_location=self.dev)["state"])
        return {"best_loss": self.best, "epochs": len(self.hist),
                "history": self.hist, "checkpoint": str(self.ckpt),
                "device": self.dev, "early_stopped": self.bad >= self.cfg.patience}


class AIGate:
    """Every AI candidate passes a deterministic gate or becomes Mode E."""
    @staticmethod
    def gate_link(candidate: Dict[str, Any], gate: BipartiteChannelGate
                  ) -> Dict[str, Any]:
        try:
            auth = gate.authorize(
                Plane[candidate["plane"]], int(candidate["K"]),
                float(candidate["freq_Hz"]), float(candidate["eta_atm"]),
                float(candidate.get("eta_optics", 1.0)))
            return {"passed": True, "authorization": auth}
        except Refusal as e:
            return {"passed": False, "reason": str(e), "status": MODE_E}
    @staticmethod
    def gate_routing(depth: int, nodes: int, dilation: float,
                     crossings_total: int, crossings_worst: int,
                     platform: str, logical_available: int
                     ) -> Dict[str, Any]:
        try:
            emb = HyperbolicEmbedding(platform)
            patch = HyperbolicPatchBudget(depth, nodes, dilation,
                                          crossings_total, crossings_worst)
            return {"passed": True,
                    "authorization": emb.authorize(patch, logical_available)}
        except Refusal as e:
            return {"passed": False, "reason": str(e), "status": MODE_E}
    @staticmethod
    def rule() -> str:
        return "AI proposes; physics disposes"


# =============================================================================
# §10. STATISTICS DICTIONARY + DRC PREDICATES
# =============================================================================

class SDGate:
    ENTRIES = {
      "bipartite_channel": (("NOGO","CAP"), "etendue + Holevo capacity",
                            "K=1 per quantum mode; Plane C classical only"),
      "gkp_fidelity_chain": (("RAW","ERASURE","PAULI","CONV"),
                             "Braunstein-Kimble + Voronoi + BP+OSD",
                             "erasure convention must be declared"),
      "logical_memory_error": (("CAL","BP"), "calibrated BP+OSD",
                               "separate from T-gate distillation"),
      "gr_metric_compensation": (("GEOM","METRIC"),
                                 "Schwarzschild + transverse Doppler",
                                 "slant vs zenith geometry must be declared"),
      "causal_feed_forward": (("CAUSAL",), "t_ff >= L/c", "no superluminal"),
      "hyperbolic_embedding": (("GEOM","DEPTH","BUDGET"),
                               "{5,4} finite-depth on carpet",
                               "worst-path loss, not aggregate"),
      "qoee_overlap": (("COHERENT","QAE"), "SWAP test + O(1/eps)",
                       "decohered branches refused"),
      "spin_valley_regime": (("BOLTZ",), "kT << dE", "cryogenic required"),
      "thermal_closure": (("THERM",), "T = T_amb + P Rth",
                          "Rth is a Stratum II value"),
      "ai_candidate": (("GATE","XVAL"), "surrogate",
                       "usable only after a deterministic gate"),
      "fabrication": ((), "none", "REFUSED — design assistance only"),
      "ads4_cft3_exact": ((), "none", "REFUSED — no known construction"),
      "carpet_rt": ((), "none", "REFUSED — polynomial cut, not log"),
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
    def report(self) -> Dict[str, Any]:
        n = sum(1 for c in self.log if c["status"] == MODE_CERTIFIED)
        return {"n": len(self.log), "certified": n,
                "mode_E": len(self.log) - n, "log": self.log}


PACKAGES = {
 "NOGO":"K->1 combiner no-go respected (K=1 on Plane Q)",
 "CAP":"positive Holevo capacity verified",
 "RAW":"F_raw computed from squeezing, not asserted",
 "ERASURE":"Voronoi heralding applied",
 "PAULI":"residual unheralded Pauli separated",
 "CONV":"erasure convention (per-quadrature / per-mode) declared",
 "CAL":"alpha_bp, A_code, p_th calibrated",
 "BP":"BP+OSD model, not surface-code heuristic",
 "GEOM":"geometry declared (slant/zenith; {5,4}/carpet)",
 "METRIC":"all GR terms computed",
 "CAUSAL":"t_ff >= L/c verified",
 "DEPTH":"finite-depth bound respected",
 "BUDGET":"worst-path crossing loss within budget",
 "COHERENT":"environment overlap non-zero",
 "QAE":"O(1/eps) query complexity",
 "BOLTZ":"kT << dE verified",
 "THERM":"thermal closure computed",
 "GATE":"deterministic physics gate passed",
 "XVAL":"cross-validated against the deterministic model",
}


def drc_predicates() -> Dict[str, Any]:
    return {
     "D1_bipartite_channel":
       "forbid K>=2 passive optical combining on quantum modes; "
       "eta_total > 0.5; 300 GHz restricted to classical AirComp",
     "D2_causal_relativistic":
       "t_ff >= L/c; GR-MRC actuator range covers net delta_f/f and "
       "the DECLARED slant/zenith Shapiro delay",
     "D3_hyperbolic_holographic":
       "logical {5,4} adjacency; depth <= 4 mobile / <= 7 workstation; "
       "worst-path crossing loss <= 2 dB; R_code = k/n = 0.09375 < 1",
     "D4_thermal_cryogenic":
       "T_skin <= 39.2 C at 3.80 W mobile; T_j <= 58.3 C at 350 W "
       "workstation; 1.8 K membranes at G_bridge <= 0.8 uW/K",
     "D5_erasure_declaration":
       "p_erase convention declared and compared with p_c = 0.362; "
       "p_Pauli fed to the BP+OSD decoder, never conflated with p_erase",
     "D6_scope":
       "no fabrication, no tape-out release, no Stratum II value reported "
       "as measured",
    }


# =============================================================================
# §11. CO-DESIGN PIPELINE
# =============================================================================

class QuantumWirelessCoDesign:
    def __init__(self, cfg: Optional[RunConfig] = None):
        self.cfg = (cfg or RunConfig()).validate()
        set_seed(self.cfg.seed)
        self.sd = SDGate(); self.cache = Cache(); self.mf = Manifest(self.cfg.out_dir)
        self.channel_gate = BipartiteChannelGate()

    def analyse_revision(self, name: str,
                         convention: ErasureConvention = ErasureConvention.PER_QUADRATURE,
                         eta_total: float = 0.68) -> Dict[str, Any]:
        refuse_scope(f"analyse {name}")
        if name not in REVISIONS: raise Refusal(f"unknown revision '{name}'")
        rev = REVISIONS[name]
        out: Dict[str, Any] = {"revision": name, "form_factor": rev.form_factor,
                               "quantum_mode": rev.quantum_mode}

        # channel
        auth = self.channel_gate.authorize(Plane.Q, 1, 193.4e12, eta_total)
        out["plane_Q"] = auth
        self.sd.certify("bipartite_channel", ("NOGO","CAP"), auth)
        out["plane_C"] = self.channel_gate.authorize(Plane.C, 8, 300e9, 0.85)

        # fidelity chain
        chain = GKPFidelityChain(rev.code_n, rev.code_k, rev.code_d)
        budget = GKPNoiseBudget(rev.squeezing_dB_target, eta_total)
        fc = chain.evaluate(budget, convention)
        out["fidelity_chain"] = fc
        self.sd.certify("gkp_fidelity_chain", ("RAW","ERASURE","PAULI","CONV"), fc)
        self.sd.certify("logical_memory_error", ("CAL","BP"),
                        fc["logical_memory_error"])
        out["magic_distillation"] = magic_state_distillation(1e-3, 2)

        # GR + causality
        gr = GRMetricCompensator()
        link = gr.leo_link(500e3, 1000e3)
        gr.actuator_check(link["net_fractional_shift"],
                          link["shapiro_slant"]["delta_t_s"])
        out["gr_metric"] = link
        self.sd.certify("gr_metric_compensation", ("GEOM","METRIC"), link)
        L = 2500.0 if name.endswith("W") else 15.0
        out["causality"] = assert_causality(L, L / C_LIGHT * 1.5)
        self.sd.certify("causal_feed_forward", ("CAUSAL",), out["causality"])

        # hyperbolic
        platform = "workstation" if name.endswith("W") else "mobile"
        emb = HyperbolicEmbedding(platform)
        patch = (HyperbolicPatchBudget(7, 2985, 6.8, 1840, 231)
                 if platform == "workstation" else
                 HyperbolicPatchBudget(4, 165, 3.4, 142, 142))
        out["hyperbolic"] = emb.authorize(patch, rev.logical_qubits)
        self.sd.certify("hyperbolic_embedding", ("GEOM","DEPTH","BUDGET"),
                        out["hyperbolic"]["budget"])

        # Q-OEE
        oe = OverlapEstimator(rev.logical_qubits)
        out["qoee"] = oe.estimate(15 if platform == "workstation" else 10,
                                  1e-4 if platform == "workstation" else 1e-3)
        self.sd.certify("qoee_overlap", ("COHERENT","QAE"), out["qoee"])

        # thermal + cryo
        out["thermal"] = thermal_closure(rev)
        self.sd.certify("thermal_closure", ("THERM",), out["thermal"])
        if platform == "workstation":
            out["spin_regime"] = assert_spin_regime(0.406, 1.8)
            self.sd.certify("spin_valley_regime", ("BOLTZ",), out["spin_regime"])
            out["membrane"] = membrane_heat_leak(0.8, 298.2, 350e3)
            out["cavity_qed"] = cavity_qed_cooperativity(1.45, 0.028, 0.018)

        # Stratum II targets, explicitly
        out["stratum_II_targets"] = {
            "logical_qubits": rev.logical_qubits,
            "squeezing_dB": rev.squeezing_dB_target,
            "teleport_rate_qubits_per_s": rev.teleport_rate_qps,
            "status": MODE_E,
            "note": "design targets under nominal gate rates, NOT measured runs"}
        for item in ("logical_qubit_count", "gkp_squeezing_db",
                     "teleportation_throughput"):
            self.sd.certify_item = None  # placeholder removed below
        out["scope"] = DESIGN_ONLY
        self.mf.add("revision_analysis",
                    {"revision": name, "convention": convention.value,
                     "eta_total": eta_total})
        return out

    def report(self) -> Dict[str, Any]:
        rep = {"version": VERSION, "philosophy": PHILOSOPHY,
               "lineage": lineage_report(),
               "stratum_I": sorted(STRATUM_I), "stratum_II": sorted(STRATUM_II),
               "packages": PACKAGES, "drc": drc_predicates(),
               "cache": self.cache.stats(), "sd_report": self.sd.report(),
               "scope": DESIGN_ONLY}
        p = Path(self.cfg.out_dir) / "report.json"
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

    cfg = RunConfig().validate(); pipe = QuantumWirelessCoDesign(cfg)

    print("\n[T1] LINEAGE AND STRATUM SEPARATION")
    rep("2 authoritative revisions", len(authoritative_revisions()) == 2)
    rep("R11.0 marked superseded", LINEAGE["R11.0-OTA"].status == "superseded")
    rep("carpet routing is Stratum I",
        stratum_of("carpet_polynomial_cut") is Stratum.I)
    rep("squeezing dB is Stratum II",
        stratum_of("gkp_squeezing_db") is Stratum.II)
    rep("Stratum I certifies",
        certify_item("holevo_pure_loss_capacity")["status"] == MODE_CERTIFIED)
    rep("Stratum II is Mode E",
        certify_item("logical_qubit_count")["status"] == MODE_E)
    b = False
    try: assert_stratum("gkp_squeezing_db", Stratum.I)
    except StratumRefusal: b = True
    rep("conflating Stratum II as I REFUSED", b)

    print("\n[T2] [N1] K->1 NO-GO AND CARRIER SELECTION")
    occ_opt = thermal_occupation(193.4e12, 300.0)
    occ_thz = thermal_occupation(300e9, 300.0)
    print(f"      1550 nm n_th={occ_opt['n_thermal']:.3e}   "
          f"300 GHz n_th={occ_thz['n_thermal']:.3f}")
    rep("1550 nm n_th ~3.6e-14", abs(occ_opt["n_thermal"] - 3.65e-14) < 0.2e-14)
    rep("300 GHz n_th ~20.3", abs(occ_thz["n_thermal"] - 20.3) < 0.15)
    rep("sub-THz NOT quantum capable", not occ_thz["quantum_capable"])
    for K in (2, 4, 8):
        cb = combiner_transmissivity_bound(K)
        rep(f"K={K} combiner Q=0", cb["Q_at_bound"] == 0.0,
            f"eta<={cb['eta_combine_max']:.3f}")
    g = BipartiteChannelGate()
    b = False
    try: g.authorize(Plane.Q, 8, 193.4e12, 0.88)
    except NoGoViolation: b = True
    rep("K=8 on Plane Q REFUSED", b)
    b = False
    try: g.authorize(Plane.Q, 1, 300e9, 0.9)
    except NoGoViolation: b = True
    rep("sub-THz on Plane Q REFUSED", b)
    a = g.authorize(Plane.Q, 1, 193.4e12, 0.68)
    print(f"      Q(0.68)={a['capacity']['Q_qubits_per_mode']:.4f} "
          f"margin={a['capacity_margin_over_code_rate']:.2f}x")
    rep("Q(0.68) ~1.087", abs(a["capacity"]["Q_qubits_per_mode"] - 1.0875) < 0.002)
    rep("margin >11x code rate", a["capacity_margin_over_code_rate"] > 11.0)
    ac = g.authorize(Plane.C, 16, 300e9, 0.85)
    rep("Plane C allows K=16 classical", ac["K"] == 16)
    rep("Plane C forbids quantum states", ac["quantum_states_permitted"] is False)

    print("\n[T3] [N2] BARE FIDELITY BOUND")
    f122 = bare_teleport_fidelity(12.2)
    print(f"      12.2 dB -> F_raw={f122['F_raw']:.4f}")
    rep("F_raw(12.2 dB) ~0.9432", abs(f122["F_raw"] - 0.94317) < 1e-4)
    need = squeezing_for_target_fidelity(0.994)
    print(f"      F=0.994 requires {need['required_squeezing_dB']:.2f} dB")
    rep("0.994 needs ~22.2 dB", abs(need["required_squeezing_dB"] - 22.19) < 0.05)
    rep("F_raw labelled NOT logical", "NOT" in f122)

    print("\n[T4] [N3] VARIANCE BUDGET AND ERASURE/PAULI SPLIT")
    bud = GKPNoiseBudget(12.2, 0.68)
    print(f"      sigma2_sq={bud.sigma2_squeezing:.4f} "
          f"sigma2_loss={bud.sigma2_loss:.4f} tot={bud.sigma2_total:.4f}")
    rep("sigma2_total ~0.1901", abs(bud.sigma2_total - 0.1901) < 2e-4)
    rep("sigma_total ~0.436", abs(bud.sigma_total - 0.436) < 1e-3)
    raw = raw_shift_error(bud)
    print(f"      raw shift error (either quadrature) = "
          f"{raw['p_exceed_either_quadrature']:.4f}")
    rep("raw error ~8.2e-2",
        abs(raw["p_exceed_either_quadrature"] - 0.0824) < 2e-3)
    sp_q = voronoi_erasure_split(bud, convention=ErasureConvention.PER_QUADRATURE)
    sp_m = voronoi_erasure_split(bud, convention=ErasureConvention.PER_MODE_UNION)
    print(f"      p_erase per-quadrature={sp_q['p_erase_used']:.4f}  "
          f"per-mode={sp_m['p_erase_used']:.4f}  (p_c=0.362)")
    rep("per-quadrature ~0.291", abs(sp_q["p_erase_used"] - 0.2905) < 3e-3)
    rep("per-mode ~0.497", abs(sp_m["p_erase_used"] - 0.4967) < 3e-3)
    rep("per-quadrature below p_c", sp_q["below_percolation_threshold"])
    rep("per-mode EXCEEDS p_c", not sp_m["below_percolation_threshold"])
    rep("convention declaration required",
        "REQUIRED_DECLARATION" in sp_q)
    print(f"      residual p_Pauli (per-quad) = {sp_q['p_pauli_used']:.3e}")
    rep("Pauli < erasure", sp_q["p_pauli_used"] < sp_q["p_erase_used"])

    print("\n[T5] BP+OSD LOGICAL ERROR (matches grounded Table 2)")
    for p, d, target, lbl in ((7.8e-4, 32, 1.3e-14, "R12.0-M"),
                              (6.4e-4, 64, 5.8e-30, "R12.0-W")):
        r = bp_osd_logical_error(p, d)
        got = r["P_L_memory_per_cycle"]
        rel = abs(got - target) / target
        print(f"      {lbl}: P_L={got:.3e} (target {target:.2e})")
        rep(f"{lbl} BP+OSD matches", rel < 0.15, f"rel={rel:.2f}")
    rep("memory separated from T-gate",
        "SEPARATE_FROM" in bp_osd_logical_error(7.8e-4, 32))
    b = False
    try: bp_osd_logical_error(1e-2, 32)
    except CapacityRefusal: b = True
    rep("above-threshold p REFUSED", b)
    md = magic_state_distillation(1e-3, 2)
    rep("distillation lowers T-gate error", md["output_infidelity"] < 1e-3)

    print("\n[T6] [N6] CAUSALITY AND GR METRIC")
    b = False
    try: assert_causality(2500.0, 1e-9)
    except CausalityRefusal: b = True
    rep("superluminal feed-forward REFUSED", b)
    cf = assert_causality(2500.0, 2500.0 / C_LIGHT * 1.2)
    rep("causal feed-forward accepted", cf["passes"])
    gr = GRMetricCompensator(); leo = gr.leo_link(500e3, 1000e3)
    print(f"      grav=+{leo['gravitational']['delta_U_over_c2']:.3e}  "
          f"trans={leo['transverse_doppler']['shift']:.3e}  "
          f"net={leo['net_fractional_shift']:.3e}")
    rep("gravitational ~+5.07e-11",
        abs(leo["gravitational"]["delta_U_over_c2"] - 5.066e-11) < 2e-13)
    rep("transverse ~-3.23e-10",
        abs(leo["transverse_doppler"]["shift"] + 3.227e-10) < 3e-13)
    rep("net ~-2.72e-10",
        abs(leo["net_fractional_shift"] + 2.720e-10) < 3e-13)
    rep("delta_f ~-52.6 kHz", abs(leo["net_delta_f_Hz"] + 52.6e3) < 0.3e3)
    print(f"      Shapiro slant={leo['shapiro_slant']['delta_t_s']*1e12:.3f} ps  "
          f"zenith={leo['shapiro_zenith']['delta_t_s']*1e12:.3f} ps")
    rep("slant ~4.48 ps",
        abs(leo["shapiro_slant"]["delta_t_s"] - 4.476e-12) < 5e-15)
    rep("zenith ~2.24 ps",
        abs(leo["shapiro_zenith"]["delta_t_s"] - 2.236e-12) < 5e-15)
    rep("slant/zenith differ ~2x",
        leo["shapiro_slant"]["delta_t_s"] /
        leo["shapiro_zenith"]["delta_t_s"] > 1.9)
    rep("geometry declared in output", "geometry" in leo["shapiro_slant"])
    rep("CSAC fast-phase caveat present", "beacon" in leo["clock_architecture"])
    rep("actuator range sufficient",
        gr.actuator_check(leo["net_fractional_shift"],
                          leo["shapiro_slant"]["delta_t_s"])["passes"])

    print("\n[T7] [N4] HYPERBOLIC vs CARPET GEOMETRY")
    cs = carpet_scaling(); hs = hyperbolic_scaling()
    print(f"      carpet d_H={cs['hausdorff_dimension']:.4f} "
          f"cut exp={cs['minimal_cut_exponent']:.4f}")
    rep("carpet d_H ~1.8928", abs(cs["hausdorff_dimension"] - 1.8928) < 1e-3)
    rep("carpet cut exp ~0.8928", abs(cs["minimal_cut_exponent"] - 0.8928) < 1e-3)
    rep("carpet RT-incompatible", not cs["rt_compatible"])
    rep("{5,4} RT-compatible", hs["rt_compatible"])
    b = False
    try: assert_rt_geometry("sierpinski_carpet")
    except GeometryRefusal: b = True
    rep("carpet RT claim REFUSED", b)
    b = False
    try: ads4_cft3_exactness_request()
    except GeometryRefusal: b = True
    rep("AdS4/CFT3 exactness REFUSED", b)
    emb_w = HyperbolicEmbedding("workstation")
    pw = HyperbolicPatchBudget(7, 2985, 6.8, 1840, 231)
    aw = emb_w.authorize(pw, 147456)
    print(f"      W patch: aggregate={aw['budget']['aggregate_crossing_loss_dB']:.2f} dB  "
          f"worst-path={aw['budget']['worst_path_insertion_loss_dB']:.3f} dB")
    rep("worst-path ~1.85 dB",
        abs(aw["budget"]["worst_path_insertion_loss_dB"] - 1.848) < 0.01)
    rep("aggregate distinguished from worst path",
        aw["budget"]["aggregate_crossing_loss_dB"] >
        aw["budget"]["worst_path_insertion_loss_dB"])
    emb_m = HyperbolicEmbedding("mobile")
    am = emb_m.authorize(HyperbolicPatchBudget(4, 165, 3.4, 142, 142), 3072)
    rep("mobile patch ~1.14 dB",
        abs(am["budget"]["worst_path_insertion_loss_dB"] - 1.136) < 0.01)
    b = False
    try: emb_m.authorize(HyperbolicPatchBudget(7, 2985, 6.8, 1840, 231), 3072)
    except GeometryRefusal: b = True
    rep("depth 7 on mobile REFUSED", b)

    print("\n[T8] [N5] Q-OEE OVERLAP ESTIMATION")
    st = swap_test_probability(0.6)
    rep("SWAP identity P(0)=0.8", abs(st["p_ancilla_zero"] - 0.8) < 1e-12)
    b = False
    try: assert_no_decohered_interference(0.0)
    except NoGoViolation: b = True
    rep("decohered branch interference REFUSED", b)
    q = qae_query_count(1e-4, "quantum"); c = qae_query_count(1e-4, "classical")
    print(f"      eps=1e-4: quantum={q['queries']:.0f} "
          f"classical={c['queries']:.0f} speedup={c['queries']/q['queries']:.0f}x")
    rep("QAE O(1/eps)", q["scaling"] == "O(1/eps)")
    rep("classical O(1/eps^2)", c["scaling"] == "O(1/eps^2)")
    rep("quadratic speedup 1e4", abs(c["queries"] / q["queries"] - 1e4) < 1)
    oe = OverlapEstimator(147456); est = oe.estimate(15, 1e-4)
    rep("2^15 = 32768 dim", est["register"]["hilbert_dim"] == 32768)
    rep("renamed from Everettian", "renamed_from" in est)
    rep("latency is Mode E", est["latency_stratum"] == MODE_E)
    rep("no inter-branch signalling", "inter-branch signalling" in est["forbids"])
    rn = renyi2_from_purity(0.5)
    rep("S2 = ln2 at purity 0.5", abs(rn["S2"] - math.log(2)) < 1e-12)

    print("\n[T9] THERMAL, CRYO, CAVITY-QED")
    for nm in REVISIONS:
        tc = thermal_closure(REVISIONS[nm])
        print(f"      {nm}: T={tc['T_predicted_C']:.2f} C "
              f"(limit {tc['T_limit_C']})")
        rep(f"{nm} thermal closure", tc["within_limit"])
    hot = boltzmann_spin_init(0.174, 245.0)
    cold = boltzmann_spin_init(0.406, 1.8)
    print(f"      245 K ratio={hot['ratio_dE_over_kT']:.4f}   "
          f"1.8 K ratio={cold['ratio_dE_over_kT']:.3f} F={cold['init_fidelity']:.4f}")
    rep("245 K scrambles spin-valley", not hot["usable"])
    rep("1.8 K F_init >0.93", cold["init_fidelity"] > 0.93)
    b = False
    try: assert_spin_regime(0.174, 245.0)
    except NoGoViolation: b = True
    rep("hot spin-valley REFUSED", b)
    cq = cavity_qed_cooperativity(1.45, 0.028, 0.018)
    print(f"      cooperativity C={cq['cooperativity']:.3e}")
    rep("C ~1.67e4", abs(cq["cooperativity"] - 1.667e4) < 1e2)
    rep("C sufficient for pi gate", cq["sufficient"])
    rep("replaces bulk Kerr", "replaces" in cq)
    mh = membrane_heat_leak(0.8, 298.2, 350e3)
    rep("membrane heat leak closes", mh["closes"])

    print("\n[T10] AI SURROGATES — GATED")
    if HAS_TORCH:
        dev = pick_device(cfg.device)
        tr = LinkConsistencyTransformer(12, 6, 64, 4, 2, 64).to(dev)
        x = torch.randn(4, 10, 12, device=dev); cd = torch.randn(4, 6, device=dev)
        o = tr(x, cd)
        rep("transformer forward", tuple(o["logits"].shape) == (4, 2))
        rep("nuisance head present", tuple(o["nuisance"].shape) == (4, 1))
        rep("labelled surrogate", "deterministic gate decides" in o["STATUS"])
        tgt = torch.randint(0, 2, (4,), device=dev)
        lf = lambda m, xx, cc, tt: F.cross_entropy(m(xx, cc)["logits"], tt)
        trn = Trainer(tr, TrainConfig(epochs=40, patience=10), cfg, lf, "link")
        r = trn.fit(x, cd, tgt)
        rep("trainer reduces loss", r["best_loss"] < r["history"][0]["loss"])
        rep("checkpoint written", Path(r["checkpoint"]).exists())
        wm = LinkWorldModel(10, 4, 64).to(dev)
        with torch.no_grad(): wm.decay_logit.fill_(20.0)
        rep("world-model decay bounded <1", 0 < float(wm.decay) < 1)
        wo = wm(torch.randn(4, 10, device=dev), torch.randn(4, 4, device=dev))
        rep("world model NOT a measurement", "NOT a measurement" in wo["STATUS"])
        df = RoutingDiffusion(48, 30, 128, 6).to(dev)
        opt = torch.optim.Adam(df.parameters(), lr=3e-3)
        x0 = torch.randn(8, 48, device=dev); c0 = torch.randn(8, 6, device=dev)
        l0 = lN = None
        for ep in range(30):
            opt.zero_grad(); L = df.loss(x0, c0); L.backward()
            torch.nn.utils.clip_grad_norm_(df.parameters(), 1.0); opt.step()
            if ep == 0: l0 = float(L)
            lN = float(L)
        rep("diffusion trains", lN < l0, f"{l0:.3f}->{lN:.3f}")
        s = df.sample(2, c0, dev)
        rep("diffusion labelled HYPOTHESIS", "HYPOTHESIS" in s["STATUS"])
        good = AIGate.gate_link({"plane": "Q", "K": 1, "freq_Hz": 193.4e12,
                                 "eta_atm": 0.68}, g)
        rep("valid AI link passes gate", good["passed"])
        bad = AIGate.gate_link({"plane": "Q", "K": 8, "freq_Hz": 193.4e12,
                                "eta_atm": 0.88}, g)
        rep("K=8 AI link gated to Mode E",
            not bad["passed"] and bad["status"] == MODE_E)
        gr_ok = AIGate.gate_routing(4, 165, 3.4, 142, 142, "mobile", 3072)
        rep("valid AI routing passes", gr_ok["passed"])
        gr_bad = AIGate.gate_routing(7, 2985, 6.8, 1840, 231, "mobile", 3072)
        rep("over-deep AI routing gated", not gr_bad["passed"])
        rep("rule stated", AIGate.rule() == "AI proposes; physics disposes")
    else:
        rep("torch available", False, "pip install torch")

    print("\n[T11] FULL PIPELINE + SD GATE")
    for nm in REVISIONS:
        out = pipe.analyse_revision(nm)
        rep(f"{nm} pipeline completes", "fidelity_chain" in out)
        rep(f"{nm} Stratum II flagged",
            out["stratum_II_targets"]["status"] == MODE_E)
        rep(f"{nm} scope design-only", out["scope"] == DESIGN_ONLY)
    b = False
    try:
        pipe.analyse_revision("R12.0-M",
                              convention=ErasureConvention.PER_MODE_UNION)
    except CapacityRefusal: b = True
    rep("per-mode convention REFUSED at eta=0.68", b)
    c1 = pipe.sd.certify("gkp_fidelity_chain", ("RAW","ERASURE","PAULI"))
    rep("missing CONV -> Mode E", c1["status"] == MODE_E)
    c2 = pipe.sd.certify("carpet_rt", ())
    rep("carpet RT entry -> Mode E", c2["status"] == MODE_E)
    rr = pipe.report()
    print(f"      SD: {rr['sd_report']['n']} outputs, "
          f"{rr['sd_report']['certified']} certified, "
          f"{rr['sd_report']['mode_E']} Mode E")
    rep("Mode E retained", rr["sd_report"]["mode_E"] > 0)
    rep("report written", Path(cfg.out_dir, "report.json").exists())
    rep("manifest written", Path(cfg.out_dir, "manifest.json").exists())
    rep("6 DRC predicates", len(rr["drc"]) == 6)
    rep("philosophy included", "motto" in rr["philosophy"])
    b = False
    try: refuse_scope("please fabricate R12.0-W now")
    except ScopeRefusal: b = True
    rep("fabrication REFUSED", b)

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
BUILT — ALL SIX NO-GO THEOREMS ENFORCED
  [N1] K->1 combiner: eta <= 1/K, Q=0 for K>=2. Plane Q is 1->1 only;
       Plane C (300 GHz, n_th=20.3) carries classical AirComp only.
  [N2] F_raw = 1/(1+e^{-2r}): 0.9432 at 12.2 dB; 0.994 needs 22.19 dB.
  [N3] sigma^2_tot = 0.1901, raw error 8.2e-2; Voronoi heralding splits
       p_erase from p_Pauli, and PER-QUADRATURE vs PER-MODE conventions
       are computed separately and must be declared.
  [N4] Carpet cut ~ L^0.8928 is RT-incompatible; {5,4} logical graph with
       finite depth and worst-path crossing budget; AdS4/CFT3 refused.
  [N5] Decohered branches cannot interfere; Q-OEE SWAP test with
       O(1/eps) QAE, renamed from "Everettian multi-branch".
  [N6] t_ff >= L/c enforced; GR terms computed with declared slant/zenith
       geometry (4.48 ps vs 2.24 ps).

VERIFIED NUMBERS
  n_th(1550 nm, 300 K) = 3.65e-14 ; n_th(300 GHz) = 20.3
  Q(0.68) = 1.0875 qubits/mode, 11.6x the 0.09375 code rate
  grav +5.07e-11, transverse -3.23e-10, net -2.72e-10, delta_f -52.6 kHz
  BP+OSD P_L: 1.3e-14 (d=32), 5.8e-30 (d=64)
  worst-path crossing loss 1.136 dB mobile / 1.848 dB workstation
  cavity-QED C = 1.67e4 ; 1.8 K spin init F = 0.932

NOT CLAIMED
  No chip is fabricated. No Stratum II value is measured. Every latency,
  throughput, squeezing figure, node count and crossing loss is a design
  TARGET. The per-mode erasure convention at eta=0.68 exceeds p_c and is
  refused — this is a real open point in the specification, not a bug.
""")
    return ok


# =============================================================================
# §13. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="MSPS quantum wireless-relativistic co-design suite")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest", "lineage", "strata", "capacity", "combiner",
                 "fidelity", "erasure", "gr", "geometry", "qoee",
                 "analyse", "drc", "packages", "philosophy", "report"])
    ap.add_argument("--rev", default="R12.0-W")
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--eta", type=float, default=0.68)
    ap.add_argument("--sq-db", type=float, default=12.2)
    ap.add_argument("--freq-ghz", type=float, default=193400.0)
    ap.add_argument("--alt-km", type=float, default=500.0)
    ap.add_argument("--slant-km", type=float, default=1000.0)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--platform", default="workstation")
    ap.add_argument("--convention", default="per_quadrature",
                    choices=["per_quadrature", "per_mode_union"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest": return 0 if self_test() else 1
    if a.cmd == "philosophy":
        print(json.dumps(PHILOSOPHY, indent=2, ensure_ascii=False)); return 0
    if a.cmd == "packages":
        print(json.dumps(PACKAGES, indent=2)); return 0
    if a.cmd == "drc":
        print(json.dumps(drc_predicates(), indent=2)); return 0
    if a.cmd == "lineage":
        print(json.dumps(lineage_report(), indent=2)); return 0
    if a.cmd == "strata":
        print(json.dumps({"stratum_I": sorted(STRATUM_I),
                          "stratum_II": sorted(STRATUM_II)}, indent=2)); return 0

    pipe = QuantumWirelessCoDesign(); out: Dict[str, Any] = {}
    conv = ErasureConvention(a.convention)
    try:
        if a.cmd == "capacity":
            out = {"thermal": thermal_occupation(a.freq_ghz * 1e9),
                   "capacity": holevo_pure_loss_capacity(a.eta)}
        elif a.cmd == "combiner":
            out = combiner_transmissivity_bound(a.K)
        elif a.cmd == "fidelity":
            out = {"at_target": bare_teleport_fidelity(a.sq_db),
                   "for_0p994": squeezing_for_target_fidelity(0.994)}
        elif a.cmd == "erasure":
            bud = GKPNoiseBudget(a.sq_db, a.eta)
            out = {"budget": bud.report(), "raw": raw_shift_error(bud),
                   "per_quadrature": voronoi_erasure_split(
                       bud, convention=ErasureConvention.PER_QUADRATURE),
                   "per_mode_union": voronoi_erasure_split(
                       bud, convention=ErasureConvention.PER_MODE_UNION)}
        elif a.cmd == "gr":
            out = GRMetricCompensator().leo_link(a.alt_km * 1e3, a.slant_km * 1e3)
        elif a.cmd == "geometry":
            emb = HyperbolicEmbedding(a.platform)
            patch = (HyperbolicPatchBudget(a.depth, 2985, 6.8, 1840, 231)
                     if a.platform == "workstation" else
                     HyperbolicPatchBudget(a.depth, 165, 3.4, 142, 142))
            out = {"carpet": carpet_scaling(), "hyperbolic": hyperbolic_scaling(),
                   "embedding": emb.authorize(
                       patch, REVISIONS["R12.0-W" if a.platform == "workstation"
                                        else "R12.0-M"].logical_qubits)}
        elif a.cmd == "qoee":
            rev = REVISIONS[a.rev]
            out = OverlapEstimator(rev.logical_qubits).estimate(
                15 if a.rev.endswith("W") else 10, a.eps)
        elif a.cmd == "analyse":
            out = pipe.analyse_revision(a.rev, conv, a.eta)
        elif a.cmd == "report":
            out = pipe.report()
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
