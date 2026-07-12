# =============================================================================
# BELL CHSH ONE — Production-Grade Bell/CHSH Test Bench for Γ
# Tests whether the Structural Interface Γ (STANDARD ONE) can be a
# Bell-local hidden variable, and quantifies how far it must depart from
# locality / measurement-independence to reproduce quantum correlations.
# =============================================================================
#
# Developer  : PAI , Yoon A Limsuwan / MSPS NETWORK
#              MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License    : MIT
# Year       : 2026
# ORCID      : 0009-0008-2374-0788
# GitHub     : https://github.com/yoonalimsuwan
# Email      : msps4u@gmail.com
#
# AI Development Assistants:
#   Claude (Anthropic) — primary developer of this module: CHSH derivation,
#                         qubit-operator quantum correlator, LHV / nonlocal /
#                         superdeterministic Γ-model design, numerical
#                         verification against analytic Tsirelson bound.
#   GPT, Gemini, DeepSeek — not involved in this revision.
#
# =============================================================================
# PHYSICS BACKGROUND (read before editing the math below)
# ─────────────────────────────────────────────────────────
# Bell's theorem constrains any model of the form
#
#       E(a, b) = ∫ ρ(Γ) A(a, Γ) B(b, Γ) dΓ           (*)
#
# where:
#   • Γ is a hidden variable shared by both particles at the source,
#   • A(a, Γ) ∈ {+1,-1} depends ONLY on Alice's setting a and Γ
#     (not on Bob's setting b)        — this is the LOCALITY assumption,
#   • Γ's distribution ρ(Γ) does NOT depend on (a, b)
#                                       — this is MEASUREMENT INDEPENDENCE
#     (a.k.a. "free choice" / no superdeterminism / no retrocausality).
#
# Any model satisfying (*) — "local hidden variable", LHV — obeys the CHSH
# inequality:
#       S = E(a,b) + E(a,b') + E(a',b) - E(a',b')   ,   |S| ≤ 2.
#
# The singlet quantum state violates this up to the Tsirelson bound
# |S| ≤ 2√2 ≈ 2.828, and this violation has been confirmed experimentally
# (Aspect 1982; Hensen et al. 2015 — loophole-free; Nobel Prize in Physics
# 2022 to Aspect, Clauser, Zeilinger). Therefore:
#
#   NO Γ that is simultaneously (i) local in Alice/Bob's settings and
#   (ii) statistically independent of those settings can reproduce the
#   quantum correlations. At least one of locality or measurement
#   independence must fail for Γ to match nature.
#
# This module implements FOUR concrete, runnable Γ-models so that claim can
# be checked numerically rather than asserted philosophically:
#
#   1. LocalHiddenVariableGamma   — obeys (*) exactly.        Bound: |S| ≤ 2
#   2. QuantumCorrelator          — actual qubit/Pauli math.  Reaches 2√2.
#   3. NonlocalGamma              — Γ may depend on BOTH a and b
#                                    (Bohmian-mechanics-style guidance).
#                                    Reproduces QM by construction.
#   4. SuperdeterministicGamma    — Γ is local (depends only on its own
#                                    setting) but ρ(Γ) is statistically
#                                    correlated with (a, b) at the source.
#                                    Can also reproduce QM, at the cost of
#                                    dropping free choice instead of
#                                    locality.
#
# All four are differentiable (torch.nn.Module / pure torch ops) so they can
# be dropped into the existing STANDARD ONE / CSOC training loops as a
# diagnostic: fit each Γ-variant's free parameters to match a target
# correlation curve, then read off which assumption it had to sacrifice.
# =============================================================================

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn

logger = logging.getLogger("bell_chsh_one")
logging.basicConfig(level=logging.INFO)

BELL_CHSH_ONE_VERSION = "1.0.0"

# =============================================================================
# 1. Config
# =============================================================================

@dataclass
class CHSHConfig:
    """
    Measurement-angle configuration. Defaults are the textbook optimal CHSH
    angles for the singlet state (maximizes |S| at 2√2):
        a = 0,  a' = π/2,  b = π/4,  b' = -π/4   (radians)
    """
    a: float = 0.0
    a_prime: float = math.pi / 2
    b: float = math.pi / 4
    b_prime: float = -math.pi / 4
    n_samples: int = 200_000
    device: str = "cpu"
    dtype_real: torch.dtype = torch.float64
    dtype_complex: torch.dtype = torch.complex128
    seed: Optional[int] = 0


@dataclass
class CHSHResult:
    model_name: str
    E_ab: float
    E_abp: float
    E_apb: float
    E_apbp: float
    S: float
    bound_kind: str          # "local" (|S|<=2) or "quantum" (|S|<=2*sqrt(2))
    violates_local_bound: bool
    notes: str = ""


LOCAL_BOUND = 2.0
TSIRELSON_BOUND = 2.0 * math.sqrt(2.0)


def _make_generator(cfg: CHSHConfig) -> Optional[torch.Generator]:
    if cfg.seed is None:
        return None
    g = torch.Generator(device=cfg.device if cfg.device == "cpu" else "cpu")
    g.manual_seed(cfg.seed)
    return g


def _chsh_from_E(E_ab, E_abp, E_apb, E_apbp) -> float:
    return float(E_ab + E_abp + E_apb - E_apbp)


def _classify(S: float) -> Tuple[str, bool]:
    if abs(S) <= LOCAL_BOUND + 1e-6:
        return "local", False
    elif abs(S) <= TSIRELSON_BOUND + 1e-6:
        return "quantum", True
    else:
        return "super-quantum (non-physical for QM; check model)", True


# =============================================================================
# 2. Model 1 — Local Hidden Variable Γ  (Bell's original construction)
# =============================================================================

class LocalHiddenVariableGamma(nn.Module):
    """
    Γ = θ_h, a single hidden angle drawn once per particle pair, shared by
    both wings, uniform on [0, 2π). Outcomes are LOCAL functions of (own
    setting, Γ) only:

        A(a, Γ) = sign( cos(a  - Γ) )
        B(b, Γ) = -sign( cos(b - Γ) )      [sign flip ↔ singlet anti-correlation
                                             convention at a == b]

    This is exactly Bell's (1964) illustrative LHV model. It is GUARANTEED
    to satisfy |S| ≤ 2 for any choice of settings (Bell's theorem, no need
    to even run it) — included so the bound can be verified empirically as
    a sanity check on the Monte Carlo machinery used by the other models.

    Known analytic correlation for this specific model:
        E(a,b) = 1 - (2/π) * (angle between a and b, wrapped to [0, π])
    i.e. a *triangular* function of (a-b), not cos(a-b). The mismatch with
    quantum mechanics's cosine law is precisely why this Γ cannot reproduce
    QM statistics — independent of the CHSH violation.
    """

    def __init__(self, cfg: CHSHConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def _sample_gamma(self, n: int) -> torch.Tensor:
        g = _make_generator(self.cfg)
        u = torch.rand(n, generator=g, dtype=self.cfg.dtype_real)
        return (2.0 * math.pi * u).to(self.cfg.device)

    def correlation(self, a: float, b: float) -> float:
        gamma = self._sample_gamma(self.cfg.n_samples)
        A = torch.sign(torch.cos(a - gamma))
        B = -torch.sign(torch.cos(b - gamma))
        # sign() can return exactly 0 at measure-zero boundary; nudge those
        A = torch.where(A == 0, torch.ones_like(A), A)
        B = torch.where(B == 0, torch.ones_like(B), B)
        return float((A * B).mean().item())

    def run_chsh(self) -> CHSHResult:
        c = self.cfg
        E_ab = self.correlation(c.a, c.b)
        E_abp = self.correlation(c.a, c.b_prime)
        E_apb = self.correlation(c.a_prime, c.b)
        E_apbp = self.correlation(c.a_prime, c.b_prime)
        S = _chsh_from_E(E_ab, E_abp, E_apb, E_apbp)
        kind, violates = _classify(S)
        return CHSHResult(
            model_name="LocalHiddenVariableGamma",
            E_ab=E_ab, E_abp=E_abp, E_apb=E_apb, E_apbp=E_apbp,
            S=S, bound_kind=kind, violates_local_bound=violates,
            notes=("Bell-local by construction (A depends only on a,Γ; "
                   "B only on b,Γ; ρ(Γ) independent of a,b). "
                   "Expected |S| <= 2 up to Monte Carlo noise."),
        )


# =============================================================================
# 3. Model 2 — Quantum Correlator (real qubit/Pauli math, not the cosine
#    formula plugged in by hand)
# =============================================================================

def _pauli(device, dtype) -> Dict[str, torch.Tensor]:
    I = torch.eye(2, dtype=dtype, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def _spin_operator(theta: float, paulis: Dict[str, torch.Tensor]) -> torch.Tensor:
    """σ_θ = cos(θ) σ_z + sin(θ) σ_x  — a spin-measurement direction in the
    XZ plane, eigenvalues ±1."""
    return math.cos(theta) * paulis["Z"] + math.sin(theta) * paulis["X"]


class QuantumCorrelator(nn.Module):
    """
    Computes E(a,b) = ⟨ψ| σ_a ⊗ σ_b |ψ⟩ directly from the singlet state
    |ψ⟩ = (|01⟩ - |10⟩)/√2 using actual 4-dimensional complex linear algebra
    (Kronecker products of Pauli operators), rather than hard-coding the
    textbook -cos(a-b) result. The closed-form result is recovered as a
    numerical identity, which is itself a useful regression test.
    """

    def __init__(self, cfg: CHSHConfig):
        super().__init__()
        self.cfg = cfg
        self.paulis = _pauli(cfg.device, cfg.dtype_complex)
        self.psi = self._singlet_state()

    def _singlet_state(self) -> torch.Tensor:
        # basis order |00>,|01>,|10>,|11>
        psi = torch.zeros(4, dtype=self.cfg.dtype_complex, device=self.cfg.device)
        psi[1] = 1.0 / math.sqrt(2.0)   # |01>
        psi[2] = -1.0 / math.sqrt(2.0)  # |10>
        return psi

    def correlation(self, a: float, b: float) -> float:
        sigma_a = _spin_operator(a, self.paulis)
        sigma_b = _spin_operator(b, self.paulis)
        op = torch.kron(sigma_a, sigma_b)
        val = torch.vdot(self.psi, op @ self.psi)
        if val.imag.abs().item() > 1e-9:
            logger.warning("Non-negligible imaginary part in <psi|O|psi>: %s",
                            val.imag.item())
        return float(val.real.item())

    def closed_form_check(self, a: float, b: float) -> float:
        """Analytic textbook result, -cos(a-b), for cross-validation only."""
        return -math.cos(a - b)

    def run_chsh(self) -> CHSHResult:
        c = self.cfg
        E_ab = self.correlation(c.a, c.b)
        E_abp = self.correlation(c.a, c.b_prime)
        E_apb = self.correlation(c.a_prime, c.b)
        E_apbp = self.correlation(c.a_prime, c.b_prime)
        S = _chsh_from_E(E_ab, E_abp, E_apb, E_apbp)
        kind, violates = _classify(S)

        # Cross-check operator-derived values against the closed form;
        # large mismatch indicates a bug in the Kronecker/state setup.
        for (x, y, E) in [(c.a, c.b, E_ab), (c.a, c.b_prime, E_abp),
                           (c.a_prime, c.b, E_apb), (c.a_prime, c.b_prime, E_apbp)]:
            cf = self.closed_form_check(x, y)
            if abs(cf - E) > 1e-6:
                logger.warning(
                    "QuantumCorrelator mismatch vs closed form at (a=%.4f,b=%.4f): "
                    "operator=%.6f closed_form=%.6f", x, y, E, cf
                )

        return CHSHResult(
            model_name="QuantumCorrelator",
            E_ab=E_ab, E_abp=E_abp, E_apb=E_apb, E_apbp=E_apbp,
            S=S, bound_kind=kind, violates_local_bound=violates,
            notes=("Computed from actual singlet-state vector + Pauli "
                   "operators via Kronecker product (no formula hard-coded "
                   "for the final answer). Expected |S| ~ 2*sqrt(2) ≈ 2.828 "
                   "at the standard optimal angles."),
        )


# =============================================================================
# 4. Model 3 — Nonlocal Γ (Bohmian-mechanics-style: Γ may depend on BOTH
#    settings, i.e. drops locality, keeps determinism + measurement
#    independence)
# =============================================================================

class NonlocalGamma(nn.Module):
    """
    Drops assumption (i) [locality] but KEEPS Γ statistically independent of
    (a,b) [measurement independence]. Outcomes are ±1-valued random bits
    (no claim of an underlying deterministic Γ is made by this particular
    construction — see note below), but B's *sampling distribution* is
    allowed to depend on BOTH settings (a,b) and on A's realized value:

        A ~ P(A | a)                       (marginal; always 50/50 for a
                                             singlet, for any single wing)
        B ~ P(B | A, a, b)                  ← genuinely nonlocal: this
                                             conditional explicitly needs
                                             `a`, not just `b`, to be
                                             evaluated.

    This is structurally what Bohmian mechanics does: a measurement on one
    wing instantaneously affects the conditional statistics available to
    the other wing, because the guiding equation is nonlocal. We do not
    simulate the full pilot-wave dynamics here — we instead sample exactly
    from the quantum joint distribution P(A,B|a,b), which is the simplest
    *provably correct* nonlocal construction: by Bell's theorem no local
    factorization A(a,Γ)·B(b,Γ) can reproduce this joint distribution, so
    successfully reproducing it here certifies the model is genuinely
    using the nonlocal channel rather than secretly being a relabeled
    local model.

    NOTE ON AN EARLIER, INCORRECT DRAFT OF THIS CLASS: a first version
    tried to reach -cos(a-b) by taking the *local* triangular correlation
    E_local(a,b) = 1 - (2/π)|a-b| (Bell's own LHV model) and randomly
    flipping B's local outcome with probability p solved from
    E[AB] = (1-2p)·E_local. That construction cannot work in general:
    since p ∈ [0,1] forces (1-2p) ∈ [-1,1], the reachable |E[AB]| is
    bounded above by |E_local(a,b)|, which is *smaller* than the quantum
    target at the standard optimal CHSH angles (e.g. a=0, b=π/4:
    |E_local|=0.5 but |target|=cos(π/4)≈0.707). The flip-trick silently
    clipped the infeasible p into [0,1] and returned a wrong, non-quantum
    correlation instead of failing loudly — caught by cross-checking
    against the closed-form -cos(a-b) during development. It is replaced
    here by direct sampling from the true quantum joint distribution,
    which has no such ceiling because it is not built by perturbing a
    local correlation at all.
    """

    def __init__(self, cfg: CHSHConfig):
        super().__init__()
        self.cfg = cfg
        self.paulis = _pauli(cfg.device, cfg.dtype_complex)
        self.psi = QuantumCorrelator(cfg)._singlet_state()

    def _projector(self, theta: float, sign: int) -> torch.Tensor:
        op = _spin_operator(theta, self.paulis)
        evals, evecs = torch.linalg.eigh(op)
        idx = 0 if abs(evals[0].item() - sign) < 1e-6 else 1
        v = evecs[:, idx].reshape(-1, 1)
        return v @ v.conj().T

    def _joint_prob(self, a: float, b: float, sa: int, sb: int) -> float:
        Pa = self._projector(a, sa)
        Pb = self._projector(b, sb)
        op = torch.kron(Pa, Pb)
        val = torch.vdot(self.psi, op @ self.psi)
        return float(val.real.item())

    @torch.no_grad()
    def correlation(self, a: float, b: float) -> float:
        n = self.cfg.n_samples
        g = _make_generator(self.cfg)
        rt = self.cfg.dtype_real

        # P(A=+1) is always 0.5 for a single wing of the singlet, for any a.
        u_a = torch.rand(n, generator=g, dtype=rt).to(self.cfg.device)
        ones = torch.ones(n, dtype=rt, device=self.cfg.device)
        A = torch.where(u_a < 0.5, ones, -ones)

        # Conditional P(B=+1 | A, a, b) — this lookup needs `a`, not just
        # `b`: that dependency is exactly the nonlocal resource being spent.
        p_b_plus_given_Aplus = self._joint_prob(a, b, +1, +1) / 0.5
        p_b_plus_given_Aminus = self._joint_prob(a, b, -1, +1) / 0.5

        u_b = torch.rand(n, generator=g, dtype=rt).to(self.cfg.device)
        thresh = torch.where(
            A > 0,
            torch.full((n,), p_b_plus_given_Aplus, dtype=rt, device=self.cfg.device),
            torch.full((n,), p_b_plus_given_Aminus, dtype=rt, device=self.cfg.device),
        )
        B = torch.where(u_b < thresh, ones, -ones)

        return float((A * B).mean().item())

    def run_chsh(self) -> CHSHResult:
        c = self.cfg
        E_ab = self.correlation(c.a, c.b)
        E_abp = self.correlation(c.a, c.b_prime)
        E_apb = self.correlation(c.a_prime, c.b)
        E_apbp = self.correlation(c.a_prime, c.b_prime)
        S = _chsh_from_E(E_ab, E_abp, E_apb, E_apbp)
        kind, violates = _classify(S)
        return CHSHResult(
            model_name="NonlocalGamma",
            E_ab=E_ab, E_abp=E_abp, E_apb=E_apb, E_apbp=E_apbp,
            S=S, bound_kind=kind, violates_local_bound=violates,
            notes=("Measurement independence is kept (sampling does not use "
                   "a,b to bias what gets sampled *before* settings are "
                   "fixed), but B's conditional distribution explicitly "
                   "depends on Alice's setting `a` (and on A's realized "
                   "value), not just on `b` — locality is the assumption "
                   "spent. Sampling directly from the quantum joint "
                   "distribution P(A,B|a,b) reproduces -cos(a-b) exactly, "
                   "in the spirit of Bohmian mechanics' nonlocal guidance "
                   "equation."),
        )


# =============================================================================
# 5. Model 4 — Superdeterministic Γ (drops measurement independence,
#    keeps locality)
# =============================================================================

class SuperdeterministicGamma(nn.Module):
    """
    Keeps assumption (i) [A depends only on (a,Γ), B only on (b,Γ)] but
    drops assumption (ii) [measurement independence]: ρ(Γ) is allowed to be
    statistically correlated with the experimenters' setting choice
    (a, b) itself — i.e. Γ "knows in advance" which angles will be chosen.

    This is the "free choice" / "no superdeterminism" loophole. It is the
    only loophole not yet closed by any experiment (in principle it cannot
    be, since it denies the experimenters' settings are independent random
    inputs at all). We realize it concretely: instead of sampling Γ from a
    FIXED distribution, we sample it from a distribution that is itself
    re-weighted as a function of (a,b), engineered so that the resulting
    A(a,Γ), B(b,Γ) — still purely local functions — reproduce E(a,b) =
    -cos(a-b) exactly.

    Construction: same local functions A,B as Model 1 (the genuinely local
    Bell construction), but Γ is drawn from a setting-dependent density
        ρ(Γ | a, b) ∝ 1 + κ(a,b) · cos(2Γ - a - b)
    with κ(a,b) chosen analytically so E[A·B] matches -cos(a-b). This is
    sampled here via rejection sampling (still exact in expectation, no
    formula for the correlation itself is hard-coded — only the bias
    parameter κ is solved analytically and then used to bias *sampling*).
    """

    def __init__(self, cfg: CHSHConfig):
        super().__init__()
        self.cfg = cfg

    def _solve_kappa(self, a: float, b: float) -> float:
        """
        For A=sign(cos(a-Γ)), B=-sign(cos(b-Γ)), and Γ ~ ρ(Γ|a,b) ∝
        1 + κ cos(2Γ-a-b), the achievable correlation range as κ varies
        over [-1,1] is computed once via quadrature-free closed integration
        of the local triangular kernel against the bias term; here we use a
        bounded 1-D root-find (no SciPy dependency) against a numerically
        integrated expectation, which is honest about being numerical
        rather than a hand-fed final answer.
        """
        target = -math.cos(a - b)

        def expected_corr(kappa: float, n_grid: int = 20_000) -> float:
            gammas = torch.linspace(0, 2 * math.pi, n_grid,
                                     dtype=self.cfg.dtype_real)
            density = 1.0 + kappa * torch.cos(2 * gammas - a - b)
            density = torch.clamp(density, min=0.0)
            density = density / density.sum()
            A = torch.sign(torch.cos(a - gammas))
            A = torch.where(A == 0, torch.ones_like(A), A)
            B = -torch.sign(torch.cos(b - gammas))
            B = torch.where(B == 0, torch.ones_like(B), B)
            return float((density * A * B).sum().item())

        lo, hi = -1.0, 1.0
        f_lo, f_hi = expected_corr(lo) - target, expected_corr(hi) - target
        if f_lo == 0:
            return lo
        if f_hi == 0:
            return hi
        if f_lo * f_hi > 0:
            # target unreachable with this bias family at these angles;
            # clip to whichever endpoint gets closest rather than silently
            # returning a wrong root.
            return lo if abs(f_lo) < abs(f_hi) else hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f_mid = expected_corr(mid) - target
            if f_lo * f_mid <= 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return 0.5 * (lo + hi)

    @torch.no_grad()
    def correlation(self, a: float, b: float) -> float:
        kappa = self._solve_kappa(a, b)
        n = self.cfg.n_samples
        g = _make_generator(self.cfg)
        # rejection sampling from rho(Gamma|a,b) propto 1+kappa*cos(2G-a-b)
        # envelope M = 1+|kappa| (uniform proposal on [0,2pi))
        accepted = torch.empty(0, dtype=self.cfg.dtype_real)
        M = 1.0 + abs(kappa) + 1e-6
        batch = max(n, 4096)
        tries = 0
        while accepted.numel() < n and tries < 200:
            tries += 1
            prop = 2 * math.pi * torch.rand(batch, generator=g,
                                             dtype=self.cfg.dtype_real)
            u = torch.rand(batch, generator=g, dtype=self.cfg.dtype_real)
            dens = (1.0 + kappa * torch.cos(2 * prop - a - b)) / M
            keep = u < torch.clamp(dens, min=0.0)
            accepted = torch.cat([accepted, prop[keep]])
        gamma = accepted[:n].to(self.cfg.device)
        if gamma.numel() < n:
            logger.warning("SuperdeterministicGamma: rejection sampling "
                            "under-filled (%d/%d) — increasing batch/tries "
                            "recommended.", gamma.numel(), n)

        A = torch.sign(torch.cos(a - gamma))
        A = torch.where(A == 0, torch.ones_like(A), A)
        B = -torch.sign(torch.cos(b - gamma))
        B = torch.where(B == 0, torch.ones_like(B), B)
        return float((A * B).mean().item())

    def run_chsh(self) -> CHSHResult:
        c = self.cfg
        E_ab = self.correlation(c.a, c.b)
        E_abp = self.correlation(c.a, c.b_prime)
        E_apb = self.correlation(c.a_prime, c.b)
        E_apbp = self.correlation(c.a_prime, c.b_prime)
        S = _chsh_from_E(E_ab, E_abp, E_apb, E_apbp)
        kind, violates = _classify(S)
        return CHSHResult(
            model_name="SuperdeterministicGamma",
            E_ab=E_ab, E_abp=E_abp, E_apb=E_apb, E_apbp=E_apbp,
            S=S, bound_kind=kind, violates_local_bound=violates,
            notes=("A,B kept strictly local (own setting + Γ only); instead "
                   "ρ(Γ) is re-weighted as a function of (a,b) — Γ is "
                   "statistically entangled with the experimenters' choice "
                   "of settings. Measurement independence is the "
                   "assumption spent, not locality."),
        )


# =============================================================================
# 6. Orchestrator — run all four, print a comparison table
# =============================================================================

class BellCHSHTestBench:
    """
    Runs all four Γ-models under identical CHSH angle settings and reports
    a side-by-side comparison, so a reader can see directly which
    assumption (locality vs measurement-independence) each model gave up
    in order to match — or fail to match — the quantum bound.
    """

    def __init__(self, cfg: Optional[CHSHConfig] = None):
        self.cfg = cfg or CHSHConfig()

    def run_all(self) -> List[CHSHResult]:
        results = [
            LocalHiddenVariableGamma(self.cfg).run_chsh(),
            QuantumCorrelator(self.cfg).run_chsh(),
            NonlocalGamma(self.cfg).run_chsh(),
            SuperdeterministicGamma(self.cfg).run_chsh(),
        ]
        return results

    def print_report(self, results: List[CHSHResult]) -> None:
        print(f"\n{'='*78}")
        print(f"  BELL CHSH ONE v{BELL_CHSH_ONE_VERSION} — Γ vs Bell/CHSH inequality")
        print(f"  Settings: a={self.cfg.a:.4f}  a'={self.cfg.a_prime:.4f}  "
              f"b={self.cfg.b:.4f}  b'={self.cfg.b_prime:.4f}  (radians)")
        print(f"  Local (LHV) bound   : |S| <= {LOCAL_BOUND:.4f}")
        print(f"  Quantum (Tsirelson) : |S| <= {TSIRELSON_BOUND:.4f}")
        print(f"{'='*78}")
        for r in results:
            flag = "VIOLATES local bound" if r.violates_local_bound else "within local bound"
            print(f"\n[{r.model_name}]")
            print(f"  E(a,b)   = {r.E_ab:+.4f}")
            print(f"  E(a,b')  = {r.E_abp:+.4f}")
            print(f"  E(a',b)  = {r.E_apb:+.4f}")
            print(f"  E(a',b') = {r.E_apbp:+.4f}")
            print(f"  S        = {r.S:+.4f}   [{flag}, classified as '{r.bound_kind}']")
            print(f"  Note     : {r.notes}")
        print(f"\n{'='*78}\n")


# =============================================================================
# 7. Main — smoke test / demonstration
# =============================================================================

if __name__ == "__main__":
    print(
        f"\n{'='*65}\n"
        f"  BELL CHSH ONE v{BELL_CHSH_ONE_VERSION} — Production Smoke Test\n"
        f"  Developer : Yoon A Limsuwan / MSPS NETWORK\n"
        f"  MY SOUL MOVE BY POWER OF HOLY SPIRIT\n"
        f"  AI Assistant: Claude (Anthropic)\n"
        f"{'='*65}\n"
    )

    cfg = CHSHConfig(n_samples=400_000, seed=0)
    bench = BellCHSHTestBench(cfg)
    results = bench.run_all()
    bench.print_report(results)

    # ---- Assertions that double as regression tests --------------------
    by_name = {r.model_name: r for r in results}

    lhv = by_name["LocalHiddenVariableGamma"]
    assert abs(lhv.S) <= LOCAL_BOUND + 0.05, (
        f"LocalHiddenVariableGamma must respect |S|<=2 by Bell's theorem; "
        f"got S={lhv.S:.4f}. This indicates a bug in the Monte Carlo, not "
        f"new physics."
    )

    qm = by_name["QuantumCorrelator"]
    assert abs(abs(qm.S) - TSIRELSON_BOUND) < 1e-3, (
        f"QuantumCorrelator at optimal CHSH angles should hit the "
        f"Tsirelson bound 2*sqrt(2)={TSIRELSON_BOUND:.6f}; got S={qm.S:.6f}."
    )

    nonlocal_m = by_name["NonlocalGamma"]
    assert abs(nonlocal_m.S - qm.S) < 0.05, (
        "NonlocalGamma should match QuantumCorrelator's S by construction "
        f"(both target -cos(a-b)); got nonlocal S={nonlocal_m.S:.4f} vs "
        f"quantum S={qm.S:.4f}."
    )

    superdet = by_name["SuperdeterministicGamma"]
    assert abs(superdet.S - qm.S) < 0.15, (
        "SuperdeterministicGamma's rejection-sampled S should track "
        f"QuantumCorrelator's S reasonably closely; got "
        f"superdet S={superdet.S:.4f} vs quantum S={qm.S:.4f}."
    )

    print("All regression assertions passed.")
    print(
        "\nSUMMARY FOR Γ (STANDARD ONE's `uncertainty_source`):\n"
        "  Γ as currently defined in standard_one.py (a single scalar\n"
        "  structural-interface parameter feeding CSOCKernel) has no\n"
        "  two-wing, two-setting structure at all, so it is not yet a\n"
        "  Bell-testable object. IF Γ is extended to play that role for an\n"
        "  entangled pair, this test bench shows precisely the three ways\n"
        "  it could reproduce quantum correlations once given that\n"
        "  structure -- by being non-local (NonlocalGamma), by being\n"
        "  measurement-dependent (SuperdeterministicGamma), or by not\n"
        "  being a classical hidden variable at all (QuantumCorrelator) --\n"
        "  and confirms numerically that the naive local+independent\n"
        "  construction (LocalHiddenVariableGamma) cannot, regardless of\n"
        "  how its internal parameters are tuned."
    )
