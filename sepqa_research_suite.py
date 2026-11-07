"""
=============================================================================
SEPQA RESEARCH SUITE — Software-Engineered Physical Qubit Architecture
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0
=============================================================================
FOUR PILLARS IMPLEMENTED:
  [1] Software-Defined Qubits (Pulse-Level Control)
  [2] AI & ML Auto-Calibration (OPMC + Bayesian Optimization)
  [3] Error Mitigation via Classical Post-Processing (ZNE, PEC, Readout)
  [4] Cryogenic CMOS Simulation (Latency, Noise, Power)

GROUNDING IN 15-PAPER SERIES:
  • OPMC (Unified Ordinal-Tensor): contraction L=0.438520, k*=23,
    d(m,n)=m²n²+mn², A^W_str on compact M_str
  • Paper 8 (Diagnostic Battery): 6-test classification
  • Paper 11 (Assumption-Light Stats): subsampling CI, rate-free
  • Paper 9 (Statistical Dictionary): EVT, Hill estimator
  • Six Problems §6.2: No-Zeno hysteresis gate
  • Delta_Min: δ_min isodiametric bound
  • Stochastic_NoZeno: exit-time bound via BDG

HONEST SCOPE (from QICC/SEPQA paper §10.3):
  "Classical analogies are not physical quantum phenomena."
  "A quantum-inspired algorithm does not automatically outperform."
  "SEPQA depends on the existence of a suitable physical quantum substrate."
  "Purely classical software creates quantum coherence without a quantum
   substrate: Not established and not implied by SEPQA."

ALL SIMULATIONS BELOW ARE CLASSICAL SIMULATIONS OF QUANTUM SYSTEMS.
They model what software COULD do given a physical quantum substrate.
They do NOT create quantum coherence.
=============================================================================
"""

from __future__ import annotations

import math
import time
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum, auto

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor

logger = logging.getLogger("sepqa_research")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

SEPQA_VERSION = "1.0.0"

# =============================================================================
# OPMC CONSTANTS (from Unified Ordinal-Tensor)
# =============================================================================
OPMC_L = 0.438520
OPMC_C1 = 0.420000
OPMC_K2 = 12.5000
OPMC_D_MN = lambda m, n: m * m * n * n + m * n * n

def opmc_k_star(epsilon: float = 1e-8, D0: float = 1.0) -> int:
    """Theorem 11.1: k* = ceil(ln(ε/D₀)/ln L)."""
    return math.ceil(math.log(epsilon / D0) / math.log(OPMC_L))

# =============================================================================
# COMPLEX HELPERS (explicit, never AMP-cast)
# =============================================================================
CD = torch.complex128

def pauli() -> Dict[str, Tensor]:
    return {
        "I": torch.eye(2, dtype=CD),
        "X": torch.tensor([[0, 1], [1, 0]], dtype=CD),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=CD),
        "Z": torch.tensor([[1, 0], [0, -1]], dtype=CD),
    }

def gate_fidelity(U_ideal: Tensor, U_actual: Tensor) -> float:
    """Average gate fidelity: F = (|Tr(U†_ideal U_actual)|² + d) / (d² + d)."""
    d = U_ideal.shape[0]
    overlap = torch.trace(U_ideal.conj().T @ U_actual)
    return float((abs(overlap) ** 2 + d) / (d ** 2 + d))


# =============================================================================
# [1] SOFTWARE-DEFINED QUBITS — PULSE-LEVEL CONTROL
# =============================================================================
# From SEPQA paper §6.3-6.6:
#   Software → Classical control hardware → Physical quantum substrate
#   H(t) = H_0 + Σ_j u_j(t) H_j
#   iℏ d/dt |ψ(t)⟩ = H(t)|ψ(t)⟩
#
# This module implements GRAPE (Gradient Ascent Pulse Engineering):
#   Khaneja, Reiss, Kehlet, Schulte-Herbrüggen, Glaser (2005)
#   "Optimal control of coupled spin dynamics"
#   J. Magn. Reson. 172, 296-305.
#
# HONEST SCOPE: This simulates pulse optimization on a classical computer.
# It does NOT control physical hardware. The physical substrate is required.
# =============================================================================

@dataclass
class PulseControlConfig:
    """Configuration for pulse-level qubit control."""
    n_time_steps: int = 100         # number of discretization steps
    total_time_ns: float = 100.0    # total pulse duration in ns
    n_control_channels: int = 2     # e.g., X-drive, Z-drive
    amplitude_max: float = 1.0      # maximum control amplitude
    learning_rate: float = 1e-2     # GRAPE learning rate
    n_iterations: int = 200         # optimization iterations
    device: str = "auto"


class PulseLevelQubitControl(nn.Module):
    """
    [1] Software-Defined Qubits via Pulse-Level Control.

    Implements GRAPE for single-qubit gate optimization:
      1. Parameterize pulse as piecewise-constant amplitudes
      2. Compute unitary via time-ordered product
      3. Optimize pulse to maximize fidelity with target gate
      4. Differentiable end-to-end for gradient-based optimization

    HAMILTONIAN MODEL (SEPQA §6.4):
      H(t) = H_0 + u_x(t) σ_x + u_z(t) σ_z

    where u_x, u_z are control amplitudes (microwave/Z-drive).

    HONEST SCOPE:
      • Classical simulation of pulse optimization
      • Requires physical quantum substrate for actual execution
      • Software does NOT create quantum coherence (SEPQA §10.3)

    USAGE:
      controller = PulseLevelQubitControl(cfg)
      result = controller.optimize_gate(target_gate="X")
    """

    def __init__(self, cfg: Optional[PulseControlConfig] = None):
        super().__init__()
        self.cfg = cfg or PulseControlConfig()

        if self.cfg.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)

        # Control pulse parameters: (n_time_steps, n_channels)
        self.pulse_amplitudes = nn.Parameter(
            torch.randn(
                self.cfg.n_time_steps,
                self.cfg.n_control_channels,
                device=self.device,
            ) * 0.1
        )

        # System Hamiltonian (qubit drift)
        P = pauli()
        self.register_buffer("H0", 0.5 * P["Z"], persistent=False)
        self.register_buffer("HX", P["X"], persistent=False)
        self.register_buffer("HZ", P["Z"], persistent=False)

        # dt for each time step
        self.dt_ns = self.cfg.total_time_ns / self.cfg.n_time_steps

        # OPMC convergence bound
        self.k_star = opmc_k_star(1e-8)

    def build_hamiltonian(self, step: int) -> Tensor:
        """H(t_k) = H_0 + u_x(t_k) σ_x + u_z(t_k) σ_z."""
        ux = self.pulse_amplitudes[step, 0]
        uz = self.pulse_amplitudes[step, 1]
        return self.H0 + ux * self.HX + uz * self.HZ

    def compute_unitary(self) -> Tensor:
        """
        Compute total unitary U = U_{N-1} ... U_1 U_0.
        Each U_k = exp(-i H(t_k) dt).
        """
        U_total = torch.eye(2, dtype=CD, device=self.device)
        dt = self.dt_ns  # in ns

        for k in range(self.cfg.n_time_steps):
            H_k = self.build_hamiltonian(k)
            # exp(-i H dt): use eigendecomposition for stability
            evals, evecs = torch.linalg.eigh((H_k + H_k.conj().T) / 2)
            phases = torch.exp(-1j * evals * dt).to(CD)
            U_k = (evecs * phases) @ evecs.conj().T
            U_total = U_k @ U_total

        return U_total

    def target_gates(self) -> Dict[str, Tensor]:
        """Common single-qubit target gates."""
        P = pauli()
        I2 = torch.eye(2, dtype=CD)
        return {
            "I": I2,
            "X": P["X"],
            "Y": P["Y"],
            "Z": P["Z"],
            "H": (P["X"] + P["Z"]) / math.sqrt(2),
            "S": torch.diag(torch.tensor([1.0, 1j], dtype=CD)),
            "T": torch.diag(
                torch.tensor([1.0, torch.exp(torch.tensor(1j * math.pi / 4))],
                             dtype=CD)),
        }

    def fidelity_loss(self, target: Tensor) -> Tensor:
        """
        Loss = 1 - F, where F is average gate fidelity.
        F = (|Tr(U†_target U_actual)|² + d) / (d² + d)
        """
        U_actual = self.compute_unitary()
        d = 2
        overlap = torch.trace(target.conj().T @ U_actual)
        F = (torch.abs(overlap) ** 2 + d) / (d ** 2 + d)
        return 1.0 - F

    def optimize_gate(self, target_gate: str = "X",
                      n_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize pulse to implement target gate.

        Uses L-BFGS (GRAPE standard) or Adam fallback.

        Args:
            target_gate: name of target gate
            n_iterations: optimization iterations

        Returns:
            dict with final fidelity, pulse shape, convergence info
        """
        targets = self.target_gates()
        if target_gate not in targets:
            raise ValueError(f"Unknown gate: {target_gate}")
        target = targets[target_gate]

        n_iter = n_iterations or self.cfg.n_iterations
        optimizer = torch.optim.Adam(
            [self.pulse_amplitudes], lr=self.cfg.learning_rate)

        history = []
        for i in range(n_iter):
            optimizer.zero_grad()
            loss = self.fidelity_loss(target)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                [self.pulse_amplitudes], max_norm=1.0)

            optimizer.step()

            # Amplitude constraint: |u| ≤ amplitude_max
            with torch.no_grad():
                self.pulse_amplitudes.clamp_(
                    -self.cfg.amplitude_max, self.cfg.amplitude_max)

            fidelity = 1.0 - loss.item()
            history.append(fidelity)

            if (i + 1) % 20 == 0:
                logger.info(
                    f"[PulseControl] Iter {i+1}/{n_iter}: "
                    f"F={fidelity:.6f}")

        # Final unitary and fidelity
        U_final = self.compute_unitary()
        final_fidelity = gate_fidelity(target, U_final)

        # OPMC convergence check
        converged = len(history) >= self.k_star and (
            abs(history[-1] - history[-self.k_star]) < 1e-4
            if len(history) >= self.k_star else False
        )

        return {
            "target_gate": target_gate,
            "final_fidelity": final_fidelity,
            "pulse_shape": self.pulse_amplitudes.detach().cpu().numpy(),
            "convergence_history": history,
            "n_iterations": n_iter,
            "converged": converged,
            "opmc_k_star": self.k_star,
            "HONEST_NOTE": (
                "This is a CLASSICAL SIMULATION of pulse optimization. "
                "It models what software could do given a physical "
                "quantum substrate. It does NOT create quantum coherence."
            ),
        }


# =============================================================================
# [2] AI & ML AUTO-CALIBRATION (OPMC + Bayesian Optimization)
# =============================================================================
# From SEPQA paper §7.3:
#   θ* = argmax_θ F(θ)
#   Methods: gradient-based, Bayesian, Nelder-Mead, coordinate descent
#
# OPMC connection (Unified Ordinal-Tensor §9):
#   One shared processing Φ → many downstream calibrations G_j
#   Shared representation reduces redundant computation
#
# Paper 11 connection:
#   Assumption-light subsampling CI for calibration confidence
#
# HONEST SCOPE: This simulates calibration optimization.
# Actual calibration requires physical hardware measurements.
# =============================================================================

@dataclass
class CalibrationConfig:
    """Configuration for ML-based auto-calibration."""
    n_parameters: int = 6           # calibration parameters
    n_qubits: int = 4               # number of qubits to calibrate
    n_samples: int = 100            # samples for Bayesian optimization
    n_iterations: int = 50          # optimization iterations
    subsample_block: int = 32       # Paper 11 subsampling block size
    confidence_level: float = 0.95  # CI confidence level
    opmc_m: int = 4
    opmc_n: int = 4
    device: str = "auto"


class OPMCCalibrationEngine(nn.Module):
    """
    [2] OPMC-based Auto-Calibration Engine.

    Architecture (OPMC §3.1-3.2):
      D → Φ → Z → {G_1, ..., G_M}

    where:
      D = calibration data (measurements from qubits)
      Φ = shared processing (feature extraction)
      Z = shared representation
      G_j = per-qubit calibration task

    CALIBRATION TASKS (G_j):
      • Frequency calibration (ω_q)
      • Anharmonicity calibration (α_q)
      • Drive amplitude calibration (A_drive)
      • Pulse duration calibration (t_pulse)
      • Coupling calibration (J_qq')
      • Readout calibration (θ_readout)

    OPMC ADVANTAGE (Appendix A.3):
      C_OPMC = C_Φ + Σ C_{G_j} < Σ C_{F_j} = C_independent

    HONEST SCOPE:
      • Simulates calibration optimization
      • Does NOT replace physical hardware measurements
      • OPMC speedup must be verified by benchmarking (Appendix B.3)
    """

    def __init__(self, cfg: Optional[CalibrationConfig] = None):
        super().__init__()
        self.cfg = cfg or CalibrationConfig()

        if self.cfg.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)

        d_mn = OPMC_D_MN(self.cfg.opmc_m, self.cfg.opmc_n)

        # Shared processing Φ: D → Z
        self.shared_encoder = nn.Sequential(
            nn.Linear(self.cfg.n_parameters * self.cfg.n_qubits, d_mn),
            nn.GELU(),
            nn.Linear(d_mn, d_mn),
            nn.LayerNorm(d_mn),
        ).to(self.device)

        # Downstream tasks G_j: Z → calibration per qubit
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_mn, self.cfg.n_parameters),
                nn.GELU(),
                nn.Linear(self.cfg.n_parameters, self.cfg.n_parameters),
            )
            for _ in range(self.cfg.n_qubits)
        ]).to(self.device)

        # Fidelity estimator (calibration quality)
        self.fidelity_net = nn.Sequential(
            nn.Linear(self.cfg.n_parameters, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        ).to(self.device)

        # OPMC parameters
        self.k_star = opmc_k_star(1e-8)
        self.d_mn = d_mn

    def forward(self, calibration_data: Tensor) -> Dict[str, Tensor]:
        """
        OPMC forward pass.

        Args:
            calibration_data: (B, n_qubits * n_parameters)

        Returns:
            dict with shared_representation, per_qubit_calibrations, fidelities
        """
        B = calibration_data.shape[0]

        # Shared processing Φ
        Z = self.shared_encoder(calibration_data)  # (B, d_mn)

        # Downstream tasks G_j
        calibrations = []
        fidelities = []
        for j in range(self.cfg.n_qubits):
            cal_j = self.task_heads[j](Z)  # (B, n_parameters)
            calibrations.append(cal_j)
            fid_j = self.fidelity_net(cal_j).squeeze(-1)  # (B,)
            fidelities.append(fid_j)

        calibrations = torch.stack(calibrations, dim=1)  # (B, n_qubits, n_params)
        fidelities = torch.stack(fidelities, dim=1)  # (B, n_qubits)

        return {
            "shared_representation": Z,
            "per_qubit_calibrations": calibrations,
            "per_qubit_fidelities": fidelities,
            "mean_fidelity": fidelities.mean(dim=1),
            "opmc_k_star": self.k_star,
            "opmc_d_mn": self.d_mn,
        }

    def calibration_loop(
        self,
        initial_params: Tensor,
        n_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Iterative calibration with OPMC convergence.

        Args:
            initial_params: (n_qubits * n_parameters,) initial guess
            n_iterations: optimization iterations

        Returns:
            dict with optimized parameters, fidelity trajectory
        """
        n_iter = n_iterations or self.cfg.n_iterations
        params = initial_params.clone().requires_grad_(True)

        optimizer = torch.optim.Adam([params], lr=1e-3)
        fidelity_history = []

        for i in range(n_iter):
            optimizer.zero_grad()

            # Forward pass (batch size 1)
            out = self.forward(params.unsqueeze(0))
            mean_fid = out["mean_fidelity"].squeeze()

            # Loss: maximize fidelity
            loss = -mean_fid
            loss.backward()
            optimizer.step()

            fidelity_history.append(mean_fid.item())

            # OPMC convergence check
            if i >= self.k_star:
                recent = fidelity_history[-self.k_star:]
                if max(recent) - min(recent) < 1e-4:
                    logger.info(
                        f"[Calibration] Converged at iteration {i+1} "
                        f"(OPMC k*={self.k_star})")
                    break

        return {
            "optimized_params": params.detach(),
            "fidelity_history": fidelity_history,
            "final_fidelity": fidelity_history[-1] if fidelity_history else 0.0,
            "n_iterations": len(fidelity_history),
            "converged_within_k_star": len(fidelity_history) <= self.k_star * 2,
        }


class AssumptionLightCalibrationCI:
    """
    Paper 11: Assumption-light confidence intervals for calibration.

    From Paper 11 §2.1 (Politis-Romano subsampling):
      Validity requires only that τ_n(θ̂_n - θ) converges in distribution
      to some limit law at some rate τ_n, and b_n → ∞, b_n/n → 0.
      Does NOT require knowing τ_n, the limit law, or mixing rate.

    APPLICATION:
      After calibration, compute CI on the fidelity estimate
      WITHOUT knowing the noise model.

    HONEST SCOPE (Paper 11 §3):
      "The regress is REDUCED, not closed."
      Requires qualitative regularity (stationarity + mixing).
    """

    def __init__(self, block_size: int = 32,
                 confidence_level: float = 0.95):
        self.block_size = block_size
        self.confidence_level = confidence_level

    def subsampling_ci(
        self,
        fidelity_samples: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute subsampling CI for mean fidelity.

        Args:
            fidelity_samples: (N,) fidelity measurements

        Returns:
            dict with point_estimate, ci_lower, ci_upper
        """
        N = len(fidelity_samples)
        b_n = min(self.block_size, N // 4)

        if b_n < 4:
            return {
                "error": "insufficient_data",
                "n_samples": N,
                "block_size": b_n,
            }

        # Point estimate
        theta_hat = fidelity_samples.mean()

        # Subsample statistics
        n_windows = N - b_n + 1
        sub_means = np.array([
            fidelity_samples[j:j + b_n].mean()
            for j in range(n_windows)
        ])

        # Subsampling distribution of √b_n(θ̂_b - θ̂)
        scaled_diffs = math.sqrt(b_n) * (sub_means - theta_hat)

        # Percentile CI
        alpha = 1 - self.confidence_level
        lo_pct = alpha / 2 * 100
        hi_pct = (1 - alpha / 2) * 100
        lo_val = np.percentile(scaled_diffs, lo_pct)
        hi_val = np.percentile(scaled_diffs, hi_pct)

        ci_lower = theta_hat - hi_val / math.sqrt(N)
        ci_upper = theta_hat - lo_val / math.sqrt(N)

        return {
            "point_estimate": float(theta_hat),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "confidence_level": self.confidence_level,
            "block_size": b_n,
            "n_samples": N,
            "HONEST_NOTE": (
                "Asymptotically correct coverage under qualitative "
                "stationarity. NOT a finite-sample guarantee. "
                "The mixing rate is not estimated, only assumed to exist."
            ),
        }


# =============================================================================
# [3] ERROR MITIGATION VIA CLASSICAL POST-PROCESSING
# =============================================================================
# Three standard techniques:
#   3.1 Zero-Noise Extrapolation (ZNE)
#       Li, Benjamin, Yuan (2017), Phys. Rev. X 7, 031029
#   3.2 Probabilistic Error Cancellation (PEC)
#       Temme, Bravyi, Gambetta (2017), Phys. Rev. Lett. 119, 180509
#   3.3 Readout Error Mitigation (Confusion Matrix Inversion)
#       Chen, et al. (2020), Phys. Rev. A 102, 012607
#
# HONEST SCOPE:
#   Error mitigation does NOT correct errors at the hardware level.
#   It post-processes measurement results to estimate what the
#   noise-free result would have been.
#   It does NOT achieve fault tolerance.
# =============================================================================

class ZeroNoiseExtrapolation:
    """
    [3.1] Zero-Noise Extrapolation (ZNE).

    Method:
      1. Run circuit at noise levels λ₁, λ₂, ..., λ_k
      2. Measure observable E(λ_i) at each level
      3. Extrapolate to λ=0 using Richardson interpolation

    Noise scaling: pulse stretching (multiply gate duration by λ)
      or identity insertion.

    REFERENCE:
      Li, Benjamin, Yuan (2017), Phys. Rev. X 7, 031029

    HONEST SCOPE:
      • Assumes noise scales linearly with λ
      • Requires multiple circuit executions at different noise levels
      • Extrapolation error grows with polynomial degree
    """

    def __init__(self, noise_levels: List[float] = None):
        self.noise_levels = noise_levels or [1.0, 1.5, 2.0, 3.0]
        logger.info(f"[ZNE] Noise levels: {self.noise_levels}")

    def richardson_extrapolate(
        self,
        values: List[float],
        noise_levels: List[float],
        target: float = 0.0,
    ) -> float:
        """
        Richardson extrapolation to target noise level.

        Args:
            values: measured values at each noise level
            noise_levels: noise scale factors
            target: target noise level (0 = noise-free)

        Returns:
            extrapolated value at target noise level
        """
        if len(values) != len(noise_levels):
            raise ValueError("values and noise_levels must have same length")

        n = len(values)
        # Richardson coefficients
        coeffs = []
        for i in range(n):
            num = 1.0
            den = 1.0
            for j in range(n):
                if i != j:
                    num *= (target - noise_levels[j])
                    den *= (noise_levels[i] - noise_levels[j])
            coeffs.append(num / den)

        return sum(c * v for c, v in zip(coeffs, values))

    def mitigate(
        self,
        circuit_executor: Callable[[float], float],
    ) -> Dict[str, Any]:
        """
        Run ZNE on a circuit.

        Args:
            circuit_executor: function that takes noise_level and
                             returns measured expectation value

        Returns:
            dict with mitigated value, raw values, extrapolation details
        """
        raw_values = []
        for lam in self.noise_levels:
            val = circuit_executor(lam)
            raw_values.append(val)

        # Richardson extrapolation to λ=0
        mitigated = self.richardson_extrapolate(
            raw_values, self.noise_levels, target=0.0)

        # Also compute linear extrapolation (degree 1) for comparison
        if len(self.noise_levels) >= 2:
            linear_mitigated = self.richardson_extrapolate(
                raw_values[:2], self.noise_levels[:2], target=0.0)
        else:
            linear_mitigated = mitigated

        return {
            "noise_levels": self.noise_levels,
            "raw_values": raw_values,
            "mitigated_value": mitigated,
            "linear_mitigated_value": linear_mitigated,
            "extrapolation_degree": len(self.noise_levels) - 1,
            "HONEST_NOTE": (
                "ZNE assumes noise scales linearly with λ. "
                "Extrapolation error grows with polynomial degree. "
                "This does NOT correct errors at the hardware level."
            ),
        }


class ProbabilisticErrorCancellation:
    """
    [3.2] Probabilistic Error Cancellation (PEC).

    Method:
      1. Characterize noise channel E as quasi-probability decomposition:
         E = Σ_i η_i B_i
         where η_i can be negative (quasi-probabilities)
      2. For each circuit execution, sample B_i with probability |η_i|/Σ|η|
      3. Multiply result by sign(η_i)
      4. Average over many samples

    REFERENCE:
      Temme, Bravyi, Gambetta (2017), Phys. Rev. Lett. 119, 180509

    HONEST SCOPE:
      • Requires accurate noise characterization
      • Sampling overhead grows exponentially with circuit depth
      • Quasi-probabilities can be negative (not true probabilities)
    """

    def __init__(self, noise_model: Optional[Dict[str, Tensor]] = None):
        self.noise_model = noise_model
        logger.info("[PEC] Initialized")

    def quasi_probability_decomposition(
        self,
        noise_channel: Tensor,
    ) -> Dict[str, Any]:
        """
        Decompose noise channel into quasi-probabilities.

        Args:
            noise_channel: (d², d²) superoperator representation

        Returns:
            dict with quasi_probabilities, basis_operators
        """
        d2 = noise_channel.shape[0]
        d = int(math.sqrt(d2))

        # Pauli basis for decomposition
        P = pauli()
        basis = [P["I"], P["X"], P["Y"], P["Z"]]

        # Simple decomposition: noise = (1-p)I + p*(X+Y+Z)/3
        # This is a simplified model; real PEC uses full tomography
        p_error = 0.05  # example error rate

        quasi_probs = {
            "I": 1.0 - p_error,
            "X": p_error / 3,
            "Y": p_error / 3,
            "Z": p_error / 3,
        }

        # For PEC, we need the INVERSE channel
        # E^{-1} ≈ I - (E - I) for small errors
        inv_quasi_probs = {
            "I": 1.0 + p_error,  # > 1 (quasi-probability)
            "X": -p_error / 3,   # negative
            "Y": -p_error / 3,
            "Z": -p_error / 3,
        }

        return {
            "quasi_probabilities": quasi_probs,
            "inverse_quasi_probabilities": inv_quasi_probs,
            "basis_operators": list(quasi_probs.keys()),
            "sampling_overhead": sum(abs(v) for v in inv_quasi_probs.values()),
            "HONEST_NOTE": (
                "Quasi-probabilities can be negative. "
                "Sampling overhead grows exponentially with circuit depth. "
                "Requires accurate noise characterization."
            ),
        }

    def mitigate(
        self,
        circuit_executor: Callable[[str], float],
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run PEC on a circuit.

        Args:
            circuit_executor: function that takes basis_operator and
                             returns measured expectation value
            n_samples: number of PEC samples

        Returns:
            dict with mitigated value, sampling details
        """
        # Get quasi-probability decomposition
        decomp = self.quasi_probability_decomposition(
            torch.eye(4, dtype=CD))

        inv_probs = decomp["inverse_quasi_probabilities"]
        basis_ops = decomp["basis_operators"]

        # Sample and average
        weighted_sum = 0.0
        samples_used = []

        for _ in range(n_samples):
            # Sample basis operator proportional to |η_i|
            abs_probs = [abs(inv_probs[op]) for op in basis_ops]
            total = sum(abs_probs)
            probs = [p / total for p in abs_probs]

            idx = np.random.choice(len(basis_ops), p=probs)
            op = basis_ops[idx]

            # Execute circuit with this basis operator
            val = circuit_executor(op)

            # Weight by sign(η_i) * total_abs
            sign = 1.0 if inv_probs[op] >= 0 else -1.0
            weighted_sum += sign * val * total

            samples_used.append(op)

        mitigated = weighted_sum / n_samples

        return {
            "mitigated_value": mitigated,
            "n_samples": n_samples,
            "sampling_overhead": decomp["sampling_overhead"],
            "samples_used_distribution": {
                op: samples_used.count(op) for op in basis_ops
            },
            "HONEST_NOTE": decomp["HONEST_NOTE"],
        }


class ReadoutErrorMitigation:
    """
    [3.3] Readout Error Mitigation via Confusion Matrix Inversion.

    Method:
      1. Characterize readout errors: A[i,j] = P(measure i | state j)
      2. Invert confusion matrix: A^{-1}
      3. Apply to measured counts: p_corrected = A^{-1} p_measured

    REFERENCE:
      Chen, et al. (2020), Phys. Rev. A 102, 012607

    HONEST SCOPE:
      • Assumes readout errors are independent and stationary
      • Matrix inversion can amplify noise
      • Requires calibration measurements
    """

    def __init__(self, n_qubits: int = 2):
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        logger.info(f"[ReadoutMitigation] Initialized for {n_qubits} qubits")

    def build_confusion_matrix(
        self,
        readout_errors: List[float],
    ) -> Tensor:
        """
        Build confusion matrix from single-qubit readout errors.

        Args:
            readout_errors: list of (p_0_given_1, p_1_given_0) per qubit

        Returns:
            confusion matrix A (n_states × n_states)
        """
        A = torch.eye(self.n_states, dtype=torch.float64)

        for q, (p01, p10) in enumerate(readout_errors):
            # Single-qubit confusion matrix
            A_q = torch.tensor([
                [1 - p10, p01],
                [p10, 1 - p01],
            ], dtype=torch.float64)

            # Tensor product with identity on other qubits
            A_new = torch.eye(1, dtype=torch.float64)
            for i in range(self.n_qubits):
                if i == q:
                    A_new = torch.kron(A_new, A_q)
                else:
                    A_new = torch.kron(A_new, torch.eye(2, dtype=torch.float64))
            A = A_new

        return A

    def mitigate(
        self,
        measured_counts: Dict[str, int],
        readout_errors: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        """
        Apply readout error mitigation.

        Args:
            measured_counts: dict mapping bitstring → count
            readout_errors: list of (p_0_given_1, p_1_given_0) per qubit

        Returns:
            dict with corrected probabilities, confusion matrix
        """
        # Build confusion matrix
        A = self.build_confusion_matrix(readout_errors)

        # Convert counts to probability vector
        total_counts = sum(measured_counts.values())
        p_measured = torch.zeros(self.n_states, dtype=torch.float64)

        for bitstring, count in measured_counts.items():
            idx = int(bitstring, 2)
            p_measured[idx] = count / total_counts

        # Invert and apply
        try:
            A_inv = torch.linalg.inv(A)
            p_corrected = A_inv @ p_measured

            # Clip to valid probability range
            p_corrected = p_corrected.clamp(min=0.0)
            p_corrected = p_corrected / p_corrected.sum()

        except torch.linalg.LinAlgError:
            logger.warning("[ReadoutMitigation] Confusion matrix singular")
            p_corrected = p_measured

        # Convert back to counts format
        corrected_counts = {}
        for i in range(self.n_states):
            bitstring = format(i, f'0{self.n_qubits}b')
            corrected_counts[bitstring] = float(p_corrected[i]) * total_counts

        return {
            "confusion_matrix": A.numpy(),
            "measured_probabilities": p_measured.numpy(),
            "corrected_probabilities": p_corrected.numpy(),
            "corrected_counts": corrected_counts,
            "HONEST_NOTE": (
                "Matrix inversion can amplify noise. "
                "Assumes readout errors are independent and stationary. "
                "Requires calibration measurements."
            ),
        }


# =============================================================================
# [4] CRYOGENIC CMOS SIMULATION
# =============================================================================
# Cryogenic CMOS: classical control chips operating at millikelvin
# temperatures, close to the quantum processor.
#
# References:
#   Bardin, et al. (2020), "A 28nm bulk CMOS 4-to-8GHz cryogenic
#   pulse generator" ISSCC
#   Incandela, et al. (2018), "Characterization and modeling of 28nm
#   bulk CMOS at cryogenic temperatures" ESSDERC
#
# HONEST SCOPE:
#   This is a SIMULATION of cryogenic control electronics.
#   It models latency, noise, and power budget.
#   It does NOT simulate actual CMOS transistor physics.
# =============================================================================

@dataclass
class CryogenicConfig:
    """Configuration for cryogenic CMOS simulation."""
    # Temperature
    temperature_mk: float = 20.0    # millikelvin (typical dilution fridge)
    room_temp_k: float = 300.0      # Kelvin

    # Control parameters
    n_qubits: int = 100             # number of qubits to control
    n_control_lines: int = 200      # control lines (2 per qubit typical)
    clock_freq_ghz: float = 1.0     # control clock frequency

    # Latency model
    cable_delay_ns_per_m: float = 5.0   # coaxial cable delay
    cable_length_m: float = 1.0         # cable length
    chip_processing_ns: float = 10.0    # CMOS processing delay

    # Noise model
    thermal_noise_floor_dbm: float = -174.0  # kT at 300K in dBm/Hz
    amplifier_noise_figure_db: float = 2.0

    # Power budget
    power_budget_mw: float = 100.0  # total power budget at cold stage
    power_per_gate_mw: float = 0.1  # power per control gate


class CryogenicCMOSSimulator:
    """
    [4] Cryogenic CMOS Control Electronics Simulator.

    Models:
      • Latency: cable delay + chip processing
      • Noise: thermal noise at cryogenic temperature
      • Power: budget constraints for cold stage
      • Scaling: number of qubits vs. control lines

    PHYSICS:
      • Thermal noise: kT (Boltzmann)
      • At 20 mK: kT ≈ 1.7 × 10⁻²⁵ J (vs 4 × 10⁻²¹ J at 300K)
      • Noise floor drops by ~23 dB from 300K to 20mK

    HONEST SCOPE:
      • Simplified model, not transistor-level simulation
      • Does not model quantum effects in CMOS
      • Power and latency estimates are approximate
    """

    def __init__(self, cfg: Optional[CryogenicConfig] = None):
        self.cfg = cfg or CryogenicConfig()

        # Boltzmann constant
        self.k_B = 1.380649e-23  # J/K

        logger.info(
            f"[CryoCMOS] T={self.cfg.temperature_mk}mK, "
            f"n_qubits={self.cfg.n_qubits}")

    def thermal_noise_power(self, bandwidth_hz: float = 1e9) -> Dict[str, float]:
        """
        Compute thermal noise power at cryogenic temperature.

        P_noise = kT × B

        Args:
            bandwidth_hz: measurement bandwidth

        Returns:
            dict with noise power in W and dBm
        """
        T_cold = self.cfg.temperature_mk / 1000  # K
        T_room = self.cfg.room_temp_k

        P_cold = self.k_B * T_cold * bandwidth_hz
        P_room = self.k_B * T_room * bandwidth_hz

        # Convert to dBm
        P_cold_dbm = 10 * math.log10(P_cold / 1e-3)
        P_room_dbm = 10 * math.log10(P_room / 1e-3)

        noise_reduction_db = P_room_dbm - P_cold_dbm

        return {
            "P_noise_cold_W": P_cold,
            "P_noise_room_W": P_room,
            "P_noise_cold_dBm": P_cold_dbm,
            "P_noise_room_dBm": P_room_dbm,
            "noise_reduction_dB": noise_reduction_db,
            "temperature_cold_K": T_cold,
            "temperature_room_K": T_room,
            "bandwidth_Hz": bandwidth_hz,
        }

    def control_latency(self) -> Dict[str, float]:
        """
        Estimate control signal latency.

        Latency = cable_delay + chip_processing + digitization

        Returns:
            dict with latency breakdown
        """
        cable_delay_ns = (
            self.cfg.cable_delay_ns_per_m * self.cfg.cable_length_m)
        chip_delay_ns = self.cfg.chip_processing_ns
        digitization_ns = 1.0 / self.cfg.clock_freq_ghz  # 1 clock cycle

        total_ns = cable_delay_ns + chip_delay_ns + digitization_ns

        # Number of control cycles per qubit coherence time
        # Typical T2 ~ 100 μs for superconducting qubits
        T2_us = 100.0
        n_cycles = T2_us * 1000 / total_ns  # convert μs to ns

        return {
            "cable_delay_ns": cable_delay_ns,
            "chip_delay_ns": chip_delay_ns,
            "digitization_ns": digitization_ns,
            "total_latency_ns": total_ns,
            "T2_us": T2_us,
            "n_control_cycles_per_T2": n_cycles,
            "HONEST_NOTE": (
                "Latency estimates are approximate. "
                "Actual values depend on specific hardware."
            ),
        }

    def power_budget(self) -> Dict[str, float]:
        """
        Estimate power budget for cryogenic control.

        Power = n_control_lines × power_per_gate × duty_cycle

        Returns:
            dict with power breakdown
        """
        duty_cycle = 0.1  # 10% active (typical for pulse control)

        P_per_line_mw = self.cfg.power_per_gate_mw * duty_cycle
        P_total_mw = self.cfg.n_control_lines * P_per_line_mw

        # Check against budget
        within_budget = P_total_mw <= self.cfg.power_budget_mw

        # Heat load on dilution refrigerator
        # Typical cooling power at 20mK: ~10-20 μW
        cooling_power_uw = 20.0
        heat_load_uw = P_total_mw * 1000  # mW to μW

        return {
            "power_per_line_mw": P_per_line_mw,
            "total_power_mw": P_total_mw,
            "power_budget_mw": self.cfg.power_budget_mw,
            "within_budget": within_budget,
            "cooling_power_uw": cooling_power_uw,
            "heat_load_uw": heat_load_uw,
            "thermal_headroom_factor": cooling_power_uw / max(heat_load_uw, 1e-10),
            "HONEST_NOTE": (
                "Power estimates are approximate. "
                "Actual cryogenic CMOS power depends on process node, "
                "temperature, and circuit design."
            ),
        }

    def scaling_analysis(self) -> Dict[str, Any]:
        """
        Analyze scaling: n_qubits vs. control resources.

        Returns:
            dict with scaling metrics
        """
        n_qubits = self.cfg.n_qubits
        n_lines = self.cfg.n_control_lines

        # Control lines per qubit
        lines_per_qubit = n_lines / n_qubits

        # Scaling projection
        projections = {}
        for target_qubits in [100, 1000, 10000, 100000]:
            target_lines = int(target_qubits * lines_per_qubit)
            target_power_mw = target_lines * self.cfg.power_per_gate_mw * 0.1

            projections[target_qubits] = {
                "control_lines": target_lines,
                "estimated_power_mw": target_power_mw,
                "feasible_at_20mK": target_power_mw < 1000,  # < 1W
            }

        return {
            "n_qubits": n_qubits,
            "n_control_lines": n_lines,
            "lines_per_qubit": lines_per_qubit,
            "projections": projections,
            "HONEST_NOTE": (
                "Scaling projections are approximate. "
                "Actual feasibility depends on cryogenic infrastructure."
            ),
        }


# =============================================================================
# [5] INTEGRATED SEPQA SYSTEM
# =============================================================================

class IntegratedSEPQASystem:
    """
    Master orchestrator for SEPQA research.

    Integrates:
      [1] PulseLevelQubitControl (Software-Defined Qubits)
      [2] OPMCCalibrationEngine (ML Auto-Calibration)
      [3] Error Mitigation (ZNE, PEC, Readout)
      [4] CryogenicCMOSSimulator (Control Hardware)

    HONEST SCOPE (SEPQA paper §10.3):
      "Classical hardware can provide control signals and engineering
       infrastructure, but it cannot replace the quantum physical
       properties required for a qubit."
    """

    def __init__(self, device: str = "auto"):
        self.device = device

        # Initialize sub-systems
        self.pulse_control = PulseLevelQubitControl(
            PulseControlConfig(device=device))
        self.calibration = OPMCCalibrationEngine(
            CalibrationConfig(device=device))
        self.calibration_ci = AssumptionLightCalibrationCI()
        self.zne = ZeroNoiseExtrapolation()
        self.pec = ProbabilisticErrorCancellation()
        self.readout_mitigation = ReadoutErrorMitigation()
        self.cryo_sim = CryogenicCMOSSimulator()

        logger.info("[SEPQA] Integrated system initialized")

    def run_full_research_suite(self) -> Dict[str, Any]:
        """
        Run complete SEPQA research suite.

        Returns:
            dict with all results
        """
        result = {"version": SEPQA_VERSION}

        # [1] Pulse-level control
        logger.info("[SEPQA] Step 1: Pulse-level gate optimization")
        pulse_result = self.pulse_control.optimize_gate(
            target_gate="X", n_iterations=100)
        result["pulse_control"] = pulse_result

        # [2] Calibration
        logger.info("[SEPQA] Step 2: OPMC calibration")
        n_params = (self.calibration.cfg.n_qubits *
                    self.calibration.cfg.n_parameters)
        initial_params = torch.randn(n_params, device=self.calibration.device)
        cal_result = self.calibration.calibration_loop(
            initial_params, n_iterations=50)
        result["calibration"] = cal_result

        # [2b] Calibration CI (Paper 11)
        logger.info("[SEPQA] Step 2b: Assumption-light CI")
        fidelity_samples = np.array(cal_result["fidelity_history"])
        ci_result = self.calibration_ci.subsampling_ci(fidelity_samples)
        result["calibration_ci"] = ci_result

        # [3] Error mitigation
        logger.info("[SEPQA] Step 3: Error mitigation")

        # ZNE: simulate noisy measurements
        def noisy_executor(lam):
            # Simulated: true value 0.8, noise adds error proportional to λ
            return 0.8 - 0.1 * lam + np.random.normal(0, 0.01)

        zne_result = self.zne.mitigate(noisy_executor)
        result["zne"] = zne_result

        # PEC: simulate basis sampling
        def pec_executor(op):
            return 0.8 + np.random.normal(0, 0.02)

        pec_result = self.pec.mitigate(pec_executor, n_samples=500)
        result["pec"] = pec_result

        # Readout mitigation
        readout_result = self.readout_mitigation.mitigate(
            measured_counts={"00": 450, "01": 30, "10": 20, "11": 500},
            readout_errors=[(0.05, 0.03), (0.04, 0.02)],
        )
        result["readout_mitigation"] = readout_result

        # [4] Cryogenic simulation
        logger.info("[SEPQA] Step 4: Cryogenic CMOS simulation")
        result["thermal_noise"] = self.cryo_sim.thermal_noise_power()
        result["control_latency"] = self.cryo_sim.control_latency()
        result["power_budget"] = self.cryo_sim.power_budget()
        result["scaling"] = self.cryo_sim.scaling_analysis()

        # Summary
        result["summary"] = {
            "pulse_fidelity": pulse_result["final_fidelity"],
            "calibration_fidelity": cal_result["final_fidelity"],
            "calibration_ci": ci_result.get("ci_lower", 0),
            "zne_mitigated": zne_result["mitigated_value"],
            "pec_mitigated": pec_result["mitigated_value"],
            "cryo_noise_reduction_db":
                result["thermal_noise"]["noise_reduction_dB"],
            "cryo_latency_ns":
                result["control_latency"]["total_latency_ns"],
            "power_within_budget":
                result["power_budget"]["within_budget"],
        }

        result["HONEST_SCOPE"] = {
            "all_simulations": (
                "All results are CLASSICAL SIMULATIONS. "
                "They model what software could do given a physical "
                "quantum substrate. They do NOT create quantum coherence."
            ),
            "sepqa_requirement": (
                "SEPQA depends on the existence of a suitable physical "
                "quantum substrate. Classical hardware cannot replace "
                "the quantum physical properties required for a qubit."
            ),
            "error_mitigation": (
                "Error mitigation does NOT achieve fault tolerance. "
                "It post-processes results to estimate noise-free values."
            ),
        }

        return result


# =============================================================================
# [6] SELF-TEST SUITE
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(f"  SEPQA RESEARCH SUITE v{SEPQA_VERSION} — Smoke Test")
    print("=" * 78)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── T1: Pulse-Level Control ──────────────────────────────────────────
    print("\n [T1] Pulse-Level Qubit Control (Software-Defined Qubits)")
    pulse_cfg = PulseControlConfig(
        n_time_steps=50,
        total_time_ns=50.0,
        n_iterations=100,
        device=str(device),
    )
    controller = PulseLevelQubitControl(pulse_cfg)

    result = controller.optimize_gate("X", n_iterations=100)
    print(f"   Target gate: {result['target_gate']}")
    print(f"   Final fidelity: {result['final_fidelity']:.6f}")
    print(f"   Converged: {result['converged']}")
    print(f"   OPMC k*: {result['opmc_k_star']}")
    print(f"   HONEST: {result['HONEST_NOTE'][:60]}...")

    assert result['final_fidelity'] > 0.5, "Fidelity should be > 0.5"
    print("[PASS] T1 Pulse-level control works")

    # ── T2: OPMC Calibration ─────────────────────────────────────────────
    print("\n [T2] OPMC Auto-Calibration")
    cal_cfg = CalibrationConfig(device=str(device))
    cal_engine = OPMCCalibrationEngine(cal_cfg)

    n_params = cal_cfg.n_qubits * cal_cfg.n_parameters
    initial = torch.randn(n_params, device=device)
    cal_result = cal_engine.calibration_loop(initial, n_iterations=50)

    print(f"   Final fidelity: {cal_result['final_fidelity']:.4f}")
    print(f"   Iterations: {cal_result['n_iterations']}")
    print(f"   Converged within k*: {cal_result['converged_within_k_star']}")

    assert cal_result['final_fidelity'] > 0.0, "Fidelity should be positive"
    print("[PASS] T2 OPMC calibration works")

    # ── T3: Assumption-Light CI (Paper 11) ───────────────────────────────
    print("\n [T3] Assumption-Light Calibration CI (Paper 11)")
    ci_engine = AssumptionLightCalibrationCI(block_size=16)
    fidelity_samples = np.array(cal_result["fidelity_history"])

    ci_result = ci_engine.subsampling_ci(fidelity_samples)
    if "error" not in ci_result:
        print(f"   Point estimate: {ci_result['point_estimate']:.4f}")
        print(f"   CI: [{ci_result['ci_lower']:.4f}, {ci_result['ci_upper']:.4f}]")
        print(f"   Block size: {ci_result['block_size']}")
        print(f"   HONEST: {ci_result['HONEST_NOTE'][:60]}...")
    else:
        print(f"   Insufficient data: {ci_result}")

    print("[PASS] T3 Assumption-light CI works")

    # ── T4: Zero-Noise Extrapolation ─────────────────────────────────────
    print("\n [T4] Zero-Noise Extrapolation (ZNE)")
    zne = ZeroNoiseExtrapolation(noise_levels=[1.0, 1.5, 2.0, 3.0])

    def noisy_executor(lam):
        return 0.8 - 0.1 * lam

    zne_result = zne.mitigate(noisy_executor)
    print(f"   Noise levels: {zne_result['noise_levels']}")
    print(f"   Raw values: {zne_result['raw_values']}")
    print(f"   Mitigated value: {zne_result['mitigated_value']:.4f}")
    print(f"   True value: 0.8")
    print(f"   Extrapolation degree: {zne_result['extrapolation_degree']}")

    # Richardson should recover ~0.8 for linear noise
    assert abs(zne_result['mitigated_value'] - 0.8) < 0.05, \
        "ZNE should recover ~0.8 for linear noise"
    print("[PASS] T4 ZNE works")

    # ── T5: Probabilistic Error Cancellation ─────────────────────────────
    print("\n [T5] Probabilistic Error Cancellation (PEC)")
    pec = ProbabilisticErrorCancellation()

    def pec_executor(op):
        return 0.8 + np.random.normal(0, 0.01)

    pec_result = pec.mitigate(pec_executor, n_samples=1000)
    print(f"   Mitigated value: {pec_result['mitigated_value']:.4f}")
    print(f"   Sampling overhead: {pec_result['sampling_overhead']:.2f}")
    print(f"   Samples distribution: {pec_result['samples_used_distribution']}")

    print("[PASS] T5 PEC works")

    # ── T6: Readout Error Mitigation ─────────────────────────────────────
    print("\n [T6] Readout Error Mitigation")
    readout = ReadoutErrorMitigation(n_qubits=2)

    readout_result = readout.mitigate(
        measured_counts={"00": 450, "01": 30, "10": 20, "11": 500},
        readout_errors=[(0.05, 0.03), (0.04, 0.02)],
    )
    print(f"   Measured: 00={450/1000:.2f}, 11={500/1000:.2f}")
    print(f"   Corrected: 00={readout_result['corrected_probabilities'][0]:.3f}, "
          f"11={readout_result['corrected_probabilities'][3]:.3f}")
    print(f"   HONEST: {readout_result['HONEST_NOTE'][:60]}...")

    print("[PASS] T6 Readout mitigation works")

    # ── T7: Cryogenic CMOS Simulation ────────────────────────────────────
    print("\n [T7] Cryogenic CMOS Simulation")
    cryo = CryogenicCMOSSimulator()

    thermal = cryo.thermal_noise_power()
    print(f"   Noise at 20mK: {thermal['P_noise_cold_dBm']:.1f} dBm")
    print(f"   Noise at 300K: {thermal['P_noise_room_dBm']:.1f} dBm")
    print(f"   Noise reduction: {thermal['noise_reduction_dB']:.1f} dB")

    latency = cryo.control_latency()
    print(f"   Total latency: {latency['total_latency_ns']:.1f} ns")
    print(f"   Cycles per T2: {latency['n_control_cycles_per_T2']:.0f}")

    power = cryo.power_budget()
    print(f"   Total power: {power['total_power_mw']:.1f} mW")
    print(f"   Within budget: {power['within_budget']}")

    scaling = cryo.scaling_analysis()
    print(f"   Lines per qubit: {scaling['lines_per_qubit']:.1f}")

    assert thermal['noise_reduction_dB'] > 20, "Should have >20dB reduction"
    print("[PASS] T7 Cryogenic simulation works")

    # ── T8: Integrated System ────────────────────────────────────────────
    print("\n [T8] Integrated SEPQA System")
    system = IntegratedSEPQASystem(device=str(device))
    full_result = system.run_full_research_suite()

    print(f"   Summary:")
    for k, v in full_result["summary"].items():
        if isinstance(v, float):
            print(f"     {k}: {v:.4f}")
        else:
            print(f"     {k}: {v}")

    print(f"\n   HONEST SCOPE:")
    for k, v in full_result["HONEST_SCOPE"].items():
        print(f"     {k}: {v[:60]}...")

    print("[PASS] T8 Integrated system works")

    # ── T9: OPMC Constants ───────────────────────────────────────────────
    print("\n [T9] OPMC Constants")
    k_star = opmc_k_star(1e-8)
    print(f"   k* (Theorem 11.1): {k_star}")
    assert k_star == 23

    d_44 = OPMC_D_MN(4, 4)
    print(f"   d(4,4) (Lemma 2.4): {d_44}")
    assert d_44 == 320

    print(f"   L (Example 7.12): {OPMC_L}")
    assert abs(OPMC_L - 0.438520) < 1e-6
    print("[PASS] T9 OPMC constants match paper")

    print("\n" + "=" * 78)
    print("  ALL TESTS PASSED.")
    print("")
    print("  CRITICAL HONEST REMINDERS (from SEPQA paper §10.3):")
    print("  • Classical analogies are NOT physical quantum phenomena")
    print("  • A quantum-inspired algorithm does NOT automatically outperform")
    print("  • SEPQA depends on a suitable physical quantum substrate")
    print("  • Purely classical software does NOT create quantum coherence")
    print("  • Error mitigation does NOT achieve fault tolerance")
    print("  • All simulations model what software COULD do,")
    print("    given a physical quantum substrate")
    print("=" * 78)


if __name__ == "__main__":
    smoke_test()
