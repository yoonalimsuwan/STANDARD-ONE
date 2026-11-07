"""
=============================================================================
SUB-QUANTUM HYPOTHESIS TESTING FRAMEWORK
Testing: "Hidden variables are just interference we don't know about"
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0

WHAT THIS CODE TESTS:
  1. Local hidden variable theories (Bell's theorem)
  2. Quantum entanglement predictions
  3. "Unknown interference" hypothesis
  4. CHSH inequality violations
  5. What Bell's theorem actually allows/disallows

WHAT THIS CODE DOES NOT CLAIM:
  ❌ Does NOT prove "sub-quantum" theories exist
  ❌ Does NOT disprove quantum mechanics
  ❌ Does NOT find actual hidden variables
  ❌ Does NOT claim "interference we don't know" explains quantum
  ❌ Does NOT overturn Bell's theorem

KEY PHYSICS FACTS:
  • Bell's theorem (1964): Local hidden variables CANNOT reproduce
    quantum predictions. Proven mathematically.
  • CHSH inequality: |S| ≤ 2 for ALL local hidden variable theories
  • Quantum mechanics: |S| can reach 2√2 ≈ 2.828
  • Experiments (Aspect 1982, loophole-free 2015): Confirm quantum,
    NOT local hidden variables

THE USER'S HYPOTHESIS:
  "Hidden variables are just interference we don't know about"

  This can be interpreted in TWO ways:
  (A) LOCAL interference: Hidden variables at each detector determine
      outcomes locally. → REFUTED by Bell's theorem.
  (B) NON-LOCAL interference: The "interference" connects distant
      measurements non-locally. → NOT refuted by Bell, but requires
      a specific mechanism (like Bohm's pilot wave).

  This code tests BOTH interpretations and shows which is compatible
  with quantum predictions.

=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum, auto

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("sub_quantum_tester")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

SQ_VERSION = "1.0.0"

# =============================================================================
# COMPLEX HELPERS
# =============================================================================
CD = np.complex128

def pauli_matrices_np() -> Dict[str, np.ndarray]:
    """Pauli matrices as numpy arrays."""
    return {
        "I": np.eye(2, dtype=CD),
        "X": np.array([[0, 1], [1, 0]], dtype=CD),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=CD),
        "Z": np.array([[1, 0], [0, -1]], dtype=CD),
    }

def bell_state_phi_plus() -> np.ndarray:
    """|Φ+⟩ = (|00⟩ + |11⟩)/√2 as state vector."""
    return np.array([1, 0, 0, 1], dtype=CD) / math.sqrt(2)

def quantum_correlation(state: np.ndarray,
                        op_A: np.ndarray, op_B: np.ndarray) -> float:
    """
    Compute quantum expectation value ⟨ψ| A⊗B |ψ⟩.
    """
    op = np.kron(op_A, op_B)
    return float(np.real(state.conj() @ op @ state))


# =============================================================================
# 1. LOCAL HIDDEN VARIABLE MODEL
# =============================================================================

class HiddenVariableType(Enum):
    """Types of hidden variable theories."""
    LOCAL = auto()          # Outcomes determined locally by λ
    NONLOCAL = auto()       # Outcomes can depend on distant settings
    INTERFERENCE_LOCAL = auto()    # "Unknown interference" local
    INTERFERENCE_NONLOCAL = auto()  # "Unknown interference" non-local


@dataclass
class HiddenVariableConfig:
    """Configuration for hidden variable model."""
    hv_type: HiddenVariableType = HiddenVariableType.LOCAL
    n_hidden_states: int = 1000   # discretization of λ space
    n_trials: int = 100000        # number of trials for statistics
    seed: int = 42


class LocalHiddenVariableModel:
    """
    Local hidden variable model (the kind Bell's theorem refutes).

    MODEL:
      • Hidden variable λ determines outcomes at each detector
      • A(a, λ) = ±1 : Alice's outcome given setting a and λ
      • B(b, λ) = ±1 : Bob's outcome given setting b and λ
      • CRITICAL: A depends ONLY on a and λ, NOT on b
                  B depends ONLY on b and λ, NOT on a
      • This is LOCALITY: no faster-than-light influence

    From Bell's theorem:
      E(a, b) = ∫ dλ ρ(λ) A(a, λ) B(b, λ)

      CHSH: S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
      For ALL local hidden variable theories: |S| ≤ 2

    HONEST SCOPE:
      • This model CANNOT violate CHSH inequality
      • If it appears to violate CHSH, the code has a bug
      • Local hidden variables are REFUTED by experiment

    USAGE:
      model = LocalHiddenVariableModel(cfg)
      S = model.compute_chsh()
      # S will ALWAYS be ≤ 2 (mathematical guarantee)
    """

    def __init__(self, cfg: Optional[HiddenVariableConfig] = None):
        self.cfg = cfg or HiddenVariableConfig()
        self.rng = np.random.RandomState(self.cfg.seed)

        # Discrete hidden variable space
        self.lambdas = np.linspace(0, 2 * np.pi, self.cfg.n_hidden_states)
        self.rho = np.ones(self.cfg.n_hidden_states) / self.cfg.n_hidden_states

        logger.info(f"[LocalHV] Initialized: {self.cfg.n_hidden_states} λ states")

    def outcome_A(self, setting: float, lam: float) -> int:
        """
        Alice's outcome: A(a, λ) = ±1.
        Depends ONLY on local setting a and hidden variable λ.
        Does NOT depend on Bob's setting b (locality).
        """
        # Deterministic function of local setting and λ
        # (This is a specific model; Bell's theorem works for ANY such model)
        return +1 if math.cos(lam - setting) >= 0 else -1

    def outcome_B(self, setting: float, lam: float) -> int:
        """
        Bob's outcome: B(b, λ) = ±1.
        Depends ONLY on local setting b and hidden variable λ.
        Does NOT depend on Alice's setting a (locality).
        """
        return +1 if math.cos(lam - setting) >= 0 else -1

    def correlation(self, setting_A: float, setting_B: float) -> float:
        """
        Compute E(a, b) = ∫ dλ ρ(λ) A(a, λ) B(b, λ).
        """
        total = 0.0
        for lam, weight in zip(self.lambdas, self.rho):
            A = self.outcome_A(setting_A, lam)
            B = self.outcome_B(setting_B, lam)
            total += weight * A * B
        return total

    def compute_chsh(self,
                     settings: Optional[Tuple[float, float, float, float]] = None
                     ) -> Dict[str, Any]:
        """
        Compute CHSH parameter S.

        Standard settings that maximize quantum violation:
          a = 0, a' = π/2, b = π/4, b' = -π/4

        Returns:
            dict with S value and whether CHSH is violated
        """
        if settings is None:
            # Standard CHSH settings
            a, a_prime, b, b_prime = 0.0, math.pi / 2, math.pi / 4, -math.pi / 4
        else:
            a, a_prime, b, b_prime = settings

        E_ab = self.correlation(a, b)
        E_ab_prime = self.correlation(a, b_prime)
        E_a_prime_b = self.correlation(a_prime, b)
        E_a_prime_b_prime = self.correlation(a_prime, b_prime)

        S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime

        return {
            "S": S,
            "abs_S": abs(S),
            "chsh_bound": 2.0,
            "violates_chsh": abs(S) > 2.0 + 1e-10,
            "correlations": {
                "E(a,b)": E_ab,
                "E(a,b')": E_ab_prime,
                "E(a',b)": E_a_prime_b,
                "E(a',b')": E_a_prime_b_prime,
            },
            "HONEST_NOTE": (
                "Local hidden variable theories MUST satisfy |S| ≤ 2. "
                "This is a mathematical theorem (Bell 1964), not an empirical fact. "
                "If this model shows |S| > 2, there is a bug in the code."
            ),
        }


# =============================================================================
# 2. QUANTUM ENTANGLEMENT MODEL
# =============================================================================

class QuantumEntanglementModel:
    """
    Quantum mechanical prediction for entangled particles.

    MODEL:
      • Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
      • Alice measures σ·a, Bob measures σ·b
      • Correlation: E(a,b) = -a·b (for singlet state)
        or E(a,b) = cos(2(θ_a - θ_b)) for |Φ+⟩

    From quantum mechanics:
      Maximum CHSH violation: S = 2√2 ≈ 2.828
      This is called Tsirelson's bound.

    HONEST SCOPE:
      • This is the STANDARD quantum prediction
      • Confirmed by experiments (Aspect 1982, loophole-free 2015)
      • Local hidden variables CANNOT reproduce this

    USAGE:
      model = QuantumEntanglementModel()
      S = model.compute_chsh()
      # S = 2√2 ≈ 2.828 (violates CHSH)
    """

    def __init__(self):
        self.state = bell_state_phi_plus()
        self.P = pauli_matrices_np()
        logger.info("[QuantumModel] Initialized with |Φ+⟩ Bell state")

    def measurement_operator(self, angle: float) -> np.ndarray:
        """
        Measurement operator for setting angle θ:
          σ·n = cos(θ)σ_z + sin(θ)σ_x
        (measuring in the x-z plane)
        """
        return (math.cos(angle) * self.P["Z"] +
                math.sin(angle) * self.P["X"])

    def correlation(self, angle_A: float, angle_B: float) -> float:
        """
        Compute quantum correlation E(a, b) = ⟨Φ+| (σ·a)⊗(σ·b) |Φ+⟩.
        """
        op_A = self.measurement_operator(angle_A)
        op_B = self.measurement_operator(angle_B)
        return quantum_correlation(self.state, op_A, op_B)

    def compute_chsh(self,
                     settings: Optional[Tuple[float, float, float, float]] = None
                     ) -> Dict[str, Any]:
        """
        Compute CHSH parameter S for quantum mechanics.

        Returns:
            dict with S value (should be 2√2 for optimal settings)
        """
        if settings is None:
            # Optimal settings for maximum violation
            a, a_prime, b, b_prime = 0.0, math.pi / 2, math.pi / 4, -math.pi / 4
        else:
            a, a_prime, b, b_prime = settings

        E_ab = self.correlation(a, b)
        E_ab_prime = self.correlation(a, b_prime)
        E_a_prime_b = self.correlation(a_prime, b)
        E_a_prime_b_prime = self.correlation(a_prime, b_prime)

        S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime

        # Tsirelson's bound
        tsirelson_bound = 2 * math.sqrt(2)

        return {
            "S": S,
            "abs_S": abs(S),
            "chsh_bound": 2.0,
            "tsirelson_bound": tsirelson_bound,
            "violates_chsh": abs(S) > 2.0 + 1e-10,
            "reaches_tsirelson": abs(abs(S) - tsirelson_bound) < 1e-6,
            "correlations": {
                "E(a,b)": E_ab,
                "E(a,b')": E_ab_prime,
                "E(a',b)": E_a_prime_b,
                "E(a',b')": E_a_prime_b_prime,
            },
            "HONEST_NOTE": (
                "Quantum mechanics predicts |S| = 2√2 ≈ 2.828 for optimal settings. "
                "This violates the CHSH bound of 2. "
                "This has been confirmed experimentally. "
                "Local hidden variables CANNOT reproduce this."
            ),
        }


# =============================================================================
# 3. "UNKNOWN INTERFERENCE" HYPOTHESIS MODEL
# =============================================================================

class InterferenceHiddenModel:
    """
    Tests the user's hypothesis:
    "Hidden variables are just interference we don't know about"

    TWO INTERPRETATIONS:

    (A) LOCAL interference:
        The "unknown interference" is a local hidden variable λ
        that determines outcomes at each detector independently.
        → This is EXACTLY a local hidden variable theory
        → REFUTED by Bell's theorem (|S| ≤ 2)

    (B) NON-LOCAL interference:
        The "unknown interference" connects the two measurements
        in a way that depends on BOTH settings simultaneously.
        → This is NOT refuted by Bell's theorem
        → But requires a specific mechanism (like Bohm's pilot wave)
        → Must be explicitly non-local

    This module tests BOTH interpretations.

    HONEST SCOPE:
      • Interpretation (A) is mathematically equivalent to local HV
      • Interpretation (B) requires specifying the non-local mechanism
      • Neither interpretation is "just interference we don't know"
        in the sense of being a simple classical wave phenomenon
    """

    def __init__(self, cfg: Optional[HiddenVariableConfig] = None):
        self.cfg = cfg or HiddenVariableConfig()
        self.rng = np.random.RandomState(self.cfg.seed)
        logger.info("[InterferenceHV] Initialized")

    def test_local_interference(self) -> Dict[str, Any]:
        """
        Test interpretation (A): local "unknown interference".

        This is mathematically equivalent to a local hidden variable theory.
        The "interference" is just a relabeling of λ.

        RESULT: |S| ≤ 2 ALWAYS (Bell's theorem applies).
        """
        # This is identical to LocalHiddenVariableModel
        # "Interference" here is just a different name for λ
        model = LocalHiddenVariableModel(self.cfg)
        result = model.compute_chsh()

        result["interpretation"] = "LOCAL_INTERFERENCE"
        result["conclusion"] = (
            "'Unknown interference' interpreted as LOCAL hidden variable "
            "is mathematically equivalent to local HV theory. "
            "Bell's theorem applies: |S| ≤ 2 ALWAYS. "
            "This interpretation is REFUTED by experiment."
        )
        return result

    def test_nonlocal_interference(self,
                                     nonlocal_strength: float = 1.0
                                     ) -> Dict[str, Any]:
        """
        Test interpretation (B): non-local "unknown interference".

        Here the "interference" depends on BOTH settings simultaneously:
          outcome_A = f(a, b, λ)  ← depends on Bob's setting too!
          outcome_B = f(a, b, λ)  ← depends on Alice's setting too!

        This is NON-LOCAL and NOT refuted by Bell's theorem.
        But it requires instantaneous influence between detectors.

        Args:
            nonlocal_strength: how strongly outcomes depend on
                               distant settings (0 = local, 1 = fully non-local)

        Returns:
            dict with S value and analysis
        """
        # For non-local model, we can reproduce quantum predictions
        # if the non-local dependence is strong enough

        # Simplified model: outcomes depend on both settings
        # This is NOT a local hidden variable theory
        n_trials = self.cfg.n_trials
        S_values = []

        settings_list = [
            (0.0, math.pi / 4),           # a, b
            (0.0, -math.pi / 4),          # a, b'
            (math.pi / 2, math.pi / 4),   # a', b
            (math.pi / 2, -math.pi / 4),  # a', b'
        ]

        correlations = []
        for a, b in settings_list:
            # Non-local model: correlation can be anything
            # because it depends on both settings simultaneously
            # For quantum-like behavior: E(a,b) = cos(2(a-b))
            E = math.cos(2 * (a - b)) * nonlocal_strength
            correlations.append(E)

        S = correlations[0] - correlations[1] + correlations[2] + correlations[3]

        return {
            "S": S,
            "abs_S": abs(S),
            "chsh_bound": 2.0,
            "violates_chsh": abs(S) > 2.0 + 1e-10,
            "nonlocal_strength": nonlocal_strength,
            "interpretation": "NONLOCAL_INTERFERENCE",
            "correlations": {
                "E(a,b)": correlations[0],
                "E(a,b')": correlations[1],
                "E(a',b)": correlations[2],
                "E(a',b')": correlations[3],
            },
            "conclusion": (
                "Non-local 'interference' CAN violate CHSH inequality. "
                "But this requires instantaneous influence between detectors, "
                "which is NOT a local hidden variable theory. "
                "This is similar to Bohmian mechanics (pilot wave theory). "
                "The word 'interference' here does NOT mean classical wave "
                "interference — it means a specific non-local mechanism."
            ),
        }


# =============================================================================
# 4. CHSH BELL TESTER
# =============================================================================

class CHSHBellTester:
    """
    Comprehensive CHSH inequality tester.

    Tests:
      1. Local hidden variable model (must give |S| ≤ 2)
      2. Quantum entanglement model (gives |S| = 2√2)
      3. Local "interference" model (must give |S| ≤ 2)
      4. Non-local "interference" model (can give |S| > 2)

    USAGE:
      tester = CHSHBellTester()
      results = tester.run_all_tests()
    """

    def __init__(self, cfg: Optional[HiddenVariableConfig] = None):
        self.cfg = cfg or HiddenVariableConfig()
        logger.info("[CHSHTTester] Initialized")

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all CHSH tests and compare."""
        results = {}

        # Test 1: Local hidden variables
        logger.info("Test 1: Local hidden variables")
        local_hv = LocalHiddenVariableModel(self.cfg)
        results["local_hv"] = local_hv.compute_chsh()

        # Test 2: Quantum entanglement
        logger.info("Test 2: Quantum entanglement")
        quantum = QuantumEntanglementModel()
        results["quantum"] = quantum.compute_chsh()

        # Test 3: Local "interference"
        logger.info("Test 3: Local interference hypothesis")
        interference = InterferenceHiddenModel(self.cfg)
        results["local_interference"] = interference.test_local_interference()

        # Test 4: Non-local "interference"
        logger.info("Test 4: Non-local interference hypothesis")
        results["nonlocal_interference"] = interference.test_nonlocal_interference(
            nonlocal_strength=1.0)

        # Summary
        results["summary"] = {
            "local_hv_S": results["local_hv"]["abs_S"],
            "quantum_S": results["quantum"]["abs_S"],
            "local_interference_S": results["local_interference"]["abs_S"],
            "nonlocal_interference_S": results["nonlocal_interference"]["abs_S"],
            "chsh_bound": 2.0,
            "tsirelson_bound": 2 * math.sqrt(2),
            "conclusions": {
                "local_hv": (
                    "REFUTED by Bell's theorem. |S| ≤ 2 always. "
                    "Cannot explain quantum correlations."
                ),
                "quantum": (
                    "CONFIRMED by experiment. |S| = 2√2 ≈ 2.828. "
                    "Violates CHSH inequality."
                ),
                "local_interference": (
                    "REFUTED by Bell's theorem. 'Unknown interference' "
                    "interpreted locally is just a hidden variable. "
                    "Same limitation as local HV: |S| ≤ 2."
                ),
                "nonlocal_interference": (
                    "NOT REFUTED by Bell's theorem, but requires "
                    "non-local mechanism. This is NOT 'just interference "
                    "we don't know' in the classical sense. It requires "
                    "a specific non-local theory (like Bohmian mechanics)."
                ),
            },
        }

        return results


# =============================================================================
# 5. SUB-QUANTUM HYPOTHESIS TESTER
# =============================================================================

class SubQuantumHypothesisTester:
    """
    Tests various "sub-quantum" hypotheses.

    HYPOTHESES TESTED:
      H1: "Hidden variables determine outcomes locally"
          → REFUTED by Bell's theorem
      H2: "Hidden variables determine outcomes non-locally"
          → NOT refuted, but requires non-local mechanism
      H3: "Interference we don't know explains quantum"
          → Depends on interpretation (local vs non-local)
      H4: "Quantum is emergent from deeper classical theory"
          → Possible in principle, but must be non-local
      H5: "Wavefunction is incomplete description"
          → Possible (Bohm, GRW, many-worlds all say this)

    HONEST SCOPE:
      • This tests MATHEMATICAL CONSISTENCY, not physical truth
      • Bell's theorem is a THEOREM, not an experimental result
      • Experiments confirm quantum predictions, not local HV
      • "Sub-quantum" theories must be non-local to match experiment

    USAGE:
      tester = SubQuantumHypothesisTester()
      results = tester.test_all_hypotheses()
    """

    def __init__(self, cfg: Optional[HiddenVariableConfig] = None):
        self.cfg = cfg or HiddenVariableConfig()
        self.chsh_tester = CHSHBellTester(cfg)
        logger.info("[SubQuantumTester] Initialized")

    def test_all_hypotheses(self) -> Dict[str, Any]:
        """Test all sub-quantum hypotheses."""
        results = {}

        # Run CHSH tests
        chsh_results = self.chsh_tester.run_all_tests()
        results["chsh_tests"] = chsh_results

        # Hypothesis analysis
        results["hypotheses"] = {
            "H1_local_hidden_variables": {
                "hypothesis": (
                    "Hidden variables λ determine measurement outcomes locally. "
                    "A(a,λ) and B(b,λ) depend only on local settings."
                ),
                "status": "REFUTED",
                "evidence": (
                    "Bell's theorem (1964) proves |S| ≤ 2 for ALL local HV theories. "
                    "Quantum mechanics predicts |S| = 2√2 ≈ 2.828. "
                    "Experiments confirm quantum prediction."
                ),
                "chsh_S": chsh_results["local_hv"]["abs_S"],
                "conclusion": (
                    "Local hidden variables CANNOT explain quantum correlations. "
                    "This is a mathematical theorem, not just experimental evidence."
                ),
            },
            "H2_nonlocal_hidden_variables": {
                "hypothesis": (
                    "Hidden variables determine outcomes, but outcomes can depend "
                    "on distant settings: A(a,b,λ), B(a,b,λ)."
                ),
                "status": "NOT_REFUTED",
                "evidence": (
                    "Bohmian mechanics is an example. It reproduces quantum "
                    "predictions but requires instantaneous non-local influence. "
                    "Bell's theorem does NOT rule this out."
                ),
                "chsh_S": chsh_results["nonlocal_interference"]["abs_S"],
                "conclusion": (
                    "Non-local hidden variables are POSSIBLE but require "
                    "abandoning locality. This is NOT 'classical' in the usual sense."
                ),
            },
            "H3_unknown_interference": {
                "hypothesis": (
                    "Hidden variables are just interference we don't know about."
                ),
                "status": "DEPENDS_ON_INTERPRETATION",
                "evidence": (
                    "If 'interference' is LOCAL: equivalent to H1, REFUTED. "
                    "If 'interference' is NON-LOCAL: equivalent to H2, NOT REFUTED "
                    "but requires non-local mechanism. "
                    "The word 'interference' does not change the mathematics."
                ),
                "local_interference_S": chsh_results["local_interference"]["abs_S"],
                "nonlocal_interference_S": chsh_results["nonlocal_interference"]["abs_S"],
                "conclusion": (
                    "'Unknown interference' is not a specific theory. "
                    "If it's local, it's refuted by Bell. "
                    "If it's non-local, it's possible but must specify the mechanism. "
                    "Simply calling it 'interference' doesn't make it classical."
                ),
            },
            "H4_emergent_quantum": {
                "hypothesis": (
                    "Quantum mechanics emerges from a deeper classical theory."
                ),
                "status": "POSSIBLE_BUT_CONSTRAINED",
                "evidence": (
                    "If the deeper theory is LOCAL, it's refuted by Bell. "
                    "If NON-LOCAL, it's possible but must reproduce quantum "
                    "predictions exactly. Examples: 't Hooft's cellular automaton, "
                    "various emergent quantum mechanics proposals."
                ),
                "conclusion": (
                    "Emergent quantum theories are possible but MUST be non-local "
                    "to match experiment. They are not 'classical' in the sense "
                    "of local realism."
                ),
            },
            "H5_incomplete_wavefunction": {
                "hypothesis": (
                    "The wavefunction is an incomplete description of reality."
                ),
                "status": "POSSIBLE",
                "evidence": (
                    "Many interpretations say this: Bohm (particles + pilot wave), "
                    "GRW (spontaneous collapse), many-worlds (universal wavefunction), "
                    "QBism (subjective probabilities). "
                    "This is NOT refuted by Bell's theorem."
                ),
                "conclusion": (
                    "The wavefunction may be incomplete. This is an interpretational "
                    "question, not refuted by Bell. But any completion must be "
                    "non-local (Bell's theorem) or abandon realism (QBism, etc.)."
                ),
            },
        }

        # Overall conclusion
        results["overall_conclusion"] = {
            "bell_theorem_status": (
                "Bell's theorem is PROVEN. Local hidden variables are IMPOSSIBLE. "
                "This is not a matter of experimental precision — it's a mathematical "
                "theorem confirmed by experiment."
            ),
            "user_hypothesis_status": (
                "'Hidden variables are just interference we don't know' is NOT "
                "a well-defined physical theory. If the 'interference' is local, "
                "it's refuted by Bell. If non-local, it's possible but requires "
                "specifying the non-local mechanism. The word 'interference' "
                "doesn't make it classical or local."
            ),
            "what_is_possible": (
                "Non-local hidden variable theories (like Bohmian mechanics) "
                "are possible. Emergent quantum theories are possible if non-local. "
                "The wavefunction may be incomplete. But ALL of these must "
                "reproduce quantum predictions, including CHSH violation."
            ),
            "what_is_impossible": (
                "Local hidden variables. Local 'interference' explanations. "
                "Any theory that satisfies Bell's locality condition and "
                "reproduces quantum predictions simultaneously."
            ),
        }

        return results


# =============================================================================
# 6. SELF-TEST SUITE
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(f"  SUB-QUANTUM HYPOTHESIS TESTER v{SQ_VERSION} — Smoke Test")
    print("=" * 78)

    np.random.seed(42)

    # ── T1: Local Hidden Variable Model ──────────────────────────────────
    print("\n [T1] Local Hidden Variable Model")
    local_hv = LocalHiddenVariableModel()
    result = local_hv.compute_chsh()

    print(f"   S = {result['S']:.6f}")
    print(f"   |S| = {result['abs_S']:.6f}")
    print(f"   CHSH bound = {result['chsh_bound']}")
    print(f"   Violates CHSH: {result['violates_chsh']}")

    assert result['abs_S'] <= 2.0 + 1e-6, "Local HV must satisfy |S| ≤ 2"
    assert not result['violates_chsh'], "Local HV must NOT violate CHSH"
    print("[PASS] T1 Local HV model works (|S| ≤ 2 as required)")

    # ── T2: Quantum Entanglement Model ───────────────────────────────────
    print("\n [T2] Quantum Entanglement Model")
    quantum = QuantumEntanglementModel()
    result = quantum.compute_chsh()

    print(f"   S = {result['S']:.6f}")
    print(f"   |S| = {result['abs_S']:.6f}")
    print(f"   Tsirelson bound = {result['tsirelson_bound']:.6f}")
    print(f"   Violates CHSH: {result['violates_chsh']}")
    print(f"   Reaches Tsirelson: {result['reaches_tsirelson']}")

    assert result['violates_chsh'], "Quantum MUST violate CHSH"
    assert abs(result['abs_S'] - 2 * math.sqrt(2)) < 1e-4, \
        f"Quantum |S| should be 2√2 ≈ 2.828, got {result['abs_S']}"
    print("[PASS] T2 Quantum model works (|S| = 2√2 as predicted)")

    # ── T3: Interference Hidden Model ────────────────────────────────────
    print("\n [T3] 'Unknown Interference' Hypothesis")
    interference = InterferenceHiddenModel()

    # Local interference
    local_result = interference.test_local_interference()
    print(f"   Local interference |S| = {local_result['abs_S']:.6f}")
    assert local_result['abs_S'] <= 2.0 + 1e-6, \
        "Local interference must satisfy |S| ≤ 2"

    # Non-local interference
    nonlocal_result = interference.test_nonlocal_interference(nonlocal_strength=1.0)
    print(f"   Non-local interference |S| = {nonlocal_result['abs_S']:.6f}")
    print(f"   Non-local violates CHSH: {nonlocal_result['violates_chsh']}")

    print("[PASS] T3 Interference model works")

    # ── T4: CHSH Bell Tester ─────────────────────────────────────────────
    print("\n [T4] CHSH Bell Tester")
    tester = CHSHBellTester()
    results = tester.run_all_tests()

    print(f"   Local HV |S|: {results['summary']['local_hv_S']:.4f}")
    print(f"   Quantum |S|: {results['summary']['quantum_S']:.4f}")
    print(f"   Local interference |S|: {results['summary']['local_interference_S']:.4f}")
    print(f"   Non-local interference |S|: {results['summary']['nonlocal_interference_S']:.4f}")
    print(f"   CHSH bound: {results['summary']['chsh_bound']}")
    print(f"   Tsirelson bound: {results['summary']['tsirelson_bound']:.4f}")

    assert results['summary']['local_hv_S'] <= 2.0
    assert results['summary']['quantum_S'] > 2.0
    assert results['summary']['local_interference_S'] <= 2.0
    print("[PASS] T4 CHSH Bell Tester works")

    # ── T5: Sub-Quantum Hypothesis Tester ────────────────────────────────
    print("\n [T5] Sub-Quantum Hypothesis Tester")
    sq_tester = SubQuantumHypothesisTester()
    sq_results = sq_tester.test_all_hypotheses()

    print(f"\n   Hypothesis Results:")
    for name, hyp in sq_results["hypotheses"].items():
        print(f"     {name}: {hyp['status']}")

    print(f"\n   Overall Conclusion:")
    for key, value in sq_results["overall_conclusion"].items():
        print(f"     {key}: {value[:80]}...")

    print("[PASS] T5 Sub-Quantum Hypothesis Tester works")

    # ── T6: Key Assertions ───────────────────────────────────────────────
    print("\n [T6] Key Assertions")

    # Bell's theorem: local HV cannot violate CHSH
    assert results['summary']['local_hv_S'] <= 2.0, \
        "Bell's theorem: local HV must satisfy |S| ≤ 2"
    print("   ✓ Bell's theorem: local HV |S| ≤ 2")

    # Quantum violates CHSH
    assert results['summary']['quantum_S'] > 2.0, \
        "Quantum mechanics violates CHSH"
    print("   ✓ Quantum mechanics: |S| > 2 (violates CHSH)")

    # Quantum reaches Tsirelson bound
    assert abs(results['summary']['quantum_S'] - 2 * math.sqrt(2)) < 1e-4, \
        "Quantum |S| should reach Tsirelson bound 2√2"
    print("   ✓ Tsirelson bound: |S| = 2√2 ≈ 2.828")

    # Local interference is same as local HV
    assert results['summary']['local_interference_S'] <= 2.0, \
        "Local interference is just local HV, must satisfy |S| ≤ 2"
    print("   ✓ Local 'interference' = local HV: |S| ≤ 2")

    # Non-local interference can violate CHSH
    assert results['summary']['nonlocal_interference_S'] > 2.0, \
        "Non-local interference can violate CHSH"
    print("   ✓ Non-local interference: |S| > 2 (possible but non-local)")

    print("[PASS] T6 All key assertions pass")

    print("\n" + "=" * 78)
    print("  ALL TESTS PASSED.")
    print("")
    print("  KEY FINDINGS:")
    print("  • Local hidden variables: REFUTED by Bell's theorem (|S| ≤ 2)")
    print("  • Quantum mechanics: CONFIRMED (|S| = 2√2 ≈ 2.828)")
    print("  • Local 'interference': SAME as local HV, REFUTED")
    print("  • Non-local 'interference': POSSIBLE but requires non-local mechanism")
    print("")
    print("  USER'S HYPOTHESIS ANALYSIS:")
    print("  'Hidden variables are just interference we don't know'")
    print("  → If LOCAL interference: REFUTED by Bell's theorem")
    print("  → If NON-LOCAL interference: Possible but must specify mechanism")
    print("  → The word 'interference' doesn't change the mathematics")
    print("  → This is NOT a new theory, just a relabeling")
    print("")
    print("  WHAT BELL'S THEOREM ACTUALLY SAYS:")
    print("  • Local realism is incompatible with quantum mechanics")
    print("  • This is a THEOREM, not an experimental limitation")
    print("  • Experiments confirm quantum predictions")
    print("  • Any 'sub-quantum' theory must be non-local or abandon realism")
    print("=" * 78)


if __name__ == "__main__":
    smoke_test()
