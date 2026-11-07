"""
=============================================================================
IBM QUANTUM TOMOGRAPHY DATA ACQUISITION + MODEL TRAINING
=============================================================================

Purpose:
  1. Connect to IBM Quantum hardware.
  2. Run single-qubit state tomography circuits in X/Y/Z bases.
  3. Reconstruct density matrices from measurement counts.
  4. Extract decoherence / coherence features.
  5. Train a neural classifier on experimental conditions.

Important:
  - This does NOT directly observe "collapse".
  - This does NOT perform continuous weak measurement on IBM hardware.
  - This performs tomography under controlled preparation/delay conditions.
  - Labels are experimental-condition labels, not metaphysical labels.

Regime labels:
  0 = coherent_reference
  1 = intermediate_dephasing
  2 = mixed_or_noise_dominated

These labels are weak labels derived from preparation/delay/purity features.
For research, replace them with calibrated labels from a known noise model.

Author:
  MSPS NETWORK / PAI & Yoon A. Limsuwan research framework
=============================================================================
"""

from __future__ import annotations

import os
import math
import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from qiskit import QuantumCircuit, ClassicalRegister, transpile

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    HAS_IBM_RUNTIME = True
except ImportError:
    HAS_IBM_RUNTIME = False

try:
    from qiskit_aer import AerSimulator
    HAS_AER = True
except ImportError:
    HAS_AER = False


# =============================================================================
# Pauli utilities
# =============================================================================

I2 = np.eye(2, dtype=np.complex128)
SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def normalize_rho(rho: np.ndarray) -> np.ndarray:
    """Project noisy reconstructed matrix to valid density matrix."""
    rho = (rho + rho.conj().T) / 2
    vals, vecs = np.linalg.eigh(rho)
    vals = np.clip(vals.real, 0, 1)
    vals = vals / max(vals.sum(), 1e-15)
    rho2 = vecs @ np.diag(vals) @ vecs.conj().T
    return rho2 / np.trace(rho2).real


def von_neumann_entropy(rho: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(rho).real
    vals = np.clip(vals, 1e-15, 1)
    return float(-np.sum(vals * np.log(vals)))


def rho_features(rho: np.ndarray, delay_steps: int = 0) -> np.ndarray:
    """
    Features for ML:
      rx, ry, rz,
      purity,
      coherence_l1,
      entropy,
      bloch_length,
      log_delay
    """
    rx = np.trace(rho @ SX).real
    ry = np.trace(rho @ SY).real
    rz = np.trace(rho @ SZ).real

    purity = np.trace(rho @ rho).real
    coherence = abs(rho[0, 1]) + abs(rho[1, 0])
    entropy = von_neumann_entropy(rho)
    bloch_len = math.sqrt(max(0.0, rx * rx + ry * ry + rz * rz))
    log_delay = math.log1p(delay_steps)

    return np.array([
        rx, ry, rz,
        purity,
        coherence,
        entropy,
        bloch_len,
        log_delay,
    ], dtype=np.float32)


# =============================================================================
# Config
# =============================================================================

@dataclass
class IBMConfig:
    token_env: str = "IBM_QUANTUM_TOKEN"
    backend_name: Optional[str] = None
    instance: Optional[str] = None
    shots: int = 2048
    optimization_level: int = 1
    use_real_hardware: bool = False
    output_dir: str = "./ibm_tomography_results"


@dataclass
class TomographyConfig:
    preparations: Tuple[str, ...] = ("plus", "zero", "one", "plus_i")
    delay_steps: Tuple[int, ...] = (0, 16, 64, 256, 1024)
    repetitions: int = 2


# =============================================================================
# IBM Connector
# =============================================================================

class IBMQuantumTomography:
    """
    Connects to IBM Quantum, runs tomography circuits, reconstructs states.
    """

    def __init__(self, cfg: IBMConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.service = None
        self.backend = None
        self.mode = "offline"

        token = os.environ.get(cfg.token_env)

        if cfg.use_real_hardware:
            if not HAS_IBM_RUNTIME:
                raise ImportError("qiskit-ibm-runtime not installed")
            if token is None:
                raise RuntimeError(f"Missing token env var: {cfg.token_env}")

            self.service = self._connect_service(token, cfg.instance)

            if cfg.backend_name:
                self.backend = self.service.backend(cfg.backend_name)
            else:
                self.backend = self.service.least_busy(
                    operational=True,
                    simulator=False,
                    min_num_qubits=1,
                )

            self.mode = "ibm_hardware"
            print(f"[IBM] Connected backend: {self.backend.name}")

        else:
            if HAS_AER:
                self.backend = AerSimulator()
                self.mode = "aer_simulator"
                print("[IBM] Using local AerSimulator")
            else:
                raise ImportError("Install qiskit-aer or enable IBM hardware")

    def _connect_service(self, token: str, instance: Optional[str]):
        """
        Support both newer and older IBM Runtime channel names.
        """
        errors = []

        for channel in ("ibm_quantum_platform", "ibm_quantum"):
            try:
                kwargs = {"channel": channel, "token": token}
                if instance:
                    kwargs["instance"] = instance
                return QiskitRuntimeService(**kwargs)
            except Exception as e:
                errors.append((channel, str(e)))

        raise RuntimeError(f"IBM service connection failed: {errors}")

    def build_tomography_circuit(
        self,
        preparation: str,
        basis: str,
        delay_steps: int = 0,
    ) -> QuantumCircuit:
        """
        Single-qubit tomography circuit.

        preparation:
          zero, one, plus, minus, plus_i, minus_i

        measurement basis:
          X, Y, Z
        """
        qc = QuantumCircuit(1)
        c = ClassicalRegister(1, "c")
        qc.add_register(c)

        # State preparation
        if preparation == "zero":
            pass
        elif preparation == "one":
            qc.x(0)
        elif preparation == "plus":
            qc.h(0)
        elif preparation == "minus":
            qc.x(0)
            qc.h(0)
        elif preparation == "plus_i":
            qc.h(0)
            qc.s(0)
        elif preparation == "minus_i":
            qc.h(0)
            qc.sdg(0)
        else:
            raise ValueError(f"Unknown preparation: {preparation}")

        # Delay to expose decoherence
        if delay_steps > 0:
            qc.delay(delay_steps, 0, unit="dt")

        # Basis rotation before Z measurement
        if basis == "X":
            qc.h(0)
        elif basis == "Y":
            qc.sdg(0)
            qc.h(0)
        elif basis == "Z":
            pass
        else:
            raise ValueError(f"Unknown basis: {basis}")

        qc.measure(0, c[0])
        return qc

    def run_circuits(self, circuits: List[QuantumCircuit]) -> List[Dict[str, int]]:
        """
        Execute circuits on IBM hardware or Aer.
        Returns list of counts dictionaries.
        """
        if self.mode == "aer_simulator":
            tqcs = transpile(
                circuits,
                backend=self.backend,
                optimization_level=self.cfg.optimization_level,
            )
            result = self.backend.run(
                tqcs,
                shots=self.cfg.shots,
            ).result()

            return [result.get_counts(i) for i in range(len(circuits))]

        elif self.mode == "ibm_hardware":
            tqcs = transpile(
                circuits,
                backend=self.backend,
                optimization_level=self.cfg.optimization_level,
            )

            sampler = SamplerV2(mode=self.backend)
            job = sampler.run(tqcs, shots=self.cfg.shots)
            result = job.result()

            all_counts = []

            for pub_result in result:
                data = pub_result.data

                # Most circuits use classical register "c"
                if hasattr(data, "c"):
                    counts = data.c.get_counts()
                elif hasattr(data, "meas"):
                    counts = data.meas.get_counts()
                else:
                    # Fallback: inspect first register-like field
                    found = False
                    counts = {}
                    for name in dir(data):
                        obj = getattr(data, name)
                        if hasattr(obj, "get_counts"):
                            counts = obj.get_counts()
                            found = True
                            break
                    if not found:
                        raise RuntimeError("Cannot extract counts from SamplerV2 result")

                all_counts.append(counts)

            return all_counts

        else:
            raise RuntimeError(f"Unknown mode: {self.mode}")

    @staticmethod
    def expectation_from_counts(counts: Dict[str, int]) -> float:
        """
        Single qubit Z expectation from counts.
        P(0) - P(1)
        """
        n0 = counts.get("0", 0)
        n1 = counts.get("1", 0)
        total = n0 + n1
        if total == 0:
            return 0.0
        return (n0 - n1) / total

    def reconstruct_rho(self, exp_x: float, exp_y: float, exp_z: float) -> np.ndarray:
        """
        rho = 1/2 (I + rx X + ry Y + rz Z)
        """
        rho = 0.5 * (I2 + exp_x * SX + exp_y * SY + exp_z * SZ)
        return normalize_rho(rho)

    def collect_dataset(self, tcfg: TomographyConfig) -> pd.DataFrame:
        """
        Collect tomography dataset over preparations, delays, repetitions.
        """
        rows = []

        for rep in range(tcfg.repetitions):
            for prep in tcfg.preparations:
                for delay in tcfg.delay_steps:
                    circuits = []
                    bases = ["X", "Y", "Z"]

                    for basis in bases:
                        circuits.append(
                            self.build_tomography_circuit(
                                preparation=prep,
                                basis=basis,
                                delay_steps=delay,
                            )
                        )

                    counts_list = self.run_circuits(circuits)

                    exps = {
                        basis: self.expectation_from_counts(counts)
                        for basis, counts in zip(bases, counts_list)
                    }

                    rho = self.reconstruct_rho(
                        exps["X"], exps["Y"], exps["Z"]
                    )

                    feats = rho_features(rho, delay_steps=delay)

                    row = {
                        "rep": rep,
                        "preparation": prep,
                        "delay_steps": delay,
                        "exp_x": exps["X"],
                        "exp_y": exps["Y"],
                        "exp_z": exps["Z"],
                        "rho_00_real": rho[0, 0].real,
                        "rho_11_real": rho[1, 1].real,
                        "rho_01_real": rho[0, 1].real,
                        "rho_01_imag": rho[0, 1].imag,
                        "purity": float(feats[3]),
                        "coherence_l1": float(feats[4]),
                        "entropy": float(feats[5]),
                        "bloch_length": float(feats[6]),
                        "mode": self.mode,
                        "backend": getattr(self.backend, "name", str(self.backend)),
                    }

                    # Weak labels for model training.
                    # Replace with calibrated labels for serious research.
                    #
                    # 0 coherent_reference
                    # 1 intermediate_dephasing
                    # 2 mixed_or_noise_dominated
                    if row["purity"] > 0.90 and row["coherence_l1"] > 0.50:
                        label = 0
                    elif row["purity"] > 0.60:
                        label = 1
                    else:
                        label = 2

                    row["label"] = label

                    for i, name in enumerate([
                        "rx", "ry", "rz", "purity_f",
                        "coherence_f", "entropy_f",
                        "bloch_f", "log_delay"
                    ]):
                        row[name] = float(feats[i])

                    rows.append(row)

                    print(
                        f"[DATA] rep={rep} prep={prep} delay={delay} "
                        f"purity={row['purity']:.3f} label={label}"
                    )

        df = pd.DataFrame(rows)
        path = self.output_dir / "tomography_dataset.csv"
        df.to_csv(path, index=False)
        print(f"[DATA] Saved dataset: {path}")
        return df


# =============================================================================
# ML Model
# =============================================================================

class DecoherenceClassifier(nn.Module):
    """
    Classify tomography feature vector into experimental regimes.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_model_from_dataframe(
    df: pd.DataFrame,
    output_dir: str = "./ibm_tomography_results",
    epochs: int = 100,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """
    Train model from tomography dataframe.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    feature_cols = [
        "rx", "ry", "rz",
        "purity_f",
        "coherence_f",
        "entropy_f",
        "bloch_f",
        "log_delay",
    ]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xn = (X - mean) / std

    # Simple split
    rng = np.random.default_rng(42)
    idx = np.arange(len(Xn))
    rng.shuffle(idx)

    n_train = max(1, int(0.8 * len(idx)))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:] if len(idx) > n_train else idx[:1]

    X_train = torch.tensor(Xn[train_idx])
    y_train = torch.tensor(y[train_idx])
    X_test = torch.tensor(Xn[test_idx])
    y_test = torch.tensor(y[test_idx])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=min(32, len(X_train)),
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoherenceClassifier(input_dim=X.shape[1]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(xb)
            pred = logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += len(xb)

        train_acc = correct / max(total, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                logits = model(X_test.to(device))
                pred = logits.argmax(dim=-1).cpu()
                test_acc = (pred == y_test).float().mean().item()

            row = {
                "epoch": epoch + 1,
                "loss": total_loss / max(total, 1),
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
            history.append(row)
            print(row)

    torch.save(model.state_dict(), out / "decoherence_classifier.pt")

    norm = {"mean": mean.tolist(), "std": std.tolist(), "feature_cols": feature_cols}
    with open(out / "feature_normalization.json", "w") as f:
        json.dump(norm, f, indent=2)

    with open(out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "model_path": str(out / "decoherence_classifier.pt"),
        "normalization_path": str(out / "feature_normalization.json"),
        "history": history,
        "HONEST_NOTE": (
            "Labels are weak experimental labels derived from tomography features. "
            "They are not direct observations of wavefunction collapse."
        ),
    }


# =============================================================================
# Inference helper
# =============================================================================

def predict_regime(
    rho: np.ndarray,
    delay_steps: int,
    model_dir: str = "./ibm_tomography_results",
) -> Dict[str, Any]:
    """
    Load trained model and predict regime from reconstructed rho.
    """
    model_dir = Path(model_dir)

    with open(model_dir / "feature_normalization.json") as f:
        norm = json.load(f)

    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)

    feats = rho_features(rho, delay_steps)
    x = (feats - mean) / std

    model = DecoherenceClassifier(input_dim=len(feats))
    model.load_state_dict(torch.load(model_dir / "decoherence_classifier.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(x).unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)[0].numpy()

    labels = [
        "coherent_reference",
        "intermediate_dephasing",
        "mixed_or_noise_dominated",
    ]

    k = int(np.argmax(probs))

    return {
        "predicted_regime": labels[k],
        "confidence": float(probs[k]),
        "probabilities": {labels[i]: float(probs[i]) for i in range(3)},
        "features": {
            norm["feature_cols"][i]: float(feats[i])
            for i in range(len(feats))
        },
    }


# =============================================================================
# Main experiment
# =============================================================================

def main():
    ibm_cfg = IBMConfig(
        backend_name=os.environ.get("IBM_BACKEND", None),
        use_real_hardware=os.environ.get("USE_IBM_HARDWARE", "0") == "1",
        shots=int(os.environ.get("IBM_SHOTS", "2048")),
    )

    tomo_cfg = TomographyConfig(
        preparations=("plus", "zero", "one", "plus_i"),
        delay_steps=(0, 16, 64, 256, 1024),
        repetitions=2,
    )

    collector = IBMQuantumTomography(ibm_cfg)
    df = collector.collect_dataset(tomo_cfg)

    train_result = train_model_from_dataframe(
        df,
        output_dir=ibm_cfg.output_dir,
        epochs=100,
    )

    print("\n[TRAIN RESULT]")
    print(json.dumps(train_result, indent=2))

    # Example inference from last reconstructed rho
    last = df.iloc[-1]
    rho = np.array([
        [last["rho_00_real"], last["rho_01_real"] + 1j * last["rho_01_imag"]],
        [last["rho_01_real"] - 1j * last["rho_01_imag"], last["rho_11_real"]],
    ], dtype=np.complex128)

    pred = predict_regime(
        rho,
        delay_steps=int(last["delay_steps"]),
        model_dir=ibm_cfg.output_dir,
    )

    print("\n[PREDICTION EXAMPLE]")
    print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()
