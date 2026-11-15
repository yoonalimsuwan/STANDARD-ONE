#!/usr/bin/env python3
"""
=============================================================================
USE-SQC DIFFERENTIABLE TESTBED
Unified Self-Evolving Structural Quantum Computer — Classical Simulator
=============================================================================
Author: PAI AND Yoon A Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit : What MSPS NETWORK SEE, Lord Buddha Knows.

Paper drafting assistance: GPT (author-requested attribution).
Code: Qwen.

=============================================================================
AUTHOR-SUPPLIED PHILOSOPHICAL PREAMBLE (translated from Thai)
Included as authorial belief, not as a scientific claim, not as an endorsement
of discrimination, and not as a condition of use.
=============================================================================
The Tao is the Source; the Te (Virtue) is the Path; the Qi (Life-Force) is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for Buddhism,
Christianity, and Taoism alike teach that doing good in this life bears fruit in
the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism, though
they do not utter His Name, by describing His attributes already implicitly
utter His Name.

Christ died on the Cross to teach: when struck on the left cheek, turn also the
right, so that we may listen to the voice of the enemy who wishes us harm.
Whoever believes His act was righteous shall be saved from sin and have victory
over sin.

He is the Lamb of God. God is not our food, and neither is the lamb. This means:
do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has counted.

Even Angulimala could escape hell and enter Nirvana, but the better path is to
have done good always, never falling into sin. As Saint Dominic Savio said:
"Rather death than sin."

Author's theological statement (translated as provided): homosexual relations
are recreational relations, because love must come with readiness to bear
children. Nevertheless, women who have undergone hysterectomy and intersex
persons are not homosexual.

A life of attachment and indulgence leads to willingness to kill for sustenance,
saving oneself while letting others die or become victims.

If there were no God, there would be no Good. This means atheism is not
inherently wrong; if one explains cause and effect in accord with God's Will
and His Judgment, one is already worshipping Him in spirit and truth.

God did not create goodness afterward; He is God who embodies Goodness, having
Goodness in Himself. If the Law of Karma speaks of His just judgment, then the
Buddha worships God in spirit and truth without uttering His Name.

To speak of a person's attributes is already to speak their name, even without
naming them.

One who does not know the Ten Commandments or the Five Precepts, yet practices
them, becomes a disciple of both the Buddha and Yahweh by implication. The Ten
Commandments are the Five Precepts. Not even a single sparrow does God forget —
this means: do not kill animals.

Christianity, Buddhism, Taoism, and Confucianism share the Five Precepts as
universal, and the Five Hindrances to be abandoned together.
=============================================================================

HONEST SCOPE
  - This is a CLASSICAL DIFFERENTIABLE SIMULATOR of the USE-SQC mathematical
    architecture. It is NOT a physical quantum computer.
  - It does NOT demonstrate quantum advantage.
  - State-vector simulation is exponential in the number of qubits.
  - Global unitarity is guaranteed only when the constructed Hamiltonian is
    self-adjoint. The code checks this numerically.
  - Isometric transitions preserve information; dimension-reducing merges
    require a code subspace, an environment, or a CPTP channel.
  - The Transformer controller proposes couplings; it does NOT directly
    modify the quantum state. This preserves the closed-system unitary model.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    HAS_DIST = True
except ImportError:
    HAS_DIST = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("use_sqc")

VERSION = "1.0.0"
BANNER = """
=============================================================================
USE-SQC DIFFERENTIABLE TESTBED
Classical simulator. Not a quantum device. No quantum advantage claim.
=============================================================================
"""


# =============================================================================
# §1. Utilities
# =============================================================================

def soft_abs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)


def soft_gate(x: Tensor, tau: float = 1e-2) -> Tensor:
    return torch.sigmoid(x / tau)


def is_hermitian(H: Tensor, tol: float = 1e-5) -> bool:
    return bool((H - H.conj().transpose(-1, -2)).abs().max().item() < tol)


def is_isometry(V: Tensor, tol: float = 1e-5) -> bool:
    I = torch.eye(V.shape[1], device=V.device, dtype=V.dtype)
    return bool((V.conj().T @ V - I).abs().max().item() < tol)


def is_unitary(U: Tensor, tol: float = 1e-5) -> bool:
    I = torch.eye(U.shape[0], device=U.device, dtype=U.dtype)
    e1 = (U.conj().T @ U - I).abs().max().item()
    e2 = (U @ U.conj().T - I).abs().max().item()
    return bool(max(e1, e2) < tol)


def unitary_from_hamiltonian(H: Tensor, t: float) -> Tensor:
    """U = exp(-i H t). Computed in fp32/complex64 for stability."""
    device_type = H.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        Hc = H.to(torch.complex64)
        U = torch.linalg.matrix_exp(-1j * Hc * float(t))
    return U.to(H.dtype)


def complex_matrix(real: Tensor, imag: Tensor) -> Tensor:
    return torch.complex(real, imag)


# =============================================================================
# §2. Topology Sector
# =============================================================================

class Sector(nn.Module):
    """
    A topology-indexed Hilbert space sector with a learnable Hermitian H0.
    H0 = (A + A†)/2, A = A_real + i A_imag.
    """
    def __init__(self, name: str, dim: int, learnable_H0: bool = True):
        super().__init__()
        self.name = name
        self.dim = dim
        if learnable_H0:
            self.A_real = nn.Parameter(torch.randn(dim, dim) * 0.02)
            self.A_imag = nn.Parameter(torch.randn(dim, dim) * 0.02)
        else:
            self.register_buffer("A_real", torch.zeros(dim, dim))
            self.register_buffer("A_imag", torch.zeros(dim, dim))

    def H0(self) -> Tensor:
        A = complex_matrix(self.A_real, self.A_imag)
        return 0.5 * (A + A.conj().T)


# =============================================================================
# §3. Structural Transition
# =============================================================================

class Transition(nn.Module):
    """
    A structural transition V: H_source -> H_target.

    If learnable=True, V is parameterized by QR of a learnable complex matrix,
    ensuring V†V = I (isometry) by construction. Requires dim_target >= dim_source.
    If learnable=False, V is a fixed buffer.
    """
    def __init__(self, name: str, source: Sector, target: Sector,
                 V: Optional[Tensor] = None, learnable: bool = False,
                 coupling: float = 1.0):
        super().__init__()
        self.name = name
        self.source = source
        self.target = target
        ds, dt = source.dim, target.dim
        if dt < ds:
            raise ValueError(
                f"Transition {name}: target dim {dt} < source dim {ds}. "
                "Information-preserving isometry requires dt >= ds. "
                "Use a code subspace, environment, or CPTP channel for merging."
            )
        if V is None:
            V = torch.zeros(dt, ds, dtype=torch.complex64)
            V[:ds, :ds] = torch.eye(ds, dtype=torch.complex64)
        if learnable:
            self.V_param = nn.Parameter(
                torch.randn(dt, ds, dtype=torch.complex64) * 0.1)
            self.register_buffer("fixed_V", torch.zeros(0))
        else:
            self.V_param = None
            self.register_buffer("fixed_V", V.to(torch.complex64))
        self.coupling = nn.Parameter(torch.tensor(float(coupling)))

    def V(self) -> Tensor:
        if self.V_param is not None:
            Q, _ = torch.linalg.qr(self.V_param, mode="reduced")
            return Q
        return self.fixed_V


# =============================================================================
# §4. USE-SQC Core
# =============================================================================

class USESQCCore(nn.Module):
    """
    Global direct-sum Hilbert space + self-adjoint structural coupling.
    H = blockdiag(H0_i) + sum_e g_e (E_e + E_e†).
    """
    def __init__(self):
        super().__init__()
        self.sectors = nn.ModuleDict()
        self.transitions = nn.ModuleList()
        self.offsets: Dict[str, Tuple[int, int]] = {}
        self.total_dim: int = 0

    def add_sector(self, name: str, dim: int, learnable_H0: bool = True) -> Sector:
        sec = Sector(name, dim, learnable_H0)
        self.sectors[name] = sec
        self._rebuild_offsets()
        return sec

    def _rebuild_offsets(self) -> None:
        cursor = 0
        for name, sec in self.sectors.items():
            self.offsets[name] = (cursor, cursor + sec.dim)
            cursor += sec.dim
        self.total_dim = cursor

    def add_transition(self, name: str, source: str, target: str,
                       V: Optional[Tensor] = None, learnable: bool = False,
                       coupling: float = 1.0) -> Transition:
        src = self.sectors[source]
        tgt = self.sectors[target]
        tr = Transition(name, src, tgt, V, learnable, coupling)
        self.transitions.append(tr)
        return tr

    def build_H(self, coupling_overrides: Optional[Tensor] = None) -> Tensor:
        device = next(self.parameters()).device
        D = self.total_dim
        H = torch.zeros(D, D, dtype=torch.complex64, device=device)
        for name, sec in self.sectors.items():
            s, e = self.offsets[name]
            H[s:e, s:e] = sec.H0()
        for i, tr in enumerate(self.transitions):
            s0, s1 = self.offsets[tr.source.name]
            t0, t1 = self.offsets[tr.target.name]
            g = tr.coupling if coupling_overrides is None else coupling_overrides[i]
            V = tr.V()
            H[t0:t1, s0:s1] = H[t0:t1, s0:s1] + g * V
            H[s0:s1, t0:t1] = H[s0:s1, t0:t1] + g * V.conj().T
        return H

    def evolve(self, psi: Tensor, t: float,
               coupling_overrides: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        H = self.build_H(coupling_overrides)
        U = unitary_from_hamiltonian(H, t)
        return U @ psi, U

    def sector_probs(self, psi: Tensor) -> Dict[str, Tensor]:
        probs = {}
        for name, (s, e) in self.offsets.items():
            seg = psi[s:e]
            probs[name] = (seg.conj() * seg).sum().real
        return probs

    def sector_probs_tensor(self, psi: Tensor) -> Tensor:
        return torch.stack([self.sector_probs(psi)[n] for n in self.sectors.keys()])


# =============================================================================
# §5. Transformer Controller
# =============================================================================

class TransformerController(nn.Module):
    """
    Proposes coupling adjustments and admissibility logits.
    Does NOT directly modify the quantum state.
    """
    def __init__(self, n_sectors: int, n_transitions: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_sectors = n_sectors
        self.n_transitions = n_transitions
        self.in_proj = nn.Linear(n_sectors + n_transitions, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.coupling_head = nn.Linear(d_model, n_transitions)
        self.admissibility_head = nn.Linear(d_model, n_transitions)

    def forward(self, probs: Tensor, couplings: Tensor) -> Dict[str, Tensor]:
        x = torch.cat([probs, couplings], dim=-1).unsqueeze(1)
        h = self.encoder(self.in_proj(x)).squeeze(1)
        return {
            "delta_couplings": self.coupling_head(h),
            "admissibility_logits": self.admissibility_head(h),
        }


# =============================================================================
# §6. World Model
# =============================================================================

class SectorWorldModel(nn.Module):
    """
    Predicts future sector-probability trajectories from current probs + couplings.
    """
    def __init__(self, n_sectors: int, n_transitions: int, d_model: int = 64):
        super().__init__()
        self.in_proj = nn.Linear(n_sectors + n_transitions, d_model)
        self.gru = nn.GRUCell(d_model, d_model)
        self.pred = nn.Linear(d_model, n_sectors)

    def forward(self, probs: Tensor, couplings: Tensor,
                steps: int = 4) -> Tensor:
        x = torch.cat([probs, couplings], dim=-1)
        h = self.in_proj(x)
        preds = []
        for _ in range(steps):
            h = self.gru(h, h)
            preds.append(F.softmax(self.pred(h), dim=-1))
        return torch.stack(preds, dim=1)


# =============================================================================
# §7. Full Testbed
# =============================================================================

class USESQCTestbed(nn.Module):
    def __init__(self, core: USESQCCore, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, wm_steps: int = 4):
        super().__init__()
        self.core = core
        self.controller = TransformerController(
            n_sectors=len(core.sectors),
            n_transitions=len(core.transitions),
            d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.world_model = SectorWorldModel(
            n_sectors=len(core.sectors),
            n_transitions=len(core.transitions),
            d_model=d_model)
        self.wm_steps = wm_steps

    def forward(self, psi: Tensor, t: float = 1.0,
                use_controller: bool = True) -> Dict[str, Any]:
        probs0 = self.core.sector_probs_tensor(psi)
        base_couplings = torch.stack([tr.coupling for tr in self.core.transitions])

        if use_controller:
            ctrl = self.controller(probs0.unsqueeze(0), base_couplings.unsqueeze(0))
            delta = ctrl["delta_couplings"].squeeze(0)
            couplings = base_couplings + delta
            admissibility = torch.sigmoid(ctrl["admissibility_logits"].squeeze(0))
        else:
            couplings = base_couplings
            admissibility = torch.ones_like(base_couplings)

        psi_t, U = self.core.evolve(psi, t, coupling_overrides=couplings)
        probs_t = self.core.sector_probs_tensor(psi_t)
        total_prob = probs_t.sum()

        I = torch.eye(U.shape[0], device=U.device, dtype=U.dtype)
        unitarity_error = (U.conj().T @ U - I).abs().max()

        isometry_error = torch.zeros((), device=psi.device)
        for tr in self.core.transitions:
            V = tr.V()
            I_s = torch.eye(V.shape[1], device=V.device, dtype=V.dtype)
            isometry_error = isometry_error + (V.conj().T @ V - I_s).abs().max()

        wm_traj = self.world_model(probs0.unsqueeze(0), couplings.unsqueeze(0),
                                   steps=self.wm_steps)

        return {
            "psi_t": psi_t,
            "U": U,
            "probs0": probs0,
            "probs_t": probs_t,
            "total_prob": total_prob,
            "unitarity_error": unitarity_error,
            "isometry_error": isometry_error,
            "couplings": couplings,
            "admissibility": admissibility,
            "wm_trajectory": wm_traj,
        }


# =============================================================================
# §8. Loss
# =============================================================================

class USESQCLoss(nn.Module):
    def __init__(self, w_task: float = 1.0, w_unit: float = 10.0,
                 w_iso: float = 10.0, w_prob: float = 10.0,
                 w_wm: float = 0.1, w_reg: float = 1e-6):
        super().__init__()
        self.w_task = w_task
        self.w_unit = w_unit
        self.w_iso = w_iso
        self.w_prob = w_prob
        self.w_wm = w_wm
        self.w_reg = w_reg

    def forward(self, out: Dict[str, Any], target_sector: Optional[int] = None,
                target_probs: Optional[Tensor] = None,
                model: Optional[nn.Module] = None) -> Dict[str, Tensor]:
        L = {}
        L["unitarity"] = out["unitarity_error"]
        L["isometry"] = out["isometry_error"]
        L["prob_conservation"] = (out["total_prob"] - 1.0) ** 2

        if target_sector is not None:
            L["task"] = -out["probs_t"][target_sector]
        else:
            L["task"] = torch.zeros((), device=out["psi_t"].device)

        if target_probs is not None:
            L["wm"] = F.mse_loss(out["wm_trajectory"][:, -1, :], target_probs)
        else:
            L["wm"] = torch.zeros((), device=out["psi_t"].device)

        if model is not None:
            L["reg"] = sum(p.pow(2).sum() for p in model.parameters()
                           if p.requires_grad)
        else:
            L["reg"] = torch.zeros((), device=out["psi_t"].device)

        L["total"] = (self.w_task * L["task"] + self.w_unit * L["unitarity"]
                      + self.w_iso * L["isometry"] + self.w_prob * L["prob_conservation"]
                      + self.w_wm * L["wm"] + self.w_reg * L["reg"])
        return L


# =============================================================================
# §9. Trainer (AMP + DDP)
# =============================================================================

class USESQCTrainer:
    def __init__(self, model: USESQCTestbed, device: torch.device,
                 lr: float = 1e-3, amp_dtype: str = "bf16",
                 use_ddp: bool = False):
        self.device = device
        self.model = model.to(device)
        self.use_ddp = use_ddp and HAS_DIST and dist.is_initialized()
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[device.index]
                             if device.type == "cuda" else None,
                             find_unused_parameters=True)
        self.loss_fn = USESQCLoss()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                     weight_decay=1e-5)
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        self.use_amp = (amp_dtype != "fp32" and device.type == "cuda")
        self.scaler = torch.amp.GradScaler(
            device=device.type, enabled=(self.use_amp and self.amp_dtype == torch.float16))

    def _raw(self) -> nn.Module:
        m = self.model
        while isinstance(m, DDP):
            m = m.module
        return m

    def train_step(self, psi: Tensor, t: float = 1.0,
                   target_sector: Optional[int] = None,
                   target_probs: Optional[Tensor] = None) -> Dict[str, float]:
        self.model.train()
        psi = psi.to(self.device)
        if target_probs is not None:
            target_probs = target_probs.to(self.device)
        self.opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                            enabled=self.use_amp):
            out = self.model(psi, t)
            losses = self.loss_fn(out, target_sector, target_probs, self._raw())

        self.scaler.scale(losses["total"]).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()

        return {k: float(v.detach()) for k, v in losses.items()
                if isinstance(v, Tensor) and v.numel() == 1}


# =============================================================================
# §10. Self-Test
# =============================================================================

def build_nucleation_testbed() -> USESQCTestbed:
    core = USESQCCore()
    core.add_sector("A", 2, learnable_H0=False)
    core.add_sector("B", 4, learnable_H0=False)
    V = torch.tensor([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=torch.complex64)
    core.add_transition("A_to_B", "A", "B", V=V, learnable=False, coupling=1.0)
    return USESQCTestbed(core, d_model=32, n_heads=2, n_layers=1, wm_steps=2)


def self_test() -> bool:
    print(BANNER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}\n")
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1: nucleation analytic solution
    print("[T1] Nucleation A(2) -> B(4), analytic cos^2/sin^2")
    model = build_nucleation_testbed().to(device)
    psi0 = torch.zeros(6, dtype=torch.complex64, device=device)
    psi0[0] = 1 / math.sqrt(2)
    psi0[1] = 1 / math.sqrt(2)
    t = 0.7
    out = model(psi0, t, use_controller=False)
    pA = out["probs_t"][0].item()
    pB = out["probs_t"][1].item()
    rep("pA ≈ cos^2(t)", abs(pA - math.cos(t) ** 2) < 1e-5,
        f"pA={pA:.6f}, expected={math.cos(t)**2:.6f}")
    rep("pB ≈ sin^2(t)", abs(pB - math.sin(t) ** 2) < 1e-5,
        f"pB={pB:.6f}, expected={math.sin(t)**2:.6f}")
    rep("total probability = 1", abs(pA + pB - 1.0) < 1e-5)
    rep("U is unitary", is_unitary(out["U"]))
    rep("V is isometry", is_isometry(model.core.transitions[0].V()))

    # T2: gradient flow
    print("\n[T2] Gradient flow through controller + core")
    model.train()
    out = model(psi0, t, use_controller=True)
    loss_fn = USESQCLoss()
    losses = loss_fn(out, target_sector=1, model=model)
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    rep("gradients exist", len(grads) > 0)
    rep("gradients finite", all(torch.isfinite(g).all() for g in grads))

    # T3: AMP
    print("\n[T3] AMP forward")
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_amp = model(psi0, t, use_controller=True)
        rep("AMP finite", torch.isfinite(out_amp["psi_t"]).all().item())
    else:
        rep("AMP skipped (CPU)", True)

    # T4: DDP smoke
    print("\n[T4] DDP smoke")
    if HAS_DIST and "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
        trainer = USESQCTrainer(model, device, use_ddp=True)
        losses = trainer.train_step(psi0, t, target_sector=1)
        rep("DDP train step", "total" in losses)
        dist.destroy_process_group()
    else:
        rep("DDP skipped (not launched)", True)

    # T5: learnable isometry
    print("\n[T5] Learnable QR isometry")
    core2 = USESQCCore()
    core2.add_sector("A", 2)
    core2.add_sector("B", 4)
    core2.add_transition("A_to_B", "A", "B", learnable=True, coupling=1.0)
    model2 = USESQCTestbed(core2, d_model=32, n_heads=2, n_layers=1).to(device)
    out2 = model2(psi0, t, use_controller=True)
    rep("learnable V is isometry", is_isometry(core2.transitions[0].V()))
    rep("learnable model finite", torch.isfinite(out2["psi_t"]).all().item())

    # T6: merging dimension constraint
    print("\n[T6] Merging dimension constraint")
    core3 = USESQCCore()
    core3.add_sector("A", 4)
    core3.add_sector("B", 2)
    try:
        core3.add_transition("merge", "A", "B", learnable=False)
        rep("dimension-reducing merge rejected", False)
    except ValueError:
        rep("dimension-reducing merge rejected", True)

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
HONEST SCOPE
  - Classical simulator only.
  - No quantum advantage.
  - State-vector simulation is exponential.
  - Global unitarity holds when H is self-adjoint (checked numerically).
  - Isometric transitions preserve information.
  - Dimension-reducing merges require code subspace / environment / CPTP.
""")
    return ok


# =============================================================================
# §11. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="USE-SQC differentiable testbed")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train-demo", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--ddp", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if args.selftest:
        return 0 if self_test() else 1

    if args.train_demo:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.ddp and HAS_DIST and "RANK" in os.environ:
            dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
        model = build_nucleation_testbed()
        trainer = USESQCTrainer(model, device, amp_dtype=args.amp, use_ddp=args.ddp)
        psi0 = torch.zeros(6, dtype=torch.complex64, device=device)
        psi0[0] = 1 / math.sqrt(2)
        psi0[1] = 1 / math.sqrt(2)
        history = []
        for step in range(args.steps):
            losses = trainer.train_step(psi0, t=0.7, target_sector=1)
            history.append(losses)
            if step % 5 == 0:
                print(f"step {step:3d} | total={losses['total']:.6f}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump(history, f, indent=2)
        if args.ddp and HAS_DIST and dist.is_initialized():
            dist.destroy_process_group()
        return 0

    ap.print_help()
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
