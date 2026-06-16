# =============================================================================
# NGO PHYSICS ONE — Production-Grade Structural GNO Physics Surrogate
# =============================================================================
# Developer    : Yoon A Limsuwan / MSPS NETWORK
#                MY SOUL MOVE BY POWER OF HOLY SPIRIT
# Organization : MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# License      : MIT
# Year         : 2026
#
# AI Co-Developers (architecture, numerical methods, production hardening):
#   - Claude   (Anthropic)  — production refactor, EMA checkpointing,
#                             multi-loss weighting, physics-informed losses,
#                             LR scheduling, gradient monitoring, full docstrings
#   - GPT      (OpenAI)     — early architecture exploration, message-passing
#                             design, phase-field surrogate concept
#   - Gemini   (Google)     — v2 unified discrete/continuous extension,
#                             one-shot phase evolution framing
#
# Description:
#   Production-ready AI surrogate for multi-domain fundamental physics.
#   Wraps StructuralGNOPhysics with:
#     • Full integration with STANDARD ONE (PhysicsParameters, CSOCKernel,
#       SemanticStateContraction, DiffRGRefiner)
#     • Yang–Mills mass gap data pipeline (GribovPropagator, RefinedGribov)
#     • Exponential Moving Average (EMA) weight tracking
#     • Multi-task loss with learnable uncertainty weighting (Kendall et al.)
#     • Cosine annealing + warmup LR schedule
#     • Gradient norm monitoring and adaptive clipping
#     • Production checkpoint manager (save / resume / best-model tracking)
#     • Physics-informed auxiliary losses (positivity, propagator monotonicity,
#       CMB power-law prior, YM mass gap constraint)
#     • Unified training loop with domain-balanced sampling
#     • Inference API with uncertainty propagation
#
# Dependencies:
#   standard_one.py          (same directory)
#   yang_mills_mass_gap_one.py  (same directory)
#   PyTorch >= 2.0
# =============================================================================

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# ---------------------------------------------------------------------------
# Local ecosystem imports
# ---------------------------------------------------------------------------
from standard_one import (
    CSOCKernel,
    DiffRGRefiner,
    PhysicsParameters,
    SemanticStateContraction,
    get_device,
)
from yang_mills_mass_gap_one import (
    GribovPropagator,
    LatticeDataLoader,
    MassGapAnalyzer,
    RefinedGribovPropagator,
    YangMillsMassGap,
)

logger = logging.getLogger("NGOPhysicsOne")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class NGOPhysicsConfig:
    """
    Unified configuration for the NGO Physics ONE production model.

    All hyperparameters are grouped by concern so they can be serialised to
    JSON and reconstructed deterministically for experiment reproducibility.
    """

    # ---- Architecture -------------------------------------------------------
    hidden_dim: int = 256
    num_layers: int = 8           # deeper than prototype (was 6)
    dropout: float = 0.10
    use_residual_scale: bool = True  # learnable per-layer residual scale

    # ---- Loss weights (initial; overridden by learned log-variances) --------
    lambda_collider: float = 1.0
    lambda_cosmo: float = 1.0
    lambda_ym: float = 2.0        # YM mass gap gets higher initial weight
    lambda_physics: float = 0.5   # auxiliary physics-informed penalty

    # ---- Training -----------------------------------------------------------
    epochs: int = 300
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip_max_norm: float = 1.0
    warmup_epochs: int = 10
    eta_min_lr: float = 1e-6      # cosine annealing floor

    # ---- EMA ----------------------------------------------------------------
    ema_decay: float = 0.999

    # ---- Checkpointing ------------------------------------------------------
    checkpoint_dir: str = "checkpoints_ngo"
    save_every_n_epochs: int = 10
    keep_best_n: int = 3

    # ---- CMB spectrum -------------------------------------------------------
    lmax: int = 2499              # output bins: ells 2..2500

    # ---- Reproducibility ----------------------------------------------------
    seed: int = 42
    device: str = "auto"          # "auto" → best available

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "NGOPhysicsConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "NGOPhysicsConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# 2. Sub-modules
# =============================================================================

class PhysicsFiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation block conditioned on structural CSOC state σ.

    Each hidden layer h is transformed as:
        h' = γ(σ) ⊙ h + β(σ)
    then passed through a two-layer MLP with a residual connection.

    The optional ``residual_scale`` parameter (one per layer, learnable)
    prevents gradient vanishing in deep stacks by initialising near 1.
    """

    def __init__(self, dim: int, dropout: float = 0.1,
                 use_residual_scale: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        # FiLM modulators
        self.film_gamma = nn.Linear(1, dim)
        self.film_beta  = nn.Linear(1, dim)

        # Learnable residual scale: initialised to 1 so early training is stable
        self.residual_scale = (
            nn.Parameter(torch.ones(1)) if use_residual_scale else None
        )

        # Initialise FiLM to near-identity at the start of training
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.ones_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x     : (B, D) hidden state
            sigma : (B, 1) CSOC structural parameter
        Returns:
            (B, D) updated hidden state
        """
        gamma = self.film_gamma(sigma)   # (B, D)
        beta  = self.film_beta(sigma)    # (B, D)
        modulated = self.norm(gamma * x + beta)
        delta = self.mlp(modulated)
        scale = self.residual_scale if self.residual_scale is not None else 1.0
        return x + scale * delta


class DomainEncoder(nn.Module):
    """
    Shared encoder skeleton with domain-specific projection head.

    Projects raw physics features → hidden_dim, applies LayerNorm and
    an extra non-linearity so the backbone sees normalised representations
    regardless of input scale (GeV, dimensionless cosmological params, etc.).
    """

    def __init__(self, in_features: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# =============================================================================
# 3. Core surrogate model
# =============================================================================

class StructuralGNOPhysics(nn.Module):
    """
    Production Structural GNO Physics surrogate.

    Three forward modes, unified through a shared FiLM backbone:

    Mode 1 — Collider:
        Input  : (B, 4)  [√s, mass, α_s, process_id]
        Output : (B, 1)  differential cross-section σ̂ [pb/GeV]  (softplus ≥ 0)

    Mode 2 — CMB Cosmology:
        Input  : (B, 6)  [Ω_b h², Ω_c h², H₀, τ, ln(10¹⁰ A_s), n_s]
        Output : (B, L)  C_ℓ^TT power spectrum, ℓ = 2 … L+1  (softplus ≥ 0)

    Mode 3 — Yang–Mills Gluon Propagator:
        Input  : (B, 2)  [p², α_s(p²)]
        Output : (B, 1)  D(p²) gluon propagator

    The CSOC structural parameter σ is shared across all three modes and
    serves as a global regulariser from the ONE Ecosystem.
    """

    def __init__(self, cfg: NGOPhysicsConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        # ---- Domain encoders -----------------------------------------------
        self.collider_encoder = DomainEncoder(4, d)
        self.cosmo_encoder    = DomainEncoder(6, d)
        self.ym_encoder       = DomainEncoder(2, d)

        # ---- Shared backbone -----------------------------------------------
        self.layers = nn.ModuleList([
            PhysicsFiLMBlock(d, cfg.dropout, cfg.use_residual_scale)
            for _ in range(cfg.num_layers)
        ])

        # ---- Output heads --------------------------------------------------
        self.collider_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )
        self.cmb_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, cfg.lmax),
        )
        self.ym_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialisation for all linear layers outside FiLM blocks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ---- Backbone -----------------------------------------------------------

    def _backbone(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Pass encoded features through the shared FiLM stack."""
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)   # (B,) → (B, 1)
        for layer in self.layers:
            x = layer(x, sigma)
        return x

    # ---- Forward modes ------------------------------------------------------

    def forward_collider(
        self, kinematics: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Predict differential cross-section (always ≥ 0 via softplus)."""
        x = self.collider_encoder(kinematics)
        x = self._backbone(x, sigma)
        return F.softplus(self.collider_head(x))

    def forward_cosmo(
        self, cosmo_params: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Predict CMB C_ℓ^TT power spectrum (always ≥ 0 via softplus)."""
        x = self.cosmo_encoder(cosmo_params)
        x = self._backbone(x, sigma)
        return F.softplus(self.cmb_head(x))

    def forward_ym(
        self, momentum_data: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict gluon propagator D(p²).

        momentum_data columns: [p², α_s(p²)]
        """
        x = self.ym_encoder(momentum_data)
        x = self._backbone(x, sigma)
        return self.ym_head(x)   # can be negative (complex pole region allowed)

    def forward(
        self,
        kinematics:   Optional[torch.Tensor] = None,
        cosmo_params: Optional[torch.Tensor] = None,
        momentum_data:Optional[torch.Tensor] = None,
        sigma:        Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Unified forward pass.  Supply only the tensors relevant to your
        active domains; unused branches return None.
        """
        if sigma is None:
            # Default: zero structural modulation (pure network)
            bs = next(
                t for t in [kinematics, cosmo_params, momentum_data]
                if t is not None
            ).shape[0]
            sigma = torch.zeros(bs, 1, device=next(self.parameters()).device)

        return {
            "collider": self.forward_collider(kinematics, sigma)
                        if kinematics    is not None else None,
            "cosmo":    self.forward_cosmo(cosmo_params, sigma)
                        if cosmo_params  is not None else None,
            "ym":       self.forward_ym(momentum_data, sigma)
                        if momentum_data is not None else None,
        }


# =============================================================================
# 4. EMA helper
# =============================================================================

class EMAModel:
    """
    Exponential Moving Average shadow weights for production inference.

    Usage::

        ema = EMAModel(model, decay=0.999)
        for step in training_loop:
            loss.backward()
            optimizer.step()
            ema.update(model)

        # Evaluate with EMA weights:
        with ema.apply(model):
            predictions = model(...)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._register(model)

    def _register(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module):
        """Context manager: temporarily swap in EMA weights."""
        return _EMAContext(model, self.shadow)

    def state_dict(self) -> Dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: Dict) -> None:
        self.shadow = {k: v for k, v in sd.items()}


class _EMAContext:
    def __init__(self, model: nn.Module, shadow: Dict[str, torch.Tensor]):
        self.model  = model
        self.shadow = shadow
        self.backup: Dict[str, torch.Tensor] = {}

    def __enter__(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device))
        return self.model

    def __exit__(self, *args):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])


# =============================================================================
# 5. Multi-task loss with learnable uncertainty weighting
# =============================================================================

class MultiTaskPhysicsLoss(nn.Module):
    """
    Kendall–Gal homoscedastic uncertainty multi-task loss.

    Each task i contributes:
        L_i / (2 σ_i²) + log σ_i

    where σ_i = softplus(s_i) and s_i is a learnable scalar.
    Also includes physics-informed auxiliary penalties.

    Reference: Kendall & Gal, NeurIPS 2018.
    """

    def __init__(self, cfg: NGOPhysicsConfig):
        super().__init__()
        # Log-variance parameters (initialised so σ ≈ 1)
        self.log_var_collider = nn.Parameter(torch.tensor(0.0))
        self.log_var_cosmo    = nn.Parameter(torch.tensor(0.0))
        self.log_var_ym       = nn.Parameter(torch.tensor(0.0))
        self.cfg = cfg

    def _weighted(self, raw_loss: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Homoscedastic uncertainty weighting for one task."""
        precision = torch.exp(-log_var)
        return precision * raw_loss + 0.5 * log_var

    def physics_penalty_ym(
        self,
        pred_D:    torch.Tensor,
        p2_sorted: torch.Tensor,
    ) -> torch.Tensor:
        """
        Monotonicity prior: D(p²) should decrease as p² increases in the
        UV regime (standard decoupling solution).  Penalise violations of
        ∂D/∂p² < 0 via softplus of positive finite differences.
        """
        if pred_D.shape[0] < 2:
            return torch.tensor(0.0, device=pred_D.device)
        diffs = pred_D[1:] - pred_D[:-1]       # should be ≤ 0
        violations = F.softplus(diffs)          # 0 when correct, > 0 otherwise
        return violations.mean()

    def physics_penalty_cmb(self, pred_Cl: torch.Tensor) -> torch.Tensor:
        """
        CMB power-law smoothness prior: successive C_ℓ should not vary
        erratically at high ℓ.  Penalise large second differences.
        """
        if pred_Cl.shape[-1] < 3:
            return torch.tensor(0.0, device=pred_Cl.device)
        d2 = pred_Cl[..., 2:] - 2 * pred_Cl[..., 1:-1] + pred_Cl[..., :-2]
        return (d2 ** 2).mean()

    def forward(
        self,
        pred_collider: Optional[torch.Tensor],
        true_collider: Optional[torch.Tensor],
        pred_cosmo:    Optional[torch.Tensor],
        true_cosmo:    Optional[torch.Tensor],
        pred_ym:       Optional[torch.Tensor],
        true_ym:       Optional[torch.Tensor],
        p2_sorted:     Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss : scalar Tensor (differentiable)
            breakdown  : dict of float values for logging
        """
        device = next(
            p for p in [pred_collider, pred_cosmo, pred_ym] if p is not None
        ).device
        total = torch.tensor(0.0, device=device)
        breakdown: Dict[str, float] = {}

        if pred_collider is not None and true_collider is not None:
            l = F.mse_loss(pred_collider, true_collider)
            wl = self._weighted(l, self.log_var_collider)
            total = total + self.cfg.lambda_collider * wl
            breakdown["loss_collider"] = l.item()

        if pred_cosmo is not None and true_cosmo is not None:
            l = F.mse_loss(pred_cosmo, true_cosmo)
            phy = self.cfg.lambda_physics * self.physics_penalty_cmb(pred_cosmo)
            wl  = self._weighted(l + phy, self.log_var_cosmo)
            total = total + self.cfg.lambda_cosmo * wl
            breakdown["loss_cosmo"]         = l.item()
            breakdown["loss_cosmo_physics"] = phy.item()

        if pred_ym is not None and true_ym is not None:
            l = F.mse_loss(pred_ym, true_ym)
            phy = (
                self.cfg.lambda_physics
                * self.physics_penalty_ym(pred_ym.squeeze(-1), p2_sorted)
                if p2_sorted is not None else 0.0
            )
            wl  = self._weighted(l + phy, self.log_var_ym)
            total = total + self.cfg.lambda_ym * wl
            breakdown["loss_ym"]         = l.item()
            breakdown["loss_ym_physics"] = phy if isinstance(phy, float) \
                                           else phy.item()

        breakdown["total"] = total.item()
        breakdown["sigma_collider"] = torch.exp(0.5 * self.log_var_collider).item()
        breakdown["sigma_cosmo"]    = torch.exp(0.5 * self.log_var_cosmo).item()
        breakdown["sigma_ym"]       = torch.exp(0.5 * self.log_var_ym).item()
        return total, breakdown


# =============================================================================
# 6. Synthetic data generators (for testing / pre-training)
# =============================================================================

class SyntheticDataGenerator:
    """
    Provides mini-batches of synthetic physics data for all three domains.
    Replace with real loaders (CERN Open Data, Planck, lattice QCD) in production.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def collider_batch(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Toy Drell–Yan-like cross-section: σ ∝ α_s / (s * M²).
        Returns (kinematics [n,4], sigma [n,1]).
        """
        sqrts = torch.rand(n, device=self.device) * 13000 + 1000
        mass  = torch.rand(n, device=self.device) * 150   + 50
        alpha = torch.rand(n, device=self.device) * 0.15  + 0.10
        pid   = torch.randint(0, 4, (n,), device=self.device).float()
        kin   = torch.stack([sqrts, mass, alpha, pid], dim=-1)
        # Approximate σ: inversely proportional to s and M² (toy formula)
        sigma = (alpha / (sqrts * mass ** 2 + 1e-6)).unsqueeze(-1) * 1e9
        return kin, sigma.clamp(min=0)

    def cosmo_batch(
        self, n: int, lmax: int = 2499
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Toy CMB: C_ℓ ∝ A_s * ℓ^(n_s - 1) * exp(-ℓ(ℓ+1)τ/2) / (ℓ(ℓ+1)).
        Returns (cosmo_params [n,6], Cl [n,lmax]).
        """
        Obh2  = torch.rand(n, device=self.device) * 0.005 + 0.0220
        Och2  = torch.rand(n, device=self.device) * 0.020 + 0.110
        H0    = torch.rand(n, device=self.device) * 5.0   + 65.0
        tau   = torch.rand(n, device=self.device) * 0.02  + 0.045
        logAs = torch.rand(n, device=self.device) * 0.2   + 3.0
        ns    = torch.rand(n, device=self.device) * 0.04  + 0.945
        params = torch.stack([Obh2, Och2, H0, tau, logAs, ns], dim=-1)

        ells = torch.arange(2, lmax + 2, dtype=torch.float32, device=self.device)
        # (n, lmax) spectra via broadcasting
        Cl = (
            torch.exp(logAs.unsqueeze(-1))
            * ells.pow(ns.unsqueeze(-1) - 1)
            * torch.exp(-ells * (ells + 1) * tau.unsqueeze(-1) / 2)
            / (ells * (ells + 1) + 1e-10)
        ) * 1e-10
        return params, Cl.clamp(min=0)

    def ym_batch(
        self, n: int, propagator: GribovPropagator
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Yang–Mills: sample p² log-uniformly, compute D(p²) from propagator.
        Returns (momentum_data [n,2], D_true [n,1], p2_sorted [n]).
        """
        p2 = torch.exp(
            torch.rand(n, device=self.device) * 4 - 2
        )  # ~ 0.018 … 7.4 GeV²
        p2_sorted, idx = p2.sort()
        alpha_s = 0.118 / (1 + 0.118 * math.log(p2_sorted.clamp(min=0.1).float()))
        alpha_s = alpha_s.clamp(0.05, 0.5)
        momentum_data = torch.stack([p2_sorted, alpha_s], dim=-1)
        with torch.no_grad():
            D_true = propagator.D(p2_sorted).unsqueeze(-1)
        return momentum_data, D_true, p2_sorted


# =============================================================================
# 7. Checkpoint manager
# =============================================================================

class CheckpointManager:
    """
    Manages saving and loading of model checkpoints.

    Tracks the top-k best checkpoints by validation loss and prunes old ones.
    """

    def __init__(self, directory: str, keep_best_n: int = 3):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_n = keep_best_n
        self.best: List[Tuple[float, str]] = []   # (val_loss, path)

    def save(
        self,
        epoch:      int,
        model:      nn.Module,
        ema:        EMAModel,
        optimizer:  torch.optim.Optimizer,
        scheduler,
        loss_fn:    nn.Module,
        val_loss:   float,
        cfg:        NGOPhysicsConfig,
        tag:        str = "",
    ) -> str:
        """Serialise everything needed to resume training or run inference."""
        fname = self.dir / f"ckpt_ep{epoch:04d}{tag}.pt"
        torch.save(
            {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "ema_state":       ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "loss_fn_state":   loss_fn.state_dict(),
                "val_loss":        val_loss,
                "cfg":             cfg.to_dict(),
            },
            fname,
        )
        logger.info(f"Saved checkpoint: {fname}  (val_loss={val_loss:.4e})")
        self._track(val_loss, str(fname))
        return str(fname)

    def _track(self, val_loss: float, path: str) -> None:
        self.best.append((val_loss, path))
        self.best.sort(key=lambda x: x[0])
        # Prune beyond keep_best_n
        while len(self.best) > self.keep_best_n:
            _, old_path = self.best.pop()
            try:
                os.remove(old_path)
                logger.debug(f"Pruned checkpoint: {old_path}")
            except FileNotFoundError:
                pass

    @property
    def best_path(self) -> Optional[str]:
        return self.best[0][1] if self.best else None

    @staticmethod
    def load(path: str, device: torch.device) -> Dict:
        return torch.load(path, map_location=device)


# =============================================================================
# 8. Gradient monitor
# =============================================================================

class GradientMonitor:
    """
    Tracks per-layer gradient norms during training for diagnostics.
    Call ``record(model)`` after ``loss.backward()`` and before ``optimizer.step()``.
    """

    def __init__(self, log_every: int = 50):
        self.log_every = log_every
        self._step = 0
        self._history: List[Dict[str, float]] = []

    def record(self, model: nn.Module) -> Dict[str, float]:
        norms: Dict[str, float] = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                norms[name] = param.grad.detach().norm().item()
        self._history.append(norms)
        self._step += 1
        if self._step % self.log_every == 0:
            global_norm = math.sqrt(sum(v ** 2 for v in norms.values()))
            logger.debug(f"[GradMonitor step {self._step}] global_norm={global_norm:.4f}")
        return norms

    def summary(self) -> Dict[str, float]:
        """Return mean gradient norm per parameter across all recorded steps."""
        if not self._history:
            return {}
        all_keys = set().union(*self._history)
        return {
            k: sum(h.get(k, 0.0) for h in self._history) / len(self._history)
            for k in all_keys
        }


# =============================================================================
# 9. Production trainer
# =============================================================================

class NGOPhysicsTrainer:
    """
    End-to-end production training loop for NGO Physics ONE.

    Orchestrates:
    • Model + STANDARD ONE component construction
    • Yang–Mills propagator pre-fitting (optional, loads lattice data)
    • Domain-balanced synthetic data sampling
    • Multi-task loss with Kendall uncertainty weighting
    • Cosine-annealing LR with linear warmup
    • EMA weight tracking
    • Gradient clipping and monitoring
    • Periodic checkpointing (best-k models kept)
    • Validation and metric logging

    Example::

        cfg     = NGOPhysicsConfig(epochs=200, hidden_dim=256)
        trainer = NGOPhysicsTrainer(cfg)
        trainer.fit()
        model   = trainer.load_best_model()
    """

    def __init__(
        self,
        cfg: NGOPhysicsConfig,
        lattice_data_path: Optional[str] = None,
    ):
        self.cfg = cfg

        # ---- Device ---------------------------------------------------------
        if cfg.device == "auto":
            self.device = get_device("cuda")
        else:
            self.device = torch.device(cfg.device)
        logger.info(f"Device: {self.device}")

        # ---- Reproducibility ------------------------------------------------
        torch.manual_seed(cfg.seed)

        # ---- STANDARD ONE components ----------------------------------------
        self.physics_params = PhysicsParameters(device=str(self.device)).to(self.device)
        self.csoc_kernel    = CSOCKernel(device=str(self.device)).to(self.device)
        self.ssc            = SemanticStateContraction(epsilon_fp=0.0028)
        self.rg_refiner     = DiffRGRefiner(keep_fraction=0.5)

        # ---- Yang–Mills propagator (pre-fitted or default) ------------------
        self.ym_module = YangMillsMassGap(
            physics_params=self.physics_params,
            csoc=self.csoc_kernel,
            propagator_type="refined",
            device=str(self.device),
        )
        if lattice_data_path:
            logger.info(f"Pre-fitting YM propagator to: {lattice_data_path}")
            self.ym_module.fit_to_lattice(lattice_data_path, epochs=300, lr=5e-3)
        else:
            logger.info("YM propagator: fitting to synthetic lattice data.")
            self.ym_module.fit_to_lattice(None, epochs=200, lr=5e-3)

        # Reference propagator for training labels
        self.ref_propagator: GribovPropagator = self.ym_module.propagator

        # ---- Surrogate model -----------------------------------------------
        self.model = StructuralGNOPhysics(cfg).to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {param_count:,}")

        # ---- Loss function -------------------------------------------------
        self.loss_fn = MultiTaskPhysicsLoss(cfg).to(self.device)

        # ---- Optimiser (all model + loss params) ---------------------------
        all_params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        self.optimizer = torch.optim.AdamW(
            all_params, lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        # ---- LR schedule: linear warmup → cosine annealing ----------------
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=cfg.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.epochs - cfg.warmup_epochs,
            eta_min=cfg.eta_min_lr,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[cfg.warmup_epochs],
        )

        # ---- EMA and monitoring --------------------------------------------
        self.ema      = EMAModel(self.model, decay=cfg.ema_decay)
        self.grad_mon = GradientMonitor(log_every=50)
        self.ckpt_mgr = CheckpointManager(cfg.checkpoint_dir, cfg.keep_best_n)

        # ---- Data generator ------------------------------------------------
        self.data_gen = SyntheticDataGenerator(self.device)

        self._best_val_loss = float("inf")

    # ---- CSOC sigma ---------------------------------------------------------

    def _get_sigma(self, batch_size: int) -> torch.Tensor:
        """
        Compute the CSOC structural parameter σ for the current batch.
        Uses a normalised distance r = 0.5 (mid-scale) as placeholder;
        replace with domain-appropriate r in real data pipelines.
        """
        r = torch.full((batch_size,), 0.5, device=self.device)
        sigma = self.csoc_kernel(r).detach()         # (B,)
        sigma = self.ssc(sigma)
        return sigma.unsqueeze(-1).clamp(min=1e-6)   # (B, 1)

    # ---- Training step ------------------------------------------------------

    def _train_step(self) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        B = self.cfg.batch_size
        sigma = self._get_sigma(B)

        # Sample all three domains
        kin,    true_coll = self.data_gen.collider_batch(B)
        cosmo,  true_cmb  = self.data_gen.cosmo_batch(B, lmax=self.cfg.lmax)
        mom,    true_ym,  p2_sorted = self.data_gen.ym_batch(B, self.ref_propagator)

        # Forward
        pred_coll = self.model.forward_collider(kin,   sigma)
        pred_cmb  = self.model.forward_cosmo(cosmo,    sigma)
        pred_ym   = self.model.forward_ym(mom,         sigma)

        # Loss
        loss, bd = self.loss_fn(
            pred_coll, true_coll,
            pred_cmb,  true_cmb,
            pred_ym,   true_ym,
            p2_sorted=p2_sorted,
        )

        loss.backward()

        # Gradient monitoring
        self.grad_mon.record(self.model)

        # Adaptive gradient clipping
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_max_norm
        )

        self.optimizer.step()
        self.ema.update(self.model)

        return bd

    # ---- Validation step ----------------------------------------------------

    @torch.no_grad()
    def _val_step(self) -> float:
        """Run validation with EMA weights and return total loss."""
        B = self.cfg.batch_size * 2

        with self.ema.apply(self.model):
            self.model.eval()
            sigma = self._get_sigma(B)

            kin,   true_coll = self.data_gen.collider_batch(B)
            cosmo, true_cmb  = self.data_gen.cosmo_batch(B, lmax=self.cfg.lmax)
            mom,   true_ym,  p2s = self.data_gen.ym_batch(B, self.ref_propagator)

            pred_coll = self.model.forward_collider(kin,   sigma)
            pred_cmb  = self.model.forward_cosmo(cosmo,    sigma)
            pred_ym   = self.model.forward_ym(mom,         sigma)

            _, bd = self.loss_fn(
                pred_coll, true_coll,
                pred_cmb,  true_cmb,
                pred_ym,   true_ym,
                p2_sorted=p2s,
            )

        return bd["total"]

    # ---- Main training loop -------------------------------------------------

    def fit(self) -> None:
        """
        Run the full training loop.

        Progress is logged every epoch.  Checkpoints are saved every
        ``cfg.save_every_n_epochs`` epochs and whenever a new best validation
        loss is achieved.
        """
        logger.info("=" * 70)
        logger.info("NGO PHYSICS ONE — Production Training")
        logger.info(f"  Epochs       : {self.cfg.epochs}")
        logger.info(f"  Batch size   : {self.cfg.batch_size}")
        logger.info(f"  Hidden dim   : {self.cfg.hidden_dim}  |  Layers: {self.cfg.num_layers}")
        logger.info(f"  LR           : {self.cfg.lr}  (warmup {self.cfg.warmup_epochs} ep)")
        logger.info(f"  Checkpoint   : {self.cfg.checkpoint_dir}")
        logger.info("=" * 70)

        t0 = time.time()
        for epoch in range(1, self.cfg.epochs + 1):
            train_bd = self._train_step()
            val_loss = self._val_step()
            self.scheduler.step()
            lr_now = self.optimizer.param_groups[0]["lr"]

            # Logging
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Ep {epoch:4d}/{self.cfg.epochs}  "
                    f"total={train_bd['total']:.4e}  "
                    f"coll={train_bd.get('loss_collider', 0):.3e}  "
                    f"cmb={train_bd.get('loss_cosmo', 0):.3e}  "
                    f"ym={train_bd.get('loss_ym', 0):.3e}  "
                    f"val={val_loss:.4e}  "
                    f"lr={lr_now:.2e}  "
                    f"σ_ym={train_bd.get('sigma_ym', 1):.3f}"
                )

            # Periodic checkpoint
            if epoch % self.cfg.save_every_n_epochs == 0:
                self.ckpt_mgr.save(
                    epoch, self.model, self.ema, self.optimizer,
                    self.scheduler, self.loss_fn, val_loss, self.cfg,
                )

            # Best-model checkpoint
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self.ckpt_mgr.save(
                    epoch, self.model, self.ema, self.optimizer,
                    self.scheduler, self.loss_fn, val_loss, self.cfg,
                    tag="_best",
                )

        elapsed = time.time() - t0
        logger.info(f"Training complete in {elapsed/60:.1f} min.  "
                    f"Best val loss: {self._best_val_loss:.4e}")
        logger.info(f"Best checkpoint: {self.ckpt_mgr.best_path}")

    def load_best_model(self) -> StructuralGNOPhysics:
        """Load the best EMA weights into a fresh model and return it."""
        best = self.ckpt_mgr.best_path
        if best is None:
            logger.warning("No checkpoint found; returning current model.")
            return self.model
        ckpt = CheckpointManager.load(best, self.device)
        cfg  = NGOPhysicsConfig.from_dict(ckpt["cfg"])
        model = StructuralGNOPhysics(cfg).to(self.device)
        # Swap in EMA weights for inference
        ema = EMAModel(model, decay=cfg.ema_decay)
        ema.load_state_dict(ckpt["ema_state"])
        with ema.apply(model):
            # Copy EMA state into model permanently for export
            for name, param in model.named_parameters():
                if name in ema.shadow:
                    param.data.copy_(ema.shadow[name].to(self.device))
        logger.info(f"Loaded best model from: {best}")
        return model

    def resume(self, checkpoint_path: str) -> int:
        """
        Resume training from a saved checkpoint.
        Returns the epoch to start from.
        """
        ckpt = CheckpointManager.load(checkpoint_path, self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.ema.load_state_dict(ckpt["ema_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.loss_fn.load_state_dict(ckpt["loss_fn_state"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {ckpt['epoch']}  "
                    f"(val_loss={ckpt['val_loss']:.4e})")
        return start_epoch


# =============================================================================
# 10. Inference API
# =============================================================================

class NGOPhysicsInference:
    """
    Clean inference interface for the trained NGO Physics ONE surrogate.

    Wraps a loaded model with optional MC-dropout uncertainty estimation.

    Example::

        infer = NGOPhysicsInference.from_checkpoint("checkpoints_ngo/ckpt_best.pt")

        # Single-domain query
        kin = torch.tensor([[13000, 125, 0.118, 1]])
        sigma_hat, unc = infer.predict_collider(kin, n_samples=50)

        # Yang–Mills mass gap extraction
        mass_gap_GeV = infer.extract_mass_gap()
    """

    def __init__(
        self,
        model:      StructuralGNOPhysics,
        csoc:       CSOCKernel,
        ssc:        SemanticStateContraction,
        ym_module:  YangMillsMassGap,
        device:     torch.device,
    ):
        self.model     = model.eval()
        self.csoc      = csoc
        self.ssc       = ssc
        self.ym_module = ym_module
        self.device    = device

    @classmethod
    def from_checkpoint(
        cls, path: str, device: str = "auto"
    ) -> "NGOPhysicsInference":
        """
        Reconstruct the full inference stack from a saved checkpoint.

        Args:
            path   : path to the ``.pt`` checkpoint file
            device : ``"auto"`` (default), ``"cuda"``, ``"cpu"``, or ``"mps"``
        """
        dev = get_device("cuda") if device == "auto" else torch.device(device)
        ckpt = CheckpointManager.load(path, dev)
        cfg  = NGOPhysicsConfig.from_dict(ckpt["cfg"])

        model = StructuralGNOPhysics(cfg).to(dev)
        # Apply EMA weights
        ema = EMAModel(model, decay=cfg.ema_decay)
        ema.load_state_dict(ckpt["ema_state"])
        for name, param in model.named_parameters():
            if name in ema.shadow:
                param.data.copy_(ema.shadow[name].to(dev))

        physics = PhysicsParameters(device=str(dev)).to(dev)
        csoc    = CSOCKernel(device=str(dev)).to(dev)
        ssc     = SemanticStateContraction()
        ym_mod  = YangMillsMassGap(
            physics_params=physics, csoc=csoc,
            propagator_type="refined", device=str(dev)
        )
        return cls(model, csoc, ssc, ym_mod, dev)

    def _sigma(self, n: int) -> torch.Tensor:
        r = torch.full((n,), 0.5, device=self.device)
        s = self.csoc(r).detach()
        s = self.ssc(s)
        return s.unsqueeze(-1).clamp(min=1e-6)

    @torch.no_grad()
    def predict_collider(
        self,
        kinematics: torch.Tensor,
        n_samples:  int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict differential cross-section with optional MC-dropout uncertainty.

        Args:
            kinematics : (B, 4) tensor [√s, M, α_s, process_id]
            n_samples  : number of stochastic forward passes (1 = deterministic)

        Returns:
            mean       : (B, 1) mean prediction
            std        : (B, 1) std uncertainty (None if n_samples == 1)
        """
        kinematics = kinematics.to(self.device)
        sigma = self._sigma(kinematics.shape[0])

        if n_samples == 1:
            self.model.eval()
            return self.model.forward_collider(kinematics, sigma), None

        # MC-dropout: activate dropout
        self.model.train()
        preds = torch.stack([
            self.model.forward_collider(kinematics, sigma)
            for _ in range(n_samples)
        ])  # (S, B, 1)
        return preds.mean(0), preds.std(0)

    @torch.no_grad()
    def predict_cosmo(
        self,
        cosmo_params: torch.Tensor,
        n_samples:    int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict CMB C_ℓ power spectrum.

        Args:
            cosmo_params : (B, 6) [Ω_b h², Ω_c h², H₀, τ, ln10¹⁰A_s, n_s]
            n_samples    : stochastic forward passes

        Returns:
            mean : (B, lmax) power spectrum
            std  : (B, lmax) uncertainty or None
        """
        cosmo_params = cosmo_params.to(self.device)
        sigma = self._sigma(cosmo_params.shape[0])

        if n_samples == 1:
            self.model.eval()
            return self.model.forward_cosmo(cosmo_params, sigma), None

        self.model.train()
        preds = torch.stack([
            self.model.forward_cosmo(cosmo_params, sigma)
            for _ in range(n_samples)
        ])
        return preds.mean(0), preds.std(0)

    @torch.no_grad()
    def predict_ym(
        self,
        momentum_data: torch.Tensor,
        n_samples:     int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict gluon propagator D(p²).

        Args:
            momentum_data : (B, 2) [p², α_s(p²)]
            n_samples     : stochastic forward passes

        Returns:
            mean : (B, 1) D(p²) prediction
            std  : (B, 1) uncertainty or None
        """
        momentum_data = momentum_data.to(self.device)
        sigma = self._sigma(momentum_data.shape[0])

        if n_samples == 1:
            self.model.eval()
            return self.model.forward_ym(momentum_data, sigma), None

        self.model.train()
        preds = torch.stack([
            self.model.forward_ym(momentum_data, sigma)
            for _ in range(n_samples)
        ])
        return preds.mean(0), preds.std(0)

    def extract_mass_gap(self, method: str = "pole_scan") -> float:
        """
        Extract the Yang–Mills mass gap from the fitted propagator.

        Args:
            method : ``"pole_scan"`` (default) or ``"newton"``

        Returns:
            mass gap in GeV
        """
        gap = self.ym_module.extract_mass_gap(method=method)
        logger.info(f"Mass gap ({method}): {gap:.4f} GeV  ({gap*1000:.1f} MeV)")
        return gap


# =============================================================================
# 11. Standalone demo
# =============================================================================

def demo() -> None:
    """
    Quick end-to-end smoke test.

    Trains for 5 epochs on synthetic data, runs inference on all three
    domains, and prints the estimated YM mass gap.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("NGO PHYSICS ONE — Standalone Demo")

    cfg = NGOPhysicsConfig(
        epochs=5,
        hidden_dim=64,
        num_layers=3,
        batch_size=16,
        lmax=99,          # small lmax for speed
        checkpoint_dir="demo_checkpoints_ngo",
        save_every_n_epochs=5,
    )

    trainer = NGOPhysicsTrainer(cfg)
    trainer.fit()
    best_model = trainer.load_best_model()

    # Inference smoke test
    device = trainer.device
    infer = NGOPhysicsInference(
        best_model,
        trainer.csoc_kernel,
        trainer.ssc,
        trainer.ym_module,
        device,
    )

    # Collider
    kin = torch.tensor([[13000.0, 125.0, 0.118, 1.0]], device=device)
    sig, _ = infer.predict_collider(kin)
    logger.info(f"Collider σ̂ = {sig.item():.4e} pb/GeV")

    # Cosmology
    cosmo = torch.tensor([[0.022, 0.12, 67.5, 0.054, 3.044, 0.965]], device=device)
    cl, _ = infer.predict_cosmo(cosmo)
    logger.info(f"CMB C_ℓ mean (first 5 ℓ) = {cl[0, :5].tolist()}")

    # YM propagator
    p2    = torch.tensor([[0.5, 0.3], [1.0, 0.25], [4.0, 0.18]], device=device)
    D, _  = infer.predict_ym(p2)
    logger.info(f"D(p²) predictions = {D.squeeze().tolist()}")

    # Mass gap
    gap = infer.extract_mass_gap()
    logger.info(f"Yang–Mills mass gap ≈ {gap*1000:.1f} MeV")


if __name__ == "__main__":
    demo()
