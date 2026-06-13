import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict

# =============================================================================
# 1. Configuration
# =============================================================================
@dataclass
class SGNOPhysicsConfig:
    hidden_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.1
    
    # Loss Weights สำหรับ Multi-objective Training
    lambda_collider: float = 1.0
    lambda_cosmo: float = 1.0
    lambda_ym: float = 2.0  # ให้ความสำคัญกับ YM Mass Gap เป็นพิเศษ

# =============================================================================
# 2. Structural FiLM Block สำหรับ Physics Operator
# =============================================================================
class PhysicsFiLMBlock(nn.Module):
    """
    MLP Block ที่ถูกควบคุมด้วยตัวแปรเชิงโครงสร้าง (CSOC sigma) 
    ผ่าน Feature-wise Linear Modulation (FiLM)
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        # FiLM Modulators: แปลงค่า sigma -> gamma (scale), beta (shift)
        self.film_gamma = nn.Linear(1, dim)
        self.film_beta  = nn.Linear(1, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # สมมติฐาน: sigma คือ CSOC state ที่ถูกส่งมาจาก CSOCKernel
        gamma = self.film_gamma(sigma)
        beta  = self.film_beta(sigma)
        
        # ปรับแก้สมการฟิสิกส์เชิงโครงสร้าง
        modulated_x = (gamma * x) + beta
        out = self.mlp(modulated_x)
        return self.norm(x + out)

# =============================================================================
# 3. StructuralGNOPhysics (Main AI Surrogate)
# =============================================================================
class StructuralGNOPhysics(nn.Module):
    def __init__(self, cfg: SGNOPhysicsConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        # --- Encoders สำหรับแต่ละโดเมนฟิสิกส์ ---
        # Collider: [sqrts, Mass, alpha_s, process_id]
        self.collider_embed = nn.Sequential(nn.Linear(4, d), nn.LayerNorm(d))
        
        # Cosmology: [Obh2, Och2, H0, tau, logAs, ns]
        self.cosmo_embed = nn.Sequential(nn.Linear(6, d), nn.LayerNorm(d))
        
        # Yang-Mills: [p2 (momentum), alpha_s_running]
        self.ym_embed = nn.Sequential(nn.Linear(2, d), nn.LayerNorm(d))

        # --- Shared Structural Backbone ---
        self.layers = nn.ModuleList([
            PhysicsFiLMBlock(d, cfg.dropout) for _ in range(cfg.num_layers)
        ])

        # --- Output Heads ---
        # พยากรณ์ Cross-section (Differential Distribution)
        self.collider_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1)
        )
        
        # พยากรณ์ C_ell Spectrum (lmax bins)
        self.cmb_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2499) # สมมติ lmax=2500
        )
        
        # พยากรณ์ Gluon Propagator D(p^2)
        self.ym_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1)
        )

    def _apply_backbone(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # กระจายค่า sigma ให้ครอบคลุม batch size
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        
        for layer in self.layers:
            x = layer(x, sigma)
        return x

    # --- Mode 1: Collider Physics ---
    def forward_collider(self, kinematics: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x = self.collider_embed(kinematics)
        x = self._apply_backbone(x, sigma)
        return F.softplus(self.collider_head(x)) # Cross-section ต้องเป็นบวกเสมอ

    # --- Mode 2: Cosmology CMB ---
    def forward_cosmo(self, cosmo_params: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x = self.cosmo_embed(cosmo_params)
        x = self._apply_backbone(x, sigma)
        return F.softplus(self.cmb_head(x))

    # --- Mode 3: Yang-Mills Mass Gap ---
    def forward_ym(self, momentum_data: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        momentum_data: (Batch, 2) ประกอบด้วย p^2 และ running coupling alpha_s(p^2)
        """
        x = self.ym_embed(momentum_data)
        x = self._apply_backbone(x, sigma)
        # Propagator function D(p^2)
        return self.ym_head(x)
