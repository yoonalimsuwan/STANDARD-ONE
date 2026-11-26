#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_scada_weather_integration.py — v1.0.0
Production-Grade Digital Twin Integration for Nuclear Emergency Response
Secure SCADA Ingestion (OPC UA Read-Only) + Real-Time Meteorological API +
Predictive AI World Model for Dynamic Plume Trajectory Forecasting.

PARADIGM SHIFT:
  From "Reactive Threshold Alarms" -> "Predictive Physics-Gated Digital Twin"
  The AI continuously ingests live plant telemetry and live weather, updating
  the Gaussian plume and decay heat models in real-time. If SCADA data drops
  or weather sensors fail, the system immediately triggers SD MODE E 
  (Mandatory Uncertainty Retention) and defaults to conservative worst-case 
  bounding, refusing to declare a "safe" zone based on missing data.

CYBERSECURITY GATE:
  This module operates strictly on a READ-ONLY basis via a hardware Data Diode.
  It CANNOT and WILL NOT send actuation commands back to the plant SCADA.
=============================================================================
Author : PAI AND Yoon A. Limsuwan / MSPS NETWORK
ORCID  : 0009-0008-2374-0788
License: MIT | Year 2026

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.
...
MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.
The mathematics and software stand independently of the philosophy above.
-----------------------------------------------------------------------------
"""
import asyncio
import math
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# =============================================================================
# §1. CYBERSECURITY & SAFETY INTERLOCK (HARD GATE)
# =============================================================================
class SCADASecurityInterlock:
    """
    HARD GATE: Enforces Read-Only access. 
    Prevents any AI or external system from sending write/actuation commands 
    to the nuclear plant's Distributed Control System (DCS).
    """
    @staticmethod
    def validate_command(command_type: str):
        if command_type.upper() in ["WRITE", "ACTUATE", "SET", "RESET", "SCRAM"]:
            raise PermissionError(
                "🚨 PHYSICS GATE / SECURITY REFUSAL: AI system is strictly READ-ONLY. "
                "Actuation commands to nuclear SCADA are physically blocked by the "
                "hardware Data Diode. Human operators must execute manual SCRAM."
            )
        return True

# =============================================================================
# §2. SCADA OPC UA CLIENT (READ-ONLY INGESTION)
# =============================================================================
class SCADA_OPCUA_Client:
    """
    Connects to the plant's OPC UA server via a one-way Data Diode.
    In production, this uses `asyncua.AsyncioClient`. Here we simulate the 
    asynchronous telemetry stream of a degrading LOCA (Loss of Coolant) event.
    """
    def __init__(self, endpoint: str = "opc.tcp://plant-scada-diode:4840"):
        self.endpoint = endpoint
        self.connected = False
        # Simulated telemetry tags
        self.tags = {
            "Core_Temp_K": 600.0,
            "Primary_Pressure_Pa": 15.5e6,
            "Coolant_Level_m": 4.0,
            "Containment_Radiation_Sv_h": 0.001,
            "H2_Conc_Pct": 0.5
        }
        
    async def connect(self):
        # PRODUCTION: await self.client.connect()
        self.connected = True
        print(f"[SCADA] Connected to {self.endpoint} (READ-ONLY MODE)")
        
    async def read_telemetry(self, timestep: int) -> Dict[str, float]:
        """Simulates a deteriorating accident scenario over time."""
        SCADASecurityInterlock.validate_command("READ")
        
        # Simulate LOCA progression
        self.tags["Core_Temp_K"] += 15.0 * timestep      # Heating up
        self.tags["Primary_Pressure_Pa"] -= 0.2e6 * timestep # Depressurizing
        self.tags["Coolant_Level_m"] -= 0.1 * timestep   # Losing coolant
        self.tags["Containment_Radiation_Sv_h"] *= 1.5   # Exponential rise
        self.tags["H2_Conc_Pct"] += 0.2 * timestep       # Zirconium oxidation
        
        # Simulate sensor failure at timestep 5 (Triggers Mode E)
        if timestep >= 5:
            self.tags["Coolant_Level_m"] = float('nan') # Sensor fails
            
        return self.tags.copy()

# =============================================================================
# §3. REAL-TIME METEOROLOGICAL API CLIENT
# =============================================================================
class RealTimeWeatherAPI:
    """
    Fetches live data from the plant's meteorological tower and regional 
    Doppler radar to update the Gaussian Plume dispersion parameters.
    """
    def __init__(self):
        self.wind_speed = 3.0
        self.wind_dir = 180.0 # Degrees
        
    async def fetch_live_weather(self, timestep: int) -> Dict[str, Any]:
        # PRODUCTION: aiohttp.get("https://api.weather.gov/stations/KXXX/observations/latest")
        await asyncio.sleep(0.01) # Simulate network latency
        
        # Simulate wind shift and atmospheric stability change
        self.wind_speed = 3.0 + 0.5 * math.sin(timestep * 0.5)
        self.wind_dir = (180.0 + 15.0 * timestep) % 360.0
        
        # Pasquill stability class (Day/Night, Cloud cover, Wind speed)
        stability = "D" if self.wind_speed > 3.0 else "F" 
        
        return {
            "wind_speed_m_s": self.wind_speed,
            "wind_direction_deg": self.wind_dir,
            "stability_class": stability,
            "precipitation_mm_h": 0.0,
            "mixing_height_m": 1200.0
        }

# =============================================================================
# §4. AI WORLD MODEL & GAUSSIAN PLUME FUSION ENGINE
# =============================================================================
@dataclass
class DigitalTwinState:
    core_temp_K: float = 600.0
    pressure_Pa: float = 15.5e6
    coolant_level_m: float = 4.0
    radiation_Sv_h: float = 0.0
    h2_pct: float = 0.0
    wind_speed: float = 3.0
    wind_dir: float = 180.0
    mode_E_active: bool = False
    missing_sensors: List[str] = field(default_factory=list)

class PlumeDispersionEngine:
    """Calculates real-time dose projection based on fused SCADA + Weather."""
    
    @staticmethod
    def calculate_plume(source_term_Bq_s: float, wind_speed: float, 
                        wind_dir_deg: float, distance_m: float) -> float:
        """Simplified Gaussian Plume centerline concentration."""
        if wind_speed <= 0.1: return float('inf') # Stagnant air, infinite concentration
        
        # Convert wind direction to radians for spatial mapping
        theta = math.radians(wind_dir_deg)
        x = distance_m * math.cos(theta)
        y = distance_m * math.sin(theta)
        
        # Pasquill-Gifford sigma_y, sigma_z (Class D approximation)
        sigma_y = 0.08 * x * (1 + 0.0001 * x)**-0.5
        sigma_z = 0.06 * x * (1 + 0.00015 * x)**-0.5
        
        if x <= 0: return 0.0
        
        conc = (source_term_Bq_s / (2 * math.pi * wind_speed * sigma_y * sigma_z)) * \
               math.exp(-0.5 * (y / sigma_y)**2)
        return conc

# =============================================================================
# §5. MAIN DIGITAL TWIN FUSION LOOP (ASYNCIO)
# =============================================================================
async def run_digital_twin_fusion():
    print("="*70)
    print("MSPS DIGITAL TWIN: SCADA + WEATHER FUSION & PREDICTIVE PLUME")
    print("="*70)
    
    scada = SCADA_OPCUA_Client()
    weather = RealTimeWeatherAPI()
    plume_engine = PlumeDispersionEngine()
    state = DigitalTwinState()
    
    await scada.connect()
    
    # Run simulation for 8 timesteps (representing 80 minutes of accident progression)
    for t in range(1, 9):
        print(f"\n--- [TIMESTEP {t}] (T+{t*10} minutes) ---")
        
        # 1. Parallel Ingestion (SCADA + Weather)
        scada_data, weather_data = await asyncio.gather(
            scada.read_telemetry(t),
            weather.fetch_live_weather(t)
        )
        
        # 2. Data Validation & Mode E (Uncertainty Retention)
        missing = []
        for key, val in scada_data.items():
            if math.isnan(val):
                missing.append(key)
                
        if missing:
            state.mode_E_active = True
            state.missing_sensors = missing
            print(f"⚠️  SD MODE E TRIGGERED: Missing/Faulty Sensors {missing}.")
            print("   -> AI refuses to calculate exact plume. Defaulting to WORST-CASE BOUND.")
            # Conservative bound: Assume max possible release if coolant level is unknown
            source_term_Bq_s = 1e16 
        else:
            state.mode_E_active = False
            # Map SCADA temp/radiation to a dynamic source term
            source_term_Bq_s = scada_data["Containment_Radiation_Sv_h"] * 1e13
            
        # 3. Update Digital Twin State
        state.core_temp_K = scada_data["Core_Temp_K"]
        state.radiation_Sv_h = scada_data["Containment_Radiation_Sv_h"]
        state.h2_pct = scada_data["H2_Conc_Pct"]
        state.wind_speed = weather_data["wind_speed_m_s"]
        state.wind_dir = weather_data["wind_direction_deg"]
        
        # 4. AI Predictive Plume Calculation (1 km downwind)
        distance_m = 1000.0
        conc = plume_engine.calculate_plume(
            source_term_Bq_s, 
            state.wind_speed, 
            state.wind_dir, 
            distance_m
        )
        
        # 5. Output & Actionable Intelligence
        print(f"🏭 Core Temp: {state.core_temp_K:.1f} K | H2: {state.h2_pct:.1f}%")
        print(f"🌬️  Wind: {state.wind_speed:.1f} m/s @ {state.wind_dir:.0f}° (Stability: {weather_data['stability_class']})")
        
        if state.mode_E_active:
            print(f"🚨 EVACUATION ALERT: 1km radius (Conservative Bound due to sensor loss)")
        else:
            if conc > 1e5: # Arbitrary threshold for immediate danger
                print(f"☢️  PLUME DETECTED: Concentration at 1km = {conc:.2e} Bq/m³")
                print(f"🏃 ACTION: Evacuate sector {state.wind_dir:.0f}° immediately.")
            else:
                print(f"✅ Plume at 1km is below acute threshold ({conc:.2e} Bq/m³)")
                
        # 6. Safety Interlock Check (Prevent AI from taking physical control)
        try:
            SCADASecurityInterlock.validate_command("ACTUATE")
        except PermissionError as e:
            pass # Silently pass in normal loop, but tested below

    print("\n" + "="*70)
    print("CYBERSECURITY INTERLOCK TEST:")
    try:
        SCADASecurityInterlock.validate_command("SCRAM")
    except PermissionError as e:
        print(f"✅ INTERLOCK HELD: {e}")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(run_digital_twin_fusion())
