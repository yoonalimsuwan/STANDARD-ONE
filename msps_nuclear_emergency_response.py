#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_nuclear_emergency_response.py — v1.0.0
Real-Time Nuclear Power Plant Emergency Response Decision Support System

For severe accident scenarios (LOCA, SBO, station blackout, core damage):
  - Real-time accident progression monitoring
  - Plume trajectory and dose projection
  - Dynamic evacuation zone calculation
  - Protective Action Recommendations (PARs)
  - Emergency classification and escalation
  - First responder guidance

PHYSICS GATE: Commercial reactors CANNOT undergo nuclear detonation.
This models chemical/thermal explosions and radiological release only.
=============================================================================
Author : PAI AND Yoon A. Limsuwan / MSPS NETWORK
ORCID  : 0009-0008-2374-0788
License: MIT | Year 2026
"""
from __future__ import annotations
import math, json, time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
from enum import Enum

class EmergencyClass(Enum):
    """NRC Emergency Classification Levels"""
    UNUSUAL_EVENT = 1      # ลดระดับความปลอดภัย
    ALERT = 2              # เหตุการณ์ที่อาจลดระดับความปลอดภัย
    SITE_AREA_EMERGENCY = 3 # เหตุการณ์ภายในพื้นที่โรงงาน
    GENERAL_EMERGENCY = 4   # เหตุการณ์รุนแรง มีการปล่อยรังสี

class AccidentType(Enum):
    LOCA = "Loss-of-Coolant Accident"
    SBO = "Station Blackout"
    TRANSIENT = "Anticipated Transient Without Scram"
    SGTR = "Steam Generator Tube Rupture"
    UNKNOWN = "Unknown/Undiagnosed"

@dataclass
class ReactorState:
    """Real-time reactor state from sensors"""
    core_temp_K: float = 600.0
    pressure_Pa: float = 15.5e6  # PWR primary
    neutron_flux_rel: float = 1.0  # relative to full power
    coolant_level_m: float = 4.0
    containment_pressure_Pa: float = 101325.0
    radiation_rate_Sv_h: float = 0.0
    hydrogen_conc_pct: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class WeatherConditions:
    """Meteorological data for plume dispersion"""
    wind_speed_m_s: float = 3.0
    wind_direction_deg: float = 180.0  # from south
    stability_class: str = "D"  # Pasquill stability
    mixing_height_m: float = 1000.0
    precipitation_mm_h: float = 0.0

class AccidentProgression:
    """Models accident progression and source term"""
    
    def __init__(self, reactor_type: str = "PWR"):
        self.reactor_type = reactor_type
        self.accident_start_time = time.time()
        
    def diagnose_accident(self, state: ReactorState) -> Dict[str, Any]:
        """Diagnose accident type from sensor readings"""
        if state.coolant_level_m < 2.0 and state.pressure_Pa < 10e6:
            accident = AccidentType.LOCA
            severity = "HIGH"
        elif state.neutron_flux_rel < 0.01 and state.core_temp_K > 900:
            accident = AccidentType.SBO
            severity = "CRITICAL"
        elif state.containment_pressure_Pa > 400000:
            accident = AccidentType.SGTR
            severity = "HIGH"
        else:
            accident = AccidentType.UNKNOWN
            severity = "MODERATE"
            
        return {
            "accident_type": accident.value,
            "severity": severity,
            "time_since_start_s": time.time() - self.accident_start_time,
            "recommended_actions": self._get_actions(accident, severity)
        }
    
    def _get_actions(self, accident: AccidentType, severity: str) -> List[str]:
        actions = []
        if severity == "CRITICAL":
            actions.extend([
                "IMMEDIATE: Initiate emergency core cooling",
                "IMMEDIATE: Seal containment",
                "IMMEDIATE: Activate filtered venting if pressure > 500 kPa",
                "Notify off-site authorities"
            ])
        if accident == AccidentType.LOCA:
            actions.extend([
                "Activate high-pressure injection",
                "Start containment spray",
                "Monitor hydrogen concentration"
            ])
        return actions
    
    def estimate_source_term(self, state: ReactorState, 
                            t_minutes: float) -> Dict[str, Any]:
        """Estimate radioactive release (source term)"""
        # Simplified source term model
        if state.core_temp_K > 1500:  # Core damage temperature
            release_fraction = min(0.5, t_minutes / 120.0)  # Up to 50% in 2 hours
            noble_gas_fraction = 1.0  # All noble gases released
            iodine_fraction = 0.25 * release_fraction
            cesium_fraction = 0.10 * release_fraction
        else:
            release_fraction = 0.001
            noble_gas_fraction = 0.1
            iodine_fraction = 0.01
            cesium_fraction = 0.005
            
        # Activity in Bq (simplified)
        total_inventory_Bq = 1e18  # Typical PWR inventory
        release_rate_Bq_s = total_inventory_Bq * release_fraction / 3600.0
        
        return {
            "release_rate_Bq_s": release_rate_Bq_s,
            "noble_gas_fraction": noble_gas_fraction,
            "iodine_fraction": iodine_fraction,
            "cesium_fraction": cesium_fraction,
            "effective_height_m": 50.0,  # Stack height or breach height
            "STATUS": "Source term estimate - requires validation"
        }

class PlumeDispersion:
    """Gaussian plume model for atmospheric dispersion"""
    
    def __init__(self, weather: WeatherConditions):
        self.weather = weather
        
    def calculate_concentration(self, source_Bq_s: float, 
                               x_m: float, y_m: float, z_m: float = 0.0
                               ) -> float:
        """Calculate concentration at downwind location"""
        if x_m <= 0:
            return 0.0
            
        # Pasquill-Gifford dispersion parameters (simplified)
        sigma_y, sigma_z = self._get_dispersion_params(x_m)
        
        # Wind direction adjustment
        wind_rad = math.radians(self.weather.wind_direction_deg)
        x_eff = x_m * math.cos(wind_rad) + y_m * math.sin(wind_rad)
        y_eff = -x_m * math.sin(wind_rad) + y_m * math.cos(wind_rad)
        
        if x_eff <= 0:
            return 0.0
            
        # Gaussian plume equation
        term_y = math.exp(-0.5 * (y_eff / sigma_y)**2)
        term_z = math.exp(-0.5 * (z_m / sigma_z)**2) + \
                 math.exp(-0.5 * ((z_m + 0) / sigma_z)**2)  # Ground reflection
        
        C = (source_Bq_s / (2 * math.pi * self.weather.wind_speed_m_s * 
             sigma_y * sigma_z)) * term_y * term_z
             
        return C
    
    def _get_dispersion_params(self, x_m: float) -> Tuple[float, float]:
        """Get sigma_y and sigma_z based on stability class"""
        # Simplified Pasquill-Gifford curves
        if self.weather.stability_class == "D":  # Neutral
            sigma_y = 0.08 * x_m * (1 + 0.0001 * x_m)**-0.5
            sigma_z = 0.06 * x_m * (1 + 0.00015 * x_m)**-0.5
        elif self.weather.stability_class == "F":  # Stable
            sigma_y = 0.04 * x_m * (1 + 0.0001 * x_m)**-0.5
            sigma_z = 0.03 * x_m * (1 + 0.0003 * x_m)**-0.5
        else:  # Unstable
            sigma_y = 0.16 * x_m * (1 + 0.0001 * x_m)**-0.5
            sigma_z = 0.12 * x_m * (1 + 0.0001 * x_m)**-0.5
            
        return sigma_y, sigma_z
    
    def calculate_dose(self, concentration_Bq_m3: float, 
                      exposure_hours: float, 
                      isotope: str = "I-131") -> float:
        """Calculate committed dose in Sv"""
        # Dose coefficients (simplified, Sv per Bq inhaled/ingested)
        dose_coeffs = {
            "I-131": 7.4e-9,   # Thyroid dose
            "Cs-137": 6.7e-9,  # Whole body
            "Xe-133": 1.2e-10  # Noble gas (external)
        }
        
        coeff = dose_coeffs.get(isotope, 1e-9)
        # Breathing rate ~20 m³/day for adult
        intake_Bq = concentration_Bq_m3 * 20.0/24.0 * exposure_hours
        dose_Sv = intake_Bq * coeff
        
        return dose_Sv

class EmergencyResponse:
    """Emergency response decision support"""
    
    def classify_emergency(self, state: ReactorState, 
                          source_term: Dict) -> EmergencyClass:
        """Classify emergency level"""
        if state.radiation_rate_Sv_h > 0.1 or \
           source_term["release_rate_Bq_s"] > 1e15:
            return EmergencyClass.GENERAL_EMERGENCY
        elif state.containment_pressure_Pa > 500000 or \
             state.hydrogen_conc_pct > 4.0:
            return EmergencyClass.SITE_AREA_EMERGENCY
        elif state.core_temp_K > 1000 or state.coolant_level_m < 1.0:
            return EmergencyClass.ALERT
        else:
            return EmergencyClass.UNUSUAL_EVENT
    
    def calculate_evacuation_zone(self, source_Bq_s: float, 
                                 weather: WeatherConditions,
                                 dose_threshold_Sv: float = 0.05
                                 ) -> Dict[str, Any]:
        """Calculate evacuation radius based on projected dose"""
        # Binary search for radius where dose = threshold
        r_min, r_max = 100.0, 50000.0  # 100 m to 50 km
        
        for _ in range(20):  # Binary search iterations
            r_mid = (r_min + r_max) / 2.0
            plume = PlumeDispersion(weather)
            conc = plume.calculate_concentration(source_Bq_s, r_mid, 0.0)
            dose = plume.calculate_dose(conc, 24.0)  # 24-hour exposure
            
            if dose > dose_threshold_Sv:
                r_min = r_mid
            else:
                r_max = r_mid
                
        evacuation_radius_m = r_min
        
        # Calculate affected population (simplified)
        area_km2 = math.pi * (evacuation_radius_m / 1000.0)**2
        pop_density = 100  # people per km² (rural)
        affected_population = int(area_km2 * pop_density)
        
        return {
            "evacuation_radius_m": evacuation_radius_m,
            "evacuation_radius_km": evacuation_radius_m / 1000.0,
            "affected_area_km2": area_km2,
            "estimated_population": affected_population,
            "dose_threshold_Sv": dose_threshold_Sv,
            "protective_actions": [
                f"Evacuate within {evacuation_radius_m/1000:.1f} km radius",
                "Distribute KI pills within 10 km",
                "Shelter-in-place 10-20 km",
                "Ban local food/milk within 30 km"
            ]
        }
    
    def generate_situation_report(self, state: ReactorState,
                                 diagnosis: Dict,
                                 emergency_class: EmergencyClass,
                                 evacuation: Dict) -> Dict[str, Any]:
        """Generate emergency situation report"""
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "emergency_classification": emergency_class.name,
            "accident_diagnosis": diagnosis,
            "reactor_state": {
                "core_temperature_K": state.core_temp_K,
                "pressure_MPa": state.pressure_Pa / 1e6,
                "radiation_rate_mSv_h": state.radiation_rate_Sv_h * 1000,
                "hydrogen_pct": state.hydrogen_conc_pct
            },
            "evacuation_recommendation": evacuation,
            "immediate_actions": diagnosis["recommended_actions"],
            "public_communication": {
                "message": self._public_message(emergency_class),
                "channels": ["Emergency broadcast", "SMS alert", "Social media"],
                "update_frequency_minutes": 30 if emergency_class.value >= 3 else 60
            },
            "STATUS": "Emergency response guidance - validate with authorities"
        }
    
    def _public_message(self, emergency_class: EmergencyClass) -> str:
        messages = {
            EmergencyClass.UNUSUAL_EVENT: 
                "เหตุการณ์ผิดปกติที่โรงไฟฟ้านิวเคลียร์ กำลังติดตามสถานการณ์",
            EmergencyClass.ALERT:
                "มีการแจ้งเตือนเหตุฉุกเฉิน กรุณาเตรียมพร้อมรับฟังคำแนะนำ",
            EmergencyClass.SITE_AREA_EMERGENCY:
                "เหตุฉุกเฉินในพื้นที่โรงงาน ผู้อยู่ใกล้เคียงโปรดอยู่ในอาคาร",
            EmergencyClass.GENERAL_EMERGENCY:
                "เหตุฉุกเฉินทั่วไป โปรดปฏิบัติตามคำแนะนำการอพยพทันที"
        }
        return messages[emergency_class]

class RealTimeMonitor:
    """Real-time monitoring and alert system"""
    
    def __init__(self):
        self.alerts = []
        self.trend_data = []
        
    def check_thresholds(self, state: ReactorState) -> List[Dict[str, Any]]:
        """Check if parameters exceed safety thresholds"""
        alerts = []
        
        if state.core_temp_K > 1200:
            alerts.append({
                "level": "CRITICAL",
                "parameter": "Core Temperature",
                "value": state.core_temp_K,
                "threshold": 1200,
                "action": "Initiate emergency cooling immediately"
            })
            
        if state.pressure_Pa > 17e6:
            alerts.append({
                "level": "HIGH",
                "parameter": "Primary Pressure",
                "value": state.pressure_Pa,
                "threshold": 17e6,
                "action": "Activate pressure relief systems"
            })
            
        if state.hydrogen_conc_pct > 4.0:
            alerts.append({
                "level": "CRITICAL",
                "parameter": "Hydrogen Concentration",
                "value": state.hydrogen_conc_pct,
                "threshold": 4.0,
                "action": "Activate PARs, prepare for deflagration"
            })
            
        if state.radiation_rate_Sv_h > 0.01:
            alerts.append({
                "level": "HIGH",
                "parameter": "Radiation Rate",
                "value": state.radiation_rate_Sv_h,
                "threshold": 0.01,
                "action": "Issue evacuation warning"
            })
            
        return alerts
    
    def predict_trend(self, current: ReactorState, 
                     dt_minutes: float = 30.0) -> Dict[str, Any]:
        """Simple linear extrapolation for short-term prediction"""
        # This is a placeholder - real systems use complex models
        temp_rate = 5.0  # K/min during accident
        predicted_temp = current.core_temp_K + temp_rate * dt_minutes
        
        return {
            "current_core_temp_K": current.core_temp_K,
            "predicted_temp_30min_K": predicted_temp,
            "time_to_critical_min": max(0, (1500 - current.core_temp_K) / temp_rate),
            "confidence": "LOW - requires validated thermal-hydraulic model",
            "STATUS": "Trend prediction - use for planning only"
        }

# =============================================================================
# Emergency Response Workflow
# =============================================================================

def run_emergency_scenario():
    """Demonstrate complete emergency response workflow"""
    print("="*70)
    print("NUCLEAR EMERGENCY RESPONSE SYSTEM - DEMONSTRATION")
    print("="*70)
    
    # Initialize systems
    weather = WeatherConditions(wind_speed_m_s=3.0, wind_direction_deg=180,
                               stability_class="D")
    progression = AccidentProgression()
    response = EmergencyResponse()
    monitor = RealTimeMonitor()
    
    # Simulate deteriorating conditions
    print("\n[SCENARIO] Loss-of-Coolant Accident with core damage")
    states = [
        ReactorState(core_temp_K=800, pressure_Pa=12e6, coolant_level_m=3.0,
                    radiation_rate_Sv_h=0.001, hydrogen_conc_pct=1.0),
        ReactorState(core_temp_K=1200, pressure_Pa=8e6, coolant_level_m=1.5,
                    radiation_rate_Sv_h=0.05, hydrogen_conc_pct=3.5),
        ReactorState(core_temp_K=1800, pressure_Pa=5e6, coolant_level_m=0.5,
                    radiation_rate_Sv_h=0.5, hydrogen_conc_pct=6.0,
                    containment_pressure_Pa=450000)
    ]
    
    for i, state in enumerate(states):
        print(f"\n--- Timestep {i+1} ---")
        
        # Check alerts
        alerts = monitor.check_thresholds(state)
        if alerts:
            print(f"⚠️  {len(alerts)} ALERTS:")
            for alert in alerts:
                print(f"   [{alert['level']}] {alert['parameter']}: "
                      f"{alert['value']:.2f} (limit: {alert['threshold']})")
        
        # Diagnose accident
        diagnosis = progression.diagnose_accident(state)
        print(f"🔍 Diagnosis: {diagnosis['accident_type']} - {diagnosis['severity']}")
        
        # Estimate source term
        source = progression.estimate_source_term(state, i*30)
        print(f"☢️  Release rate: {source['release_rate_Bq_s']:.2e} Bq/s")
        
        # Classify emergency
        emergency_class = response.classify_emergency(state, source)
        print(f"🚨 Emergency Class: {emergency_class.name}")
        
        # Calculate evacuation zone
        if source["release_rate_Bq_s"] > 1e12:
            evac = response.calculate_evacuation_zone(
                source["release_rate_Bq_s"], weather)
            print(f"🏃 Evacuation radius: {evac['evacuation_radius_km']:.1f} km")
            print(f"👥 Affected population: ~{evac['estimated_population']:,}")
            
            # Generate situation report
            report = response.generate_situation_report(
                state, diagnosis, emergency_class, evac)
            print(f"📢 Public message: {report['public_communication']['message']}")

if __name__ == "__main__":
    run_emergency_scenario()
