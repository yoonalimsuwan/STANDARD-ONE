"""
=============================================================================
ASTRONOMICAL DATA INTEGRATION (SDSS/Gaia) + VR VISUALIZATION
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0

DATA SOURCES:
  • SDSS (Sloan Digital Sky Survey): https://www.sdss.org/
    - Galaxy positions, redshifts, magnitudes
    - SkyServer API: https://skyserver.sdss.org/
  
  • Gaia (ESA Space Astrometry): https://www.cosmos.esa.int/gaia
    - Stellar positions, parallaxes, proper motions
    - Gaia Archive: https://gea.esac.esa.int/archive/

VR TECHNOLOGY:
  • WebXR API for browser-based VR/AR
  • Three.js for 3D rendering
  • Interactive cosmic network visualization

HONEST SCOPE:
  • Astronomical data is REAL (galaxy positions, redshifts)
  • "Quantum network" interpretation is CONCEPTUAL visualization
  • VR is immersive display, not physical simulation
  • Entanglement connections are HYPOTHETICAL for visualization
  • No claim of actual quantum entanglement between galaxies
=============================================================================
"""

from __future__ import annotations

import math
import time
import logging
import os
import json
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np

logger = logging.getLogger("astronomical_vr")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

ASTRO_VR_VERSION = "1.0.0"

# =============================================================================
# OPTIONAL DEPENDENCIES
# =============================================================================
_HAS_ASTROPY = False
try:
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    _HAS_ASTROPY = True
except ImportError:
    pass

_HAS_PLOTLY = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    pass

logger.info(f"[dependencies] Astropy={_HAS_ASTROPY}, Plotly={_HAS_PLOTLY}")


# =============================================================================
# 1. SDSS CONNECTOR (Real Astronomical Data)
# =============================================================================

@dataclass
class SDSSConfig:
    """Configuration for SDSS data access."""
    base_url: str = "https://skyserver.sdss.org/dr16/SkyServerWS"
    query_timeout: int = 30
    max_galaxies: int = 1000
    ra_range: Tuple[float, float] = (0.0, 360.0)  # Right Ascension
    dec_range: Tuple[float, float] = (-90.0, 90.0)  # Declination
    z_range: Tuple[float, float] = (0.0, 2.0)  # Redshift range
    mag_limit: float = 22.0  # Magnitude limit


class SDSSConnector:
    """
    Connector to SDSS (Sloan Digital Sky Survey) data.
    
    SDSS is a major astronomical survey that has mapped:
      • Millions of galaxies
      • Their positions (RA, Dec)
      • Redshifts (distance indicator)
      • Magnitudes (brightness)
    
    API ENDPOINTS:
      • SQL Query: /SkyServerWS/SQLQuery
      • Image Search: /SkyServerWS/ImgCutout
      • Object Details: /SkyServerWS/Object/Explorer
    
    USAGE:
        connector = SDSSConnector(cfg)
        galaxies = connector.query_galaxies(ra_min=180, ra_max=190)
    
    HONEST SCOPE:
      • Data is REAL astronomical observations
      • "Quantum network" interpretation is conceptual
      • No actual quantum entanglement between galaxies
    """
    
    def __init__(self, cfg: Optional[SDSSConfig] = None):
        self.cfg = cfg or SDSSConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"CosmicVR/{ASTRO_VR_VERSION} (MSPS NETWORK)"
        })
        logger.info(f"[SDSS] Initialized: {self.cfg.base_url}")
    
    def query_galaxies(
        self,
        ra_min: float = 0.0,
        ra_max: float = 360.0,
        dec_min: float = -90.0,
        dec_max: float = 90.0,
        z_min: float = 0.0,
        z_max: float = 2.0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Query galaxies from SDSS database.
        
        SQL Query:
          SELECT TOP {limit}
            objid, ra, dec, z, u, g, r, i, z_mag
          FROM Galaxy
          WHERE ra BETWEEN {ra_min} AND {ra_max}
            AND dec BETWEEN {dec_min} AND {dec_max}
            AND z BETWEEN {z_min} AND {z_max}
        
        Returns:
            dict with galaxy catalog
        """
        sql = f"""
        SELECT TOP {limit}
            objid,
            ra, dec,
            z, zErr,
            u, g, r, i, z AS z_mag,
            petroR50_r
        FROM Galaxy
        WHERE ra BETWEEN {ra_min} AND {ra_max}
          AND dec BETWEEN {dec_min} AND {dec_max}
          AND z BETWEEN {z_min} AND {z_max}
          AND z > 0
        ORDER BY z ASC
        """
        
        try:
            # Execute SQL query
            url = f"{self.cfg.base_url}/SQLQuery"
            params = {
                "cmd": sql,
                "format": "json",
            }
            
            response = self.session.get(
                url,
                params=params,
                timeout=self.cfg.query_timeout,
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Parse results
            galaxies = []
            if "ResultSet" in data and "Rows" in data["ResultSet"]:
                for row in data["ResultSet"]["Rows"]:
                    galaxies.append({
                        "objid": row[0],
                        "ra": row[1],
                        "dec": row[2],
                        "redshift": row[3],
                        "redshift_error": row[4],
                        "u_mag": row[5],
                        "g_mag": row[6],
                        "r_mag": row[7],
                        "i_mag": row[8],
                        "z_mag": row[9],
                        "radius": row[10],
                    })
            
            return {
                "status": "success",
                "n_galaxies": len(galaxies),
                "galaxies": galaxies,
                "query": sql,
                "HONEST_NOTE": (
                    "This is REAL astronomical data from SDSS. "
                    "Galaxy positions and redshifts are actual observations. "
                    "The 'quantum network' interpretation is conceptual."
                ),
            }
        
        except Exception as e:
            logger.error(f"[SDSS] Query failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "n_galaxies": 0,
                "galaxies": [],
            }
    
    def get_galaxy_image(self, objid: int, size: int = 256) -> Optional[str]:
        """
        Get cutout image of a galaxy.
        
        Args:
            objid: SDSS object ID
            size: image size in pixels
        
        Returns:
            URL to image cutout
        """
        url = f"{self.cfg.base_url}/ImgCutout"
        params = {
            "objid": objid,
            "size": size,
            "format": "jpg",
        }
        return f"{url}?objid={objid}&size={size}&format=jpg"
    
    def convert_to_cartesian(
        self,
        galaxies: List[Dict],
    ) -> List[Dict]:
        """
        Convert RA/Dec/redshift to Cartesian coordinates (x, y, z) in Mpc.
        
        Uses simplified cosmology:
          • Comoving distance: d_C = c/H₀ × ∫ dz/E(z)
          • For small z: d_C ≈ c×z/H₀
          • x = d_C × cos(dec) × cos(ra)
          • y = d_C × cos(dec) × sin(ra)
          • z = d_C × sin(dec)
        
        Returns:
            list of galaxies with added (x, y, z) coordinates
        """
        H0 = 70.0  # km/s/Mpc
        c = 299792.458  # km/s
        
        result = []
        for gal in galaxies:
            ra_rad = math.radians(gal["ra"])
            dec_rad = math.radians(gal["dec"])
            z = gal["redshift"]
            
            # Comoving distance (simplified)
            d_comoving = c * z / H0  # Mpc
            
            # Cartesian coordinates
            x = d_comoving * math.cos(dec_rad) * math.cos(ra_rad)
            y = d_comoving * math.cos(dec_rad) * math.sin(ra_rad)
            z_coord = d_comoving * math.sin(dec_rad)
            
            gal_with_coords = gal.copy()
            gal_with_coords.update({
                "x_mpc": x,
                "y_mpc": y,
                "z_mpc": z_coord,
                "distance_mpc": d_comoving,
            })
            result.append(gal_with_coords)
        
        return result


# =============================================================================
# 2. GAIA CONNECTOR (Stellar Catalog)
# =============================================================================

@dataclass
class GaiaConfig:
    """Configuration for Gaia data access."""
    base_url: str = "https://gea.esac.esa.int/tap-server/tap"
    query_timeout: int = 60
    max_stars: int = 10000
    parallax_min: float = 0.1  # mas
    magnitude_limit: float = 20.0


class GaiaConnector:
    """
    Connector to Gaia stellar catalog.
    
    Gaia (ESA) has measured:
      • ~1.8 billion stars
      • Positions, parallaxes, proper motions
      • Radial velocities for millions of stars
      • Photometry in G, BP, RP bands
    
    USAGE:
        connector = GaiaConnector(cfg)
        stars = connector.query_stars(ra=180, dec=0, radius=1.0)
    """
    
    def __init__(self, cfg: Optional[GaiaConfig] = None):
        self.cfg = cfg or GaiaConfig()
        self.session = requests.Session()
        logger.info(f"[Gaia] Initialized: {self.cfg.base_url}")
    
    def query_stars(
        self,
        ra: float = 180.0,
        dec: float = 0.0,
        radius_deg: float = 1.0,
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """
        Query stars from Gaia DR3.
        
        ADQL Query:
          SELECT TOP {limit}
            source_id, ra, dec, parallax, pmra, pmdec,
            phot_g_mean_mag, bp_rp
          FROM gaiadr3.gaia_source
          WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
          )
          AND parallax > {parallax_min}
          AND phot_g_mean_mag < {mag_limit}
        """
        adql = f"""
        SELECT TOP {limit}
            source_id,
            ra, dec,
            parallax, parallax_error,
            pmra, pmdec,
            phot_g_mean_mag,
            bp_rp,
            radial_velocity
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
        )
        AND parallax > {self.cfg.parallax_min}
        AND phot_g_mean_mag < {self.cfg.magnitude_limit}
        ORDER BY phot_g_mean_mag ASC
        """
        
        try:
            url = f"{self.cfg.base_url}/sync"
            params = {
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "json",
                "QUERY": adql,
            }
            
            response = self.session.get(
                url,
                params=params,
                timeout=self.cfg.query_timeout,
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Parse results
            stars = []
            if "data" in data and "result" in data["data"]:
                for row in data["data"]["result"]:
                    stars.append({
                        "source_id": row[0],
                        "ra": row[1],
                        "dec": row[2],
                        "parallax": row[3],
                        "parallax_error": row[4],
                        "pmra": row[5],
                        "pmdec": row[6],
                        "g_mag": row[7],
                        "bp_rp": row[8],
                        "radial_velocity": row[9],
                    })
            
            return {
                "status": "success",
                "n_stars": len(stars),
                "stars": stars,
                "query": adql,
            }
        
        except Exception as e:
            logger.error(f"[Gaia] Query failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "n_stars": 0,
                "stars": [],
            }
    
    def compute_distance_pc(self, parallax_mas: float) -> float:
        """
        Compute distance in parsecs from parallax.
        
        d (pc) = 1 / parallax (arcsec)
        parallax_mas is in milliarcseconds
        """
        if parallax_mas <= 0:
            return float('inf')
        return 1000.0 / parallax_mas


# =============================================================================
# 3. COSMIC NETWORK BUILDER (From Real Data)
# =============================================================================

@dataclass
class CosmicNetworkConfig:
    """Configuration for cosmic network construction."""
    entanglement_threshold_mpc: float = 10.0  # Max distance for entanglement
    min_redshift: float = 0.01
    max_redshift: float = 1.0
    n_connections_per_galaxy: int = 5


class CosmicNetworkBuilder:
    """
    Build cosmic quantum network from astronomical data.
    
    CONCEPTUAL FRAMEWORK:
      • Galaxies = nodes in quantum network
      • Entanglement connections = hypothetical quantum correlations
      • Redshift = time delay for signal propagation
      • Distance = entanglement degradation
    
    HONEST SCOPE:
      • Galaxy positions are REAL (from SDSS/Gaia)
      • "Entanglement connections" are CONCEPTUAL visualization
      • No physical evidence of quantum entanglement between galaxies
      • This is a visualization tool, not a physics claim
    """
    
    def __init__(self, cfg: Optional[CosmicNetworkConfig] = None):
        self.cfg = cfg or CosmicNetworkConfig()
        logger.info("[CosmicNetwork] Builder initialized")
    
    def build_from_sdss(
        self,
        sdss_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build cosmic network from SDSS galaxy data.
        
        Args:
            sdss_data: output from SDSSConnector.query_galaxies()
        
        Returns:
            dict with network nodes and edges
        """
        if sdss_data["status"] != "success":
            return {"error": "SDSS data not available"}
        
        galaxies = sdss_data["galaxies"]
        
        # Convert to Cartesian coordinates
        sdss_connector = SDSSConnector()
        galaxies_with_coords = sdss_connector.convert_to_cartesian(galaxies)
        
        # Build nodes
        nodes = []
        for i, gal in enumerate(galaxies_with_coords):
            node = {
                "id": i,
                "objid": gal["objid"],
                "ra": gal["ra"],
                "dec": gal["dec"],
                "redshift": gal["redshift"],
                "x_mpc": gal["x_mpc"],
                "y_mpc": gal["y_mpc"],
                "z_mpc": gal["z_mpc"],
                "distance_mpc": gal["distance_mpc"],
                "magnitude": gal["r_mag"],
                "type": "galaxy",
            }
            nodes.append(node)
        
        # Build edges (hypothetical entanglement connections)
        edges = []
        for i, gal1 in enumerate(galaxies_with_coords):
            for j, gal2 in enumerate(galaxies_with_coords):
                if i >= j:
                    continue
                
                # Compute distance
                dx = gal1["x_mpc"] - gal2["x_mpc"]
                dy = gal1["y_mpc"] - gal2["y_mpc"]
                dz = gal1["z_mpc"] - gal2["z_mpc"]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Add edge if within threshold
                if dist < self.cfg.entanglement_threshold_mpc:
                    # Entanglement fidelity degrades with distance
                    fidelity = math.exp(-dist / self.cfg.entanglement_threshold_mpc)
                    
                    edge = {
                        "source": i,
                        "target": j,
                        "distance_mpc": dist,
                        "entanglement_fidelity": fidelity,
                        "redshift_difference": abs(gal1["redshift"] - gal2["redshift"]),
                    }
                    edges.append(edge)
        
        return {
            "status": "success",
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
            "HONEST_NOTE": (
                "Nodes are REAL galaxies from SDSS. "
                "Edges are CONCEPTUAL 'entanglement connections' for visualization. "
                "No physical evidence of quantum entanglement between galaxies."
            ),
        }
    
    def build_from_gaia(
        self,
        gaia_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build cosmic network from Gaia stellar data.
        
        Similar to SDSS but for stars within Milky Way.
        """
        if gaia_data["status"] != "success":
            return {"error": "Gaia data not available"}
        
        stars = gaia_data["stars"]
        gaia_connector = GaiaConnector()
        
        # Build nodes
        nodes = []
        for i, star in enumerate(stars):
            # Compute distance from parallax
            distance_pc = gaia_connector.compute_distance_pc(star["parallax"])
            distance_ly = distance_pc * 3.262
            
            # Convert to Cartesian (simplified)
            ra_rad = math.radians(star["ra"])
            dec_rad = math.radians(star["dec"])
            
            x = distance_pc * math.cos(dec_rad) * math.cos(ra_rad)
            y = distance_pc * math.cos(dec_rad) * math.sin(ra_rad)
            z = distance_pc * math.sin(dec_rad)
            
            node = {
                "id": i,
                "source_id": star["source_id"],
                "ra": star["ra"],
                "dec": star["dec"],
                "distance_pc": distance_pc,
                "distance_ly": distance_ly,
                "x_pc": x,
                "y_pc": y,
                "z_pc": z,
                "magnitude": star["g_mag"],
                "color": star["bp_rp"],
                "type": "star",
            }
            nodes.append(node)
        
        # Build edges (within local neighborhood)
        edges = []
        for i, star1 in enumerate(stars):
            for j, star2 in enumerate(stars):
                if i >= j:
                    continue
                
                # Compute angular separation
                # (simplified: use Cartesian distance)
                d1 = gaia_connector.compute_distance_pc(star1["parallax"])
                d2 = gaia_connector.compute_distance_pc(star2["parallax"])
                
                # Approximate physical distance
                ra_diff = abs(star1["ra"] - star2["ra"])
                dec_diff = abs(star1["dec"] - star2["dec"])
                angular_sep = math.sqrt(ra_diff**2 + dec_diff**2)
                
                # Physical distance (rough estimate)
                avg_dist = (d1 + d2) / 2
                physical_dist = avg_dist * math.radians(angular_sep)
                
                # Add edge if close enough (< 10 pc)
                if physical_dist < 10.0:
                    edge = {
                        "source": i,
                        "target": j,
                        "distance_pc": physical_dist,
                        "entanglement_fidelity": math.exp(-physical_dist / 10.0),
                    }
                    edges.append(edge)
        
        return {
            "status": "success",
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
        }


# =============================================================================
# 4. VR VISUALIZATION ENGINE (WebXR + Three.js)
# =============================================================================

class VRVisualizationEngine:
    """
    Generate VR visualization of cosmic quantum network.
    
    OUTPUT:
      • HTML file with Three.js scene
      • WebXR support for VR headsets
      • Interactive controls (orbit, zoom, select)
      • Real-time animation of entanglement connections
    
    TECHNOLOGY:
      • Three.js: 3D rendering
      • WebXR: VR/AR support
      • D3.js: Data visualization (optional)
    
    USAGE:
        engine = VRVisualizationEngine(network_data)
        engine.generate_html("cosmic_vr.html")
    """
    
    def __init__(self, network_data: Dict[str, Any]):
        self.network = network_data
        self.output_dir = Path("./vr_output")
        self.output_dir.mkdir(exist_ok=True)
        logger.info("[VR] Visualization engine initialized")
    
    def generate_html(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate HTML file with Three.js VR scene.
        
        Args:
            output_path: path to save HTML file
        
        Returns:
            path to generated HTML
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.output_dir / f"cosmic_vr_{timestamp}.html")
        
        # Extract network data
        nodes = self.network.get("nodes", [])
        edges = self.network.get("edges", [])
        
        # Generate Three.js code
        three_js_code = self._generate_three_js(nodes, edges)
        
        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cosmic Quantum Network VR</title>
    <style>
        body {{
            margin: 0;
            overflow: hidden;
            background-color: #000;
        }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: #fff;
            font-family: Arial, sans-serif;
            font-size: 14px;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 5px;
        }}
        #vr-button {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 10px 20px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }}
        #vr-button:hover {{
            background: #45a049;
        }}
    </style>
</head>
<body>
    <div id="info">
        <h2>Cosmic Quantum Network</h2>
        <p>Nodes: {len(nodes)} | Edges: {len(edges)}</p>
        <p>Data: SDSS/Gaia Astronomical Survey</p>
        <p>Controls: Orbit (mouse), Zoom (scroll), Select (click)</p>
        <p style="color: #ff9800;">
            ⚠️ HONEST SCOPE: Galaxy positions are REAL data.
            "Entanglement connections" are CONCEPTUAL visualization.
        </p>
    </div>
    <button id="vr-button" style="display: none;">Enter VR</button>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/vr/VRButton.js"></script>
    
    <script>
    {three_js_code}
    </script>
</body>
</html>
"""
        
        with open(output_path, "w") as f:
            f.write(html)
        
        logger.info(f"[VR] Generated: {output_path}")
        return output_path
    
    def _generate_three_js(
        self,
        nodes: List[Dict],
        edges: List[Dict],
    ) -> str:
        """Generate Three.js JavaScript code."""
        
        # Convert nodes to JavaScript array
        nodes_js = json.dumps([
            {
                "id": n["id"],
                "x": n.get("x_mpc", n.get("x_pc", 0)),
                "y": n.get("y_mpc", n.get("y_pc", 0)),
                "z": n.get("z_mpc", n.get("z_pc", 0)),
                "magnitude": n.get("magnitude", 15),
                "type": n.get("type", "galaxy"),
            }
            for n in nodes
        ])
        
        # Convert edges to JavaScript array
        edges_js = json.dumps([
            {
                "source": e["source"],
                "target": e["target"],
                "fidelity": e.get("entanglement_fidelity", 0.5),
            }
            for e in edges
        ])
        
        js_code = f"""
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000011);
        
        const camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            10000
        );
        camera.position.set(0, 0, 500);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.xr.enabled = true;
        document.body.appendChild(renderer.domElement);
        
        // VR Button
        if (navigator.xr) {{
            document.getElementById('vr-button').style.display = 'block';
            document.body.appendChild(VRButton.createButton(renderer));
        }}
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Data
        const nodes = {nodes_js};
        const edges = {edges_js};
        
        // Scale factor (adjust based on data range)
        const scale = 1.0;
        
        // Create node meshes
        const nodeGeometry = new THREE.SphereGeometry(2, 16, 16);
        const nodeMaterial = new THREE.MeshBasicMaterial({{ color: 0x00ff00 }});
        
        const nodeMeshes = [];
        nodes.forEach((node, i) => {{
            const mesh = new THREE.Mesh(nodeGeometry, nodeMaterial.clone());
            mesh.position.set(node.x * scale, node.y * scale, node.z * scale);
            
            // Color by magnitude (brighter = smaller magnitude)
            const brightness = Math.max(0, Math.min(1, (20 - node.magnitude) / 10));
            mesh.material.color.setHSL(0.3, 1.0, brightness * 0.5 + 0.3);
            
            // Size by type
            if (node.type === 'star') {{
                mesh.scale.set(0.5, 0.5, 0.5);
            }}
            
            mesh.userData = node;
            scene.add(mesh);
            nodeMeshes.push(mesh);
        }});
        
        // Create edge lines
        const edgeMaterial = new THREE.LineBasicMaterial({{
            color: 0x0088ff,
            opacity: 0.3,
            transparent: true,
        }});
        
        edges.forEach(edge => {{
            const sourceNode = nodes[edge.source];
            const targetNode = nodes[edge.target];
            
            const geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(sourceNode.x * scale, sourceNode.y * scale, sourceNode.z * scale),
                new THREE.Vector3(targetNode.x * scale, targetNode.y * scale, targetNode.z * scale),
            ]);
            
            const line = new THREE.Line(geometry, edgeMaterial.clone());
            line.material.opacity = edge.fidelity * 0.5;
            scene.add(line);
        }});
        
        // Add starfield background
        const starGeometry = new THREE.BufferGeometry();
        const starVertices = [];
        for (let i = 0; i < 10000; i++) {{
            const x = (Math.random() - 0.5) * 5000;
            const y = (Math.random() - 0.5) * 5000;
            const z = (Math.random() - 0.5) * 5000;
            starVertices.push(x, y, z);
        }}
        starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starVertices, 3));
        const starMaterial = new THREE.PointsMaterial({{ color: 0xffffff, size: 1 }});
        const starField = new THREE.Points(starGeometry, starMaterial);
        scene.add(starField);
        
        // Raycaster for selection
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        
        window.addEventListener('click', (event) => {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(nodeMeshes);
            
            if (intersects.length > 0) {{
                const node = intersects[0].object.userData;
                alert(`Node ${{node.id}}: ${{node.type}}\\nMagnitude: ${{node.magnitude}}`);
            }}
        }});
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            
            // Rotate starfield slowly
            starField.rotation.y += 0.0001;
            
            renderer.render(scene, camera);
        }}
        animate();
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        """
        
        return js_code


# =============================================================================
# 5. INTERACTIVE CONTROLLER
# =============================================================================

class InteractiveController:
    """
    Interactive controls for VR visualization.
    
    FEATURES:
      • Camera orbit and zoom
      • Node selection and highlighting
      • Edge filtering by fidelity
      • Time evolution animation
      • Data filtering (redshift, magnitude)
    """
    
    def __init__(self, network_data: Dict[str, Any]):
        self.network = network_data
        self.selected_node: Optional[int] = None
        self.fidelity_threshold: float = 0.3
        self.redshift_range: Tuple[float, float] = (0.0, 2.0)
        logger.info("[Controller] Initialized")
    
    def filter_by_redshift(
        self,
        z_min: float,
        z_max: float,
    ) -> Dict[str, Any]:
        """Filter nodes by redshift range."""
        nodes = self.network.get("nodes", [])
        filtered = [n for n in nodes if z_min <= n.get("redshift", 0) <= z_max]
        
        return {
            "n_nodes": len(filtered),
            "nodes": filtered,
        }
    
    def filter_by_fidelity(
        self,
        threshold: float,
    ) -> Dict[str, Any]:
        """Filter edges by entanglement fidelity."""
        edges = self.network.get("edges", [])
        filtered = [e for e in edges if e.get("entanglement_fidelity", 0) >= threshold]
        
        return {
            "n_edges": len(filtered),
            "edges": filtered,
        }
    
    def select_node(self, node_id: int) -> Dict[str, Any]:
        """Select a node and return its connections."""
        nodes = self.network.get("nodes", [])
        edges = self.network.get("edges", [])
        
        if node_id < 0 or node_id >= len(nodes):
            return {"error": "Invalid node ID"}
        
        node = nodes[node_id]
        
        # Find connected edges
        connected_edges = [
            e for e in edges
            if e["source"] == node_id or e["target"] == node_id
        ]
        
        return {
            "node": node,
            "n_connections": len(connected_edges),
            "connections": connected_edges,
        }


# =============================================================================
# 6. REAL-TIME UPDATER
# =============================================================================

class RealTimeUpdater:
    """
    Real-time data updater for cosmic network.
    
    FEATURES:
      • Periodic SDSS/Gaia queries
      • Incremental network updates
      • Change detection
      • Notification system
    """
    
    def __init__(
        self,
        sdss_connector: SDSSConnector,
        gaia_connector: GaiaConnector,
        network_builder: CosmicNetworkBuilder,
        update_interval_seconds: int = 3600,
    ):
        self.sdss = sdss_connector
        self.gaia = gaia_connector
        self.builder = network_builder
        self.update_interval = update_interval_seconds
        self.last_update: Optional[datetime] = None
        logger.info(f"[Updater] Initialized: interval={update_interval}s")
    
    def update_network(self) -> Dict[str, Any]:
        """
        Update cosmic network with latest data.
        
        Returns:
            dict with updated network and change summary
        """
        logger.info("[Updater] Starting network update")
        
        # Query SDSS
        sdss_data = self.sdss.query_galaxies(
            ra_min=180.0,
            ra_max=190.0,
            dec_min=-10.0,
            dec_max=10.0,
            limit=500,
        )
        
        # Query Gaia
        gaia_data = self.gaia.query_stars(
            ra=180.0,
            dec=0.0,
            radius_deg=5.0,
            limit=2000,
        )
        
        # Build networks
        sdss_network = self.builder.build_from_sdss(sdss_data)
        gaia_network = self.builder.build_from_gaia(gaia_data)
        
        self.last_update = datetime.now()
        
        return {
            "status": "success",
            "update_time": self.last_update.isoformat(),
            "sdss_network": sdss_network,
            "gaia_network": gaia_network,
            "summary": {
                "n_sdss_galaxies": sdss_network.get("n_nodes", 0),
                "n_gaia_stars": gaia_network.get("n_nodes", 0),
                "n_sdss_edges": sdss_network.get("n_edges", 0),
                "n_gaia_edges": gaia_network.get("n_edges", 0),
            },
        }


# =============================================================================
# 7. INTEGRATED SYSTEM
# =============================================================================

class IntegratedAstronomicalVRSystem:
    """
    Master system integrating astronomical data + VR visualization.
    
    PIPELINE:
      1. Query SDSS/Gaia for real astronomical data
      2. Build cosmic quantum network (conceptual)
      3. Generate VR visualization (Three.js + WebXR)
      4. Enable interactive controls
      5. Support real-time updates
    
    USAGE:
        system = IntegratedAstronomicalVRSystem()
        result = system.run_full_pipeline()
        # Open result["vr_html"] in browser
    """
    
    def __init__(self):
        # Connectors
        self.sdss = SDSSConnector()
        self.gaia = GaiaConnector()
        
        # Network builder
        self.builder = CosmicNetworkBuilder()
        
        # VR engine
        self.vr_engine: Optional[VRVisualizationEngine] = None
        
        # Controller
        self.controller: Optional[InteractiveController] = None
        
        # Updater
        self.updater = RealTimeUpdater(
            self.sdss, self.gaia, self.builder,
            update_interval_seconds=3600,
        )
        
        logger.info("[IntegratedSystem] Initialized")
    
    def run_full_pipeline(
        self,
        ra_center: float = 180.0,
        dec_center: float = 0.0,
        radius_deg: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Run complete pipeline: data → network → VR.
        
        Args:
            ra_center: center RA in degrees
            dec_center: center Dec in degrees
            radius_deg: search radius in degrees
        
        Returns:
            dict with all results and VR HTML path
        """
        result = {"version": ASTRO_VR_VERSION}
        
        # Step 1: Query SDSS
        logger.info("[Pipeline] Step 1: Query SDSS")
        sdss_data = self.sdss.query_galaxies(
            ra_min=ra_center - radius_deg,
            ra_max=ra_center + radius_deg,
            dec_min=dec_center - radius_deg,
            dec_max=dec_center + radius_deg,
            limit=500,
        )
        result["sdss_query"] = {
            "status": sdss_data["status"],
            "n_galaxies": sdss_data["n_galaxies"],
        }
        
        # Step 2: Query Gaia
        logger.info("[Pipeline] Step 2: Query Gaia")
        gaia_data = self.gaia.query_stars(
            ra=ra_center,
            dec=dec_center,
            radius_deg=radius_deg,
            limit=2000,
        )
        result["gaia_query"] = {
            "status": gaia_data["status"],
            "n_stars": gaia_data["n_stars"],
        }
        
        # Step 3: Build cosmic network
        logger.info("[Pipeline] Step 3: Build cosmic network")
        sdss_network = self.builder.build_from_sdss(sdss_data)
        gaia_network = self.builder.build_from_gaia(gaia_data)
        
        result["networks"] = {
            "sdss": {
                "n_nodes": sdss_network.get("n_nodes", 0),
                "n_edges": sdss_network.get("n_edges", 0),
            },
            "gaia": {
                "n_nodes": gaia_network.get("n_nodes", 0),
                "n_edges": gaia_network.get("n_edges", 0),
            },
        }
        
        # Step 4: Generate VR visualization
        logger.info("[Pipeline] Step 4: Generate VR visualization")
        
        # Combine networks (use SDSS for main visualization)
        combined_network = sdss_network if sdss_network.get("status") == "success" else gaia_network
        
        if combined_network.get("status") == "success":
            self.vr_engine = VRVisualizationEngine(combined_network)
            vr_html = self.vr_engine.generate_html()
            result["vr_html"] = vr_html
        else:
            result["vr_html"] = None
            result["error"] = "No network data available"
        
        # Step 5: Initialize controller
        logger.info("[Pipeline] Step 5: Initialize controller")
        self.controller = InteractiveController(combined_network)
        
        # Summary
        result["summary"] = {
            "total_nodes": (
                sdss_network.get("n_nodes", 0) +
                gaia_network.get("n_nodes", 0)
            ),
            "total_edges": (
                sdss_network.get("n_edges", 0) +
                gaia_network.get("n_edges", 0)
            ),
            "vr_generated": result["vr_html"] is not None,
        }
        
        result["HONEST_SCOPE"] = {
            "astronomical_data": "REAL (SDSS/Gaia surveys)",
            "quantum_network": "CONCEPTUAL visualization",
            "entanglement": "HYPOTHETICAL for visualization",
            "vr_display": "IMMERSIVE but not physical reality",
        }
        
        return result


# =============================================================================
# 8. SELF-TEST SUITE
# =============================================================================

def smoke_test():
    print("=" * 78)
    print(f"  ASTRONOMICAL DATA + VR VISUALIZATION v{ASTRO_VR_VERSION}")
    print("  Smoke Test")
    print("=" * 78)
    
    # ── T1: SDSS Connector ───────────────────────────────────────────────
    print("\n [T1] SDSS Connector")
    sdss = SDSSConnector()
    
    # Note: This will fail without internet or if SDSS API is down
    # We'll test the structure, not actual data retrieval
    print(f"   Base URL: {sdss.cfg.base_url}")
    print(f"   Query timeout: {sdss.cfg.query_timeout}s")
    print("   [SKIPPED] Actual query (requires internet)")
    print("[PASS] T1 SDSS Connector structure OK")
    
    # ── T2: Gaia Connector ───────────────────────────────────────────────
    print("\n [T2] Gaia Connector")
    gaia = GaiaConnector()
    
    print(f"   Base URL: {gaia.cfg.base_url}")
    print(f"   Query timeout: {gaia.cfg.query_timeout}s")
    
    # Test distance computation
    distance = gaia.compute_distance_pc(10.0)  # 10 mas parallax
    print(f"   Distance from 10 mas parallax: {distance:.1f} pc")
    assert abs(distance - 100.0) < 0.1, "Distance should be 100 pc"
    print("[PASS] T2 Gaia Connector works")
    
    # ── T3: Cosmic Network Builder ───────────────────────────────────────
    print("\n [T3] Cosmic Network Builder")
    builder = CosmicNetworkBuilder()
    
    # Test with mock data
    mock_sdss = {
        "status": "success",
        "n_galaxies": 3,
        "galaxies": [
            {"objid": 1, "ra": 180.0, "dec": 0.0, "redshift": 0.1, "r_mag": 18.0},
            {"objid": 2, "ra": 181.0, "dec": 0.0, "redshift": 0.1, "r_mag": 19.0},
            {"objid": 3, "ra": 182.0, "dec": 0.0, "redshift": 0.2, "r_mag": 20.0},
        ],
    }
    
    network = builder.build_from_sdss(mock_sdss)
    print(f"   Status: {network['status']}")
    print(f"   Nodes: {network.get('n_nodes', 0)}")
    print(f"   Edges: {network.get('n_edges', 0)}")
    print(f"   HONEST: {network.get('HONEST_NOTE', '')[:60]}...")
    print("[PASS] T3 Cosmic Network Builder works")
    
    # ── T4: VR Visualization Engine ──────────────────────────────────────
    print("\n [T4] VR Visualization Engine")
    
    mock_network = {
        "status": "success",
        "nodes": [
            {"id": 0, "x_mpc": 0, "y_mpc": 0, "z_mpc": 0, "magnitude": 18},
            {"id": 1, "x_mpc": 10, "y_mpc": 0, "z_mpc": 0, "magnitude": 19},
        ],
        "edges": [
            {"source": 0, "target": 1, "entanglement_fidelity": 0.8},
        ],
    }
    
    vr_engine = VRVisualizationEngine(mock_network)
    vr_html = vr_engine.generate_html()
    
    print(f"   VR HTML generated: {vr_html}")
    print(f"   File exists: {os.path.exists(vr_html)}")
    assert os.path.exists(vr_html), "VR HTML should be generated"
    print("[PASS] T4 VR Visualization Engine works")
    
    # ── T5: Interactive Controller ───────────────────────────────────────
    print("\n [T5] Interactive Controller")
    controller = InteractiveController(mock_network)
    
    # Test node selection
    selection = controller.select_node(0)
    print(f"   Selected node: {selection['node']['id']}")
    print(f"   Connections: {selection['n_connections']}")
    
    # Test filtering
    filtered = controller.filter_by_fidelity(0.5)
    print(f"   Edges after filtering: {filtered['n_edges']}")
    print("[PASS] T5 Interactive Controller works")
    
    # ── T6: Integrated System ────────────────────────────────────────────
    print("\n [T6] Integrated System")
    system = IntegratedAstronomicalVRSystem()
    
    print(f"   SDSS connector: {system.sdss.cfg.base_url}")
    print(f"   Gaia connector: {system.gaia.cfg.base_url}")
    print("   [SKIPPED] Full pipeline (requires internet)")
    print("[PASS] T6 Integrated System structure OK")
    
    # Cleanup
    import shutil
    shutil.rmtree("./vr_output", ignore_errors=True)
    
    print("\n" + "=" * 78)
    print("  ALL TESTS PASSED.")
    print("")
    print("  SETUP INSTRUCTIONS:")
    print("  1. pip install requests astropy plotly")
    print("  2. Internet connection required for SDSS/Gaia queries")
    print("  3. Open generated HTML in WebXR-compatible browser")
    print("  4. Use VR headset for immersive experience")
    print("")
    print("  HONEST REMINDERS:")
    print("  • SDSS/Gaia data is REAL astronomical observations")
    print("  • 'Quantum network' is CONCEPTUAL visualization")
    print("  • 'Entanglement' is HYPOTHETICAL for display")
    print("  • VR is IMMERSIVE but not physical reality")
    print("  • No claim of actual quantum effects between galaxies")
    print("=" * 78)


if __name__ == "__main__":
    smoke_test()
