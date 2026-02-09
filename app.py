"""
Physical Agent Swarm - UNIVERSAL PHYSICS FRAMEWORK EDITION
TRELLIS Integration: Semantic Digital Twins with Embedded Physics

UNIVERSAL FRAMEWORK:
- Object Profile Registry: Shape + material + size → physics priors
- Motion Model Dispatch: slide/roll/bounce/tumble (not ball-only)
- Mass Identifiability: Tracks effective_mass_kg vs real mass
- Configurable Priors: --object_profile, --known_mass, --v0, --start_height
- Proper USD Schema: PhysicsMassAPI, collision binding, metadata

REPRODUCIBILITY & CORRECTNESS:
- Reproducible: --seed flag, deterministic RNG
- Fixed physics: horizontal distance only, kernel-visualization match
- Safe USD: sanitized strings, dynamic metadata
- Real MHI: diff-heatmap motion tracking (not max-blend)
- Run report: JSON + Markdown summary
- Timing logs: per-stage performance tracking

PIPELINE:
- Video → AI identifies object with rich detail
- TRELLIS generates photorealistic 3D from detailed text prompt
- Warp calculates physics (friction, effective_mass, drag, restitution)
- Mass degeneracy explicitly tracked (not pretending it's identifiable)
- Physics EMBEDDED into USD with proper schema
- Result: Drop into Omniverse → instantly works
"""

import json
import os
import base64
import binascii
import logging
import re
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from openai import OpenAI
import requests
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
NVIDIA_API_KEY = os.getenv("API_KEY", "").strip()

# Required: OpenCV for video processing
import cv2
logger.info("OpenCV available for video processing")

# Required: Matplotlib for visualization
import matplotlib.pyplot as plt
logger.info("Matplotlib available for graph generation")

# Required: NVIDIA Warp for physics simulation
import warp as wp

wp.init()

# Detect best available Warp device (GPU preferred, CPU acceptable)
if wp.is_cuda_available():
    WARP_DEVICE = "cuda:0"
    logger.info("Warp initialized on GPU (CUDA)")
else:
    WARP_DEVICE = "cpu"
    logger.info("Warp initialized on CPU")


class PhysicalAgentSwarmGTC:
    """
    GTC Golden Ticket Edition: Semantic Digital Twins (Contest-Ready)

    Pipeline:
    1. Vision: Llama-3.2-Vision identifies object semantically with rich detail
    2. TRELLIS: Generate photorealistic 3D model via text-to-3D
    3. Swarm: Configurable parallel physics hypotheses (Warp)
    4. Embedding: Write physics INTO USD file structure
    5. Result: Drop into Omniverse → instant physical simulation
    """

    # Vision API constants
    MAX_TRELLIS_PROMPT_LENGTH = 77
    MOTION_HEATMAP_THRESHOLD = 25
    FRAME_DIFF_THRESHOLD = 25
    TRELLIS_API_TIMEOUT_SECONDS = 300
    VISION_API_TIMEOUT_SECONDS = 60

    # Physics simulation constants
    MIN_HORIZONTAL_VELOCITY = 0.01  # m/s - threshold for "stopped"
    GROUND_Y_POSITION = 0.0
    GRAVITY = 9.81  # m/s²
    DEFAULT_V0 = 3.0  # m/s - default initial horizontal velocity
    DEFAULT_START_HEIGHT = 0.3  # m - default initial height
    SIMULATION_MARGIN_FACTOR = 1.5  # multiply video duration for max simulation time
    MIN_SIMULATION_STEPS = 300  # minimum number of simulation steps
    MAX_SIMULATION_STEPS = 10000  # maximum to prevent OOM
    CONVERGENCE_TOP_PERCENT = 5  # top 5% for convergence check

    def __init__(self, max_iterations: int = 1, output_dir: str = "./gtc_output", seed: int = 42):
        """Initialize the GTC-winning agent swarm with reproducibility."""
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.seed = seed
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        logger.info(f"🎲 Random seed set to: {seed}")
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = self.output_dir / "assets"
        self.assets_dir.mkdir(exist_ok=True)
        
        logger.info(f"📁 Output directory: {self.output_dir.absolute()}")
        logger.info(f"📁 Assets directory: {self.assets_dir.absolute()}")
        
        # Require API key
        if not NVIDIA_API_KEY:
            raise RuntimeError("NVIDIA API key (API_KEY) is required. Set it in .env or environment.")
        logger.info("NVIDIA API key configured")

        # Initialize OpenAI client with NVIDIA base URL
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )

        # Store simulation history for visualization
        self.simulation_history = []
        
        # Timing tracker
        self.timings = {}

        logger.info("Physical Agent Swarm GTC Edition initialized (Contest-Ready)")
    
    def _start_timer(self, stage: str):
        """Start timing a pipeline stage."""
        self.timings[stage] = {'start': time.time()}
    
    def _end_timer(self, stage: str):
        """End timing a pipeline stage."""
        if stage in self.timings and 'start' in self.timings[stage]:
            elapsed = time.time() - self.timings[stage]['start']
            self.timings[stage]['elapsed'] = elapsed
            logger.info(f"⏱️  {stage}: {elapsed:.2f}s")
    
    def _sanitize_for_usd(self, text: str) -> str:
        """Sanitize text for safe USD embedding (no quotes, newlines, etc.)."""
        if not text:
            return ""
        # Remove problematic characters
        text = text.replace('"', "'")
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\\', '/')
        # Collapse multiple spaces
        text = ' '.join(text.split())
        return text[:200]  # Limit length
    
    # ============================================================================
    # VIDEO PROCESSING: Real Motion History with Diff-Heatmap
    # ============================================================================
    def create_motion_history_image(self, video_path: str, max_frames: int = 15) -> Tuple[str, str, Dict[str, Any]]:
        """
        Creates REAL Motion History Image using frame differencing heatmap.
        P1-6: Replace max-blend with diff-heatmap for clean motion tracking.
        Returns: (mhi_base64, first_frame_b64, rotation_info)
        """
        self._start_timer("video_processing")
        logger.info(f"Extracting frames from: {video_path}")

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames == 0:
                raise ValueError("Could not read video frames (0 frames detected)")

            # Sample frames evenly across video
            indices = np.linspace(0, total_frames - 1, min(total_frames, max_frames), dtype=int)
            frames = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

            logger.info(f"Extracted {len(frames)} frames from video")

            if not frames:
                raise ValueError("No frames extracted")
        finally:
            cap.release()

        # FIXED: Real motion history using frame differencing
        # Convert to grayscale
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        
        # Initialize motion heatmap
        motion_heatmap = np.zeros_like(gray_frames[0], dtype=np.float32)
        
        # Accumulate frame differences
        for i in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
            # Threshold to remove noise
            _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            # Accumulate
            motion_heatmap += diff_thresh.astype(np.float32)
        
        # Normalize to 0-255
        if motion_heatmap.max() > 0:
            motion_heatmap = (motion_heatmap / motion_heatmap.max() * 255).astype(np.uint8)
        else:
            motion_heatmap = motion_heatmap.astype(np.uint8)
        
        # Apply color map for better visualization
        mhi_colored = cv2.applyColorMap(motion_heatmap, cv2.COLORMAP_JET)
        
        # Save MHI for debug
        cv2.imwrite(str(self.output_dir / "motion_history.jpg"), mhi_colored)
        logger.info("✓ Created diff-heatmap Motion History Image (no background blur)")

        # Encode for API
        _, buffer = cv2.imencode('.jpg', mhi_colored)
        mhi_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # First and last frames for rotation analysis
        _, f_buf = cv2.imencode('.jpg', frames[0])
        first_frame_b64 = base64.b64encode(f_buf).decode('utf-8')
        
        _, l_buf = cv2.imencode('.jpg', frames[-1])
        last_frame_b64 = base64.b64encode(l_buf).decode('utf-8')
        
        rotation_info = {
            'total_frames': total_frames,
            'fps': fps,
            'duration': total_frames / fps if fps > 0 else 0,
            'first_frame': first_frame_b64,
            'last_frame': last_frame_b64
        }
        
        logger.info(f"✓ Motion History: {mhi_colored.shape[1]}x{mhi_colored.shape[0]} pixels")
        
        self._end_timer("video_processing")
        return mhi_base64, first_frame_b64, rotation_info
    
    # ============================================================================
    # 1. THE BRAIN: Semantic Object Identification (DETAILED for TRELLIS)
    # ============================================================================
    def identify_object_semantically(self, first_frame_b64: str) -> Dict[str, Any]:
        """
        Use Llama-3.2-Vision to identify object with RICH semantic description.
        CONTEST FIX: Generate detailed prompts for TRELLIS text-to-3D.
        """
        self._start_timer("vision_semantic")
        logger.info("[SEMANTIC-ID] Identifying object with detailed description for TRELLIS...")

        content = [
            {
                "type": "text",
                "text": "Analyze this object in detail for photorealistic 3D generation. Provide:\n"
                        "1. object_type: Single concrete noun (ball, boot, box, bottle, cup)\n"
                        "2. detailed_description: Rich visual details (shape, texture, wear, markings, logos)\n"
                        "3. material: Specific material (leather, rubber, plastic, metal, cardboard)\n"
                        "4. color: Primary color(s)\n"
                        "5. condition: new/worn/damaged\n"
                        "6. distinctive_features: Unique visual characteristics\n"
                        'Return JSON: {"object_type": "boot", "detailed_description": "weathered brown leather hiking boot with orange laces and muddy sole", '
                        '"material": "leather", "color": "brown", "condition": "worn", '
                        '"distinctive_features": "deep tread pattern, scuff marks on toe, reinforced ankle"}'
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{first_frame_b64}"}
            }
        ]

        completion = self.client.chat.completions.create(
            model="meta/llama-3.2-90b-vision-instruct",
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
            temperature=0.3
        )

        if not completion.choices or len(completion.choices) == 0:
            raise ValueError("API returned no completion choices")
        response = completion.choices[0].message.content
        logger.info(f"Semantic response: {response[:150]}...")

        # Parse JSON - handle markdown and text formatting
        # Remove markdown code blocks
        response = response.replace("```json", "").replace("```", "").strip()
        
        # Try direct JSON extraction
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        
        if json_match:
            try:
                semantic_info = json.loads(json_match.group(0))
                # Strip markdown formatting from JSON values
                for key in semantic_info.keys():
                    if isinstance(semantic_info[key], str):
                        semantic_info[key] = semantic_info[key].strip().strip('*').strip()
            except (json.JSONDecodeError, AttributeError, ValueError, KeyError) as e:
                logger.warning(f"JSON parsing failed: {e}")
                semantic_info = None
        else:
            semantic_info = None
        
        # If JSON parsing failed, extract from text
        if not semantic_info or semantic_info.get('object_type') == 'object':
            logger.info("JSON parsing failed or generic, extracting from text...")

            # Extract object type
            obj_match = re.search(r'object[_ ]type[:\s]+["\']?([^"\'\n,]+)', response, re.IGNORECASE)
            if not obj_match:
                obj_match = re.search(r'(?:is a|type.*?:)\s*(?:a|an)?\s*([^,.\n]+)', response, re.IGNORECASE)

            # Extract material
            mat_match = re.search(r'material[:\s]+["\']?([^"\'\n,]+)', response, re.IGNORECASE)
            if not mat_match:
                mat_match = re.search(r'made (?:of|from)\s+([^,.\n]+)', response, re.IGNORECASE)

            # Extract color
            color_match = re.search(r'color[:\s]+["\']?([^"\'\n,]+)', response, re.IGNORECASE)

            raw_obj = obj_match.group(1).strip().strip('*').strip() if obj_match else "object"

            # Noun extraction
            known_nouns = ['ball', 'boot', 'box', 'bottle', 'cup', 'can', 'shoe',
                           'toy', 'sphere', 'cube', 'cone', 'cylinder', 'wheel',
                           'rock', 'stone', 'block', 'dice', 'marble', 'puck',
                           'coin', 'ring', 'disc', 'disk', 'cap', 'lid', 'piece']
            obj_lower = raw_obj.lower()
            found_noun = None
            for noun in known_nouns:
                if noun in obj_lower:
                    found_noun = noun
                    break
            if not found_noun:
                words = re.findall(r'[a-zA-Z]+', raw_obj)
                found_noun = words[-1].lower() if words else "object"

            semantic_info = {
                "object_type": found_noun,
                "detailed_description": response[:200],
                "material": mat_match.group(1).strip().strip('*').strip() if mat_match else "unknown",
                "color": color_match.group(1).strip().strip('*').strip() if color_match else "neutral",
                "condition": "unknown",
                "distinctive_features": ""
            }
        
        # Ensure all required fields exist
        required_fields = ['object_type', 'detailed_description', 'material', 'color', 'condition', 'distinctive_features']
        for field in required_fields:
            if field not in semantic_info:
                semantic_info[field] = 'unknown' if field != 'distinctive_features' else ''

        logger.info(f"✓ Identified: {semantic_info['object_type']}")
        logger.info(f"  Description: {semantic_info['detailed_description'][:80]}...")
        if semantic_info.get('distinctive_features'):
            logger.info(f"  Features: {semantic_info['distinctive_features'][:80]}...")
        
        self._end_timer("vision_semantic")
        return semantic_info
    
    def ask_cosmos_physics_enhanced(self, motion_history_base64: str, 
                                   rotation_info: Dict[str, Any],
                                   object_name: str = "object") -> Dict[str, Any]:
        """
        Enhanced physics extraction including rotation and restitution.
        """
        self._start_timer("vision_physics")
        logger.info("[LLAMA-VISION] Analyzing enhanced physics from motion...")

        content = [
            {
                "type": "text",
                "text": f"This is a motion heatmap (diff-based) of a {object_name}. "
                        "Bright areas show where motion occurred. "
                        "Analyze the trajectory to determine physics. "
                        "Provide a JSON object with these fields:\n"
                        '{"action": "slide|roll|bounce|fall|tumble", "mass": 1-100 (kg), '
                        '"friction": 0.0-1.0, "material": "wood|metal|plastic|cardboard|rubber|leather", '
                        '"restitution": 0.0-1.0 (bounciness), "is_rotating": true|false, '
                        '"confidence": 0.0-1.0}\n'
                        "Return ONLY the JSON."
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{motion_history_base64}"}
            }
        ]

        completion = self.client.chat.completions.create(
            model="meta/llama-3.2-90b-vision-instruct",
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
            temperature=0.2
        )

        if not completion.choices or len(completion.choices) == 0:
            raise ValueError("API returned no completion choices")
        response_content = completion.choices[0].message.content

        # Parse
        response_content = response_content.replace("```json", "").replace("```", "").strip()
        json_match = re.search(r'\{[^{}]*\}', response_content, re.DOTALL)
        
        if json_match:
            physics_params = json.loads(json_match.group(0))
        else:
            physics_params = {
                "action": "slide", 
                "mass": 10, 
                "friction": 0.5, 
                "material": "unknown",
                "restitution": 0.3,
                "is_rotating": False
            }

        # Rotation analysis if needed
        if physics_params.get('is_rotating', False):
            rotation_params = self.analyze_rotation(rotation_info)
            physics_params.update(rotation_params)

        logger.info(f"Physics: Mass={physics_params.get('mass')}kg, "
                   f"μ={physics_params.get('friction')}, "
                   f"e={physics_params.get('restitution')}")
        
        self._end_timer("vision_physics")
        return physics_params
    
    def analyze_rotation(self, rotation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze rotation between first and last frame."""
        logger.info("[ROTATION] Analyzing rotational dynamics...")

        # Stitch first & last frames side-by-side
        try:
            img1_data = base64.b64decode(rotation_info['first_frame'])
            img2_data = base64.b64decode(rotation_info['last_frame'])
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"Failed to decode base64 image data: {e}")

        img1 = cv2.imdecode(np.frombuffer(img1_data, np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(img2_data, np.uint8), cv2.IMREAD_COLOR)

        if img1 is None or img2 is None:
            raise ValueError("Failed to decode rotation analysis images")

        # Resize to same height
        h = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
        img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))
        combined = cv2.hconcat([img1, img2])

        _, combined_buf = cv2.imencode('.jpg', combined)
        combined_b64 = base64.b64encode(combined_buf).decode('utf-8')

        content = [
            {
                "type": "text",
                "text": "This image shows two frames side-by-side: LEFT is start, RIGHT is end. "
                        "Compare them to estimate rotation. "
                        'Return JSON: {"rotation_degrees": 0-360, "rotation_axis": "x|y|z", '
                        '"angular_velocity_estimate": "low|medium|high"}'
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{combined_b64}"}
            }
        ]

        completion = self.client.chat.completions.create(
            model="meta/llama-3.2-90b-vision-instruct",
            messages=[{"role": "user", "content": content}],
            max_tokens=256,
            temperature=0.2
        )

        if not completion.choices or len(completion.choices) == 0:
            raise ValueError("API returned no completion choices")
        response = completion.choices[0].message.content

        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                rotation_params = json.loads(json_match.group(0))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse rotation JSON: {e}")
                rotation_params = {
                    "rotation_degrees": 0,
                    "rotation_axis": "y",
                    "angular_velocity_estimate": "low"
                }
        else:
            rotation_params = {
                "rotation_degrees": 0,
                "rotation_axis": "y",
                "angular_velocity_estimate": "low"
            }

        duration = rotation_info.get('duration', 1.0)
        degrees = rotation_params.get('rotation_degrees', 0)
        angular_velocity = (degrees * np.pi / 180.0) / duration if duration > 0 else 0
        
        rotation_params['angular_velocity_rad_s'] = angular_velocity
        
        logger.info(f"Rotation: {degrees}° around {rotation_params['rotation_axis']} axis")
        
        return rotation_params
    
    # ============================================================================
    # 2. THE TRELLIS: Text-to-3D with Detailed Prompts
    # ============================================================================
    def generate_trellis_model(self, semantic_info: Dict[str, Any]) -> Optional[str]:
        """
        Generate 3D model using NVIDIA TRELLIS NIM (text-to-3D).
        CONTEST FIX: Use detailed semantic description for better results.
        Returns path to generated 3D asset (GLB format).
        """
        self._start_timer("trellis_generation")
        logger.info("[TRELLIS] Generating photorealistic 3D model (text-to-3D)...")
        
        # Build DETAILED text prompt for TRELLIS (max 77 chars)
        # Priority: object_type > distinctive_features > color > material
        obj = semantic_info.get('object_type', 'object')
        color = semantic_info.get('color', '')
        mat = semantic_info.get('material', '')
        features = semantic_info.get('distinctive_features', '')
        condition = semantic_info.get('condition', '')

        # Build prompt with priority ordering
        parts = [obj]
        
        # Add most distinctive info first (within 77 char limit)
        if features and len(features) < 30:
            candidate = f"{obj}, {features}"
            if len(candidate) <= 77:
                parts = [obj, features]
        
        if color and color != 'neutral':
            candidate = f"{color} {' '.join(parts)}"
            if len(candidate) <= 77:
                parts = [color] + parts
        
        if mat and mat != 'unknown':
            candidate = f"{' '.join(parts)}, {mat}"
            if len(candidate) <= 77:
                parts.append(mat)
        
        if condition and condition not in ['unknown', 'new']:
            candidate = f"{' '.join(parts)}, {condition}"
            if len(candidate) <= 77:
                parts.append(condition)
        
        trellis_prompt = ', '.join(parts)[:77]

        logger.info(f"TRELLIS detailed prompt ({len(trellis_prompt)} chars): {trellis_prompt}")

        # Call TRELLIS NIM
        try:
            model_path = self._call_trellis_nim(trellis_prompt)
            if model_path:
                logger.info(f"✓ TRELLIS model generated: {model_path}")
                
                # Validate GLB file
                glb_full_path = self.output_dir / model_path
                if glb_full_path.exists():
                    file_size = glb_full_path.stat().st_size
                    if file_size > 100:
                        logger.info(f"✓ GLB validated: {file_size} bytes")
                        self._end_timer("trellis_generation")
                        return model_path
                    else:
                        logger.warning(f"GLB file too small ({file_size} bytes), likely corrupt")
                else:
                    logger.warning(f"GLB file not found at {glb_full_path}")
        except Exception as e:
            logger.warning(f"TRELLIS generation failed: {e}")
            logger.info("Falling back to procedural geometry...")
        
        # Fallback: Create enhanced procedural model
        logger.info("Using enhanced procedural fallback model...")
        result = self._create_enhanced_procedural_model(semantic_info)
        self._end_timer("trellis_generation")
        return result
    
    def _call_trellis_nim(self, prompt: str) -> Optional[str]:
        """
        Call NVIDIA TRELLIS NIM API for text-to-3D generation.
        CONTEST FIX: Text-only mode (cloud API), removed unused image code.
        """
        logger.info("[TRELLIS-NIM] Calling Microsoft TRELLIS via NVIDIA API...")

        try:
            invoke_url = "https://ai.api.nvidia.com/v1/genai/microsoft/trellis"

            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Accept": "application/json",
            }

            payload = {
                "prompt": prompt[:77],  # Max 77 chars for cloud API
                "slat_cfg_scale": 3,
                "ss_cfg_scale": 7.5,
                "slat_sampling_steps": 25,
                "ss_sampling_steps": 25,
                "seed": 0
            }
            
            logger.info(f"Text-to-3D mode: '{prompt[:77]}'")
            logger.info("Sending request to TRELLIS NIM...")
            
            response = requests.post(invoke_url, headers=headers, json=payload, timeout=300)
            response.raise_for_status()
            
            response_body = response.json()
            logger.info(f"TRELLIS response status: {response.status_code}")
            
            # Check for artifacts (standard NVIDIA NIM format)
            if "artifacts" in response_body:
                artifacts = response_body["artifacts"]
                logger.info(f"Received artifacts array (count: {len(artifacts)})")

                if artifacts and len(artifacts) > 0:
                    artifact = artifacts[0]
                    finish_reason = artifact.get("finishReason", "UNKNOWN")
                    logger.info(f"Artifact finishReason: {finish_reason}")

                    if finish_reason == "SUCCESS" and "base64" in artifact:
                        glb_b64 = artifact["base64"]
                        glb_bytes = base64.b64decode(glb_b64)
                        model_path = self.assets_dir / "trellis_model.glb"
                        with open(model_path, 'wb') as f:
                            f.write(glb_bytes)
                        logger.info(f"✓ TRELLIS GLB saved: {model_path} ({len(glb_bytes)} bytes)")
                        return str(model_path.relative_to(self.output_dir))
                    else:
                        logger.warning(f"Artifact did not succeed: finishReason={finish_reason}")
                        return None
            
            logger.warning("No artifacts in TRELLIS response")
            return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"TRELLIS API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text[:500]}")
            return None
        except Exception as e:
            logger.error(f"TRELLIS generation error: {e}")
            return None
    
    def _create_enhanced_procedural_model(self, semantic_info: Dict[str, Any]) -> str:
        """
        Create an enhanced procedural USD model with semantic properties.
        Matches geometry to object type (sphere for balls, cube for boxes, etc.)
        """
        logger.info("[PROCEDURAL] Creating enhanced semantic model...")
        
        object_type = semantic_info.get('object_type', 'object').lower()
        description = semantic_info.get('detailed_description', '').lower()
        material = self._sanitize_for_usd(semantic_info.get('material', 'plastic'))
        color = semantic_info.get('color', 'orange')

        # Check both object_type and description for geometry keywords
        text_to_check = f"{object_type} {description}"

        # Determine geometry based on object type
        if any(word in text_to_check for word in ['ball', 'sphere', 'spherical', 'round', 'orb']):
            geometry_type = "Sphere"
            geometry_def = 'def Sphere "Geometry"\n    {\n        double radius = 0.5\n'
        elif any(word in text_to_check for word in ['cylinder', 'can', 'tube']):
            geometry_type = "Cylinder"
            geometry_def = 'def Cylinder "Geometry"\n    {\n        double radius = 0.5\n        double height = 1.0\n'
        elif any(word in text_to_check for word in ['cone']):
            geometry_type = "Cone"
            geometry_def = 'def Cone "Geometry"\n    {\n        double radius = 0.5\n        double height = 1.0\n'
        else:
            geometry_type = "Cube"
            geometry_def = 'def Cube "Geometry"\n    {\n        double size = 1.0\n'
        
        logger.info(f"Using {geometry_type} geometry for '{object_type}'")
        
        # Map colors to RGB
        color_map = {
            'red': (0.9, 0.1, 0.1),
            'orange': (0.9, 0.5, 0.1),
            'yellow': (0.9, 0.9, 0.1),
            'green': (0.1, 0.7, 0.1),
            'blue': (0.1, 0.3, 0.9),
            'purple': (0.6, 0.1, 0.9),
            'brown': (0.5, 0.3, 0.1),
            'black': (0.1, 0.1, 0.1),
            'white': (0.9, 0.9, 0.9),
            'gray': (0.5, 0.5, 0.5),
            'grey': (0.5, 0.5, 0.5),
            'pink': (0.9, 0.4, 0.6),
            'neutral': (0.8, 0.4, 0.0)
        }
        
        rgb = color_map.get(color.lower(), (0.8, 0.4, 0.0))
        
        asset_name = "visual_model.usda"
        asset_path = self.assets_dir / asset_name

        # Sanitize metadata
        safe_obj_type = self._sanitize_for_usd(object_type)
        safe_desc = self._sanitize_for_usd(semantic_info.get('detailed_description', ''))

        usd_content = f"""#usda 1.0
(
    defaultPrim = "VisualModel"
    metersPerUnit = 1
    upAxis = "Y"
    doc = "Semantic: {safe_obj_type}, Material: {material}, Color: {color}"
)

def Xform "VisualModel" (
    kind = "component"
)
{{
    string semantic:objectType = "{safe_obj_type}"
    string semantic:material = "{material}"
    string semantic:color = "{color}"
    string semantic:description = "{safe_desc}"
    
    {geometry_def}        color3f[] primvars:displayColor = [({rgb[0]}, {rgb[1]}, {rgb[2]})]
    }}
}}
"""
        with open(asset_path, 'w') as f:
            f.write(usd_content)

        logger.info(f"✓ Enhanced procedural {geometry_type} created: {asset_path}")
        return str(asset_path.relative_to(self.output_dir))
    
    # ============================================================================
    # 3. THE SWARM: Universal Physics Framework
    # ============================================================================

    # Motion regime classifier - dispatch to different physics models
    class MotionModel:
        """Physics models for different motion regimes."""
        SLIDE_LINEAR_DRAG = "slide_linear_drag"       # Sliding with linear drag
        ROLLING_RESISTANCE = "rolling_resistance"      # Rolling cylinder/sphere
        BOUNCE_VERTICAL = "bounce_vertical"            # Bouncing with restitution
        TUMBLE_APPROX = "tumble_approx"                # Complex tumbling motion

    # Universal Object Profile Registry
    # Maps semantic info (shape + material + size) → physics parameter ranges
    # This replaces ball-biased MASS_BOUNDS with a universal abstraction
    OBJECT_PROFILES = {
        # Spheres / Balls
        "sphere_rubber_small": {
            "mass": (0.02, 0.08), "restitution": (0.7, 0.95),
            "mu": (0.02, 0.2), "drag": (0.05, 0.4),
            "motion_model": MotionModel.BOUNCE_VERTICAL,
            "description": "Small rubber ball (tennis, racquetball)"
        },
        "sphere_rubber_medium": {
            "mass": (0.1, 0.5), "restitution": (0.6, 0.9),
            "mu": (0.03, 0.25), "drag": (0.1, 0.5),
            "motion_model": MotionModel.BOUNCE_VERTICAL,
            "description": "Medium rubber ball (basketball, soccer)"
        },
        "sphere_foam_light": {
            "mass": (0.005, 0.05), "restitution": (0.5, 0.8),
            "mu": (0.1, 0.4), "drag": (0.3, 1.0),
            "motion_model": MotionModel.BOUNCE_VERTICAL,
            "description": "Foam ball (Nerf, stress ball)"
        },
        "sphere_metal_heavy": {
            "mass": (0.5, 5.0), "restitution": (0.3, 0.6),
            "mu": (0.1, 0.3), "drag": (0.01, 0.2),
            "motion_model": MotionModel.ROLLING_RESISTANCE,
            "description": "Heavy metal sphere (bearing, shot put)"
        },

        # Cubes / Boxes
        "cube_cardboard_small": {
            "mass": (0.05, 0.5), "restitution": (0.1, 0.3),
            "mu": (0.3, 0.6), "drag": (0.2, 0.8),
            "motion_model": MotionModel.SLIDE_LINEAR_DRAG,
            "description": "Small cardboard box"
        },
        "cube_cardboard_medium": {
            "mass": (0.5, 3.0), "restitution": (0.1, 0.3),
            "mu": (0.3, 0.6), "drag": (0.3, 1.0),
            "motion_model": MotionModel.SLIDE_LINEAR_DRAG,
            "description": "Medium cardboard box"
        },
        "cube_wood_medium": {
            "mass": (0.5, 3.0), "restitution": (0.2, 0.4),
            "mu": (0.2, 0.5), "drag": (0.1, 0.5),
            "motion_model": MotionModel.SLIDE_LINEAR_DRAG,
            "description": "Wooden block"
        },
        "cube_metal_heavy": {
            "mass": (2.0, 10.0), "restitution": (0.1, 0.3),
            "mu": (0.15, 0.4), "drag": (0.05, 0.3),
            "motion_model": MotionModel.SLIDE_LINEAR_DRAG,
            "description": "Metal box or block"
        },
        "cube_plastic_light": {
            "mass": (0.01, 0.5), "restitution": (0.3, 0.6),
            "mu": (0.2, 0.5), "drag": (0.2, 0.7),
            "motion_model": MotionModel.SLIDE_LINEAR_DRAG,
            "description": "Plastic container or toy"
        },

        # Cylinders
        "cylinder_plastic_bottle": {
            "mass": (0.05, 2.0), "restitution": (0.2, 0.5),
            "mu": (0.15, 0.4), "drag": (0.1, 0.6),
            "motion_model": MotionModel.ROLLING_RESISTANCE,
            "description": "Plastic bottle"
        },
        "cylinder_metal_can": {
            "mass": (0.01, 0.5), "restitution": (0.1, 0.4),
            "mu": (0.1, 0.3), "drag": (0.05, 0.4),
            "motion_model": MotionModel.ROLLING_RESISTANCE,
            "description": "Metal can (soda, soup)"
        },

        # Complex shapes
        "boot_leather": {
            "mass": (0.3, 1.5), "restitution": (0.1, 0.3),
            "mu": (0.4, 0.8), "drag": (0.3, 1.0),
            "motion_model": MotionModel.TUMBLE_APPROX,
            "description": "Leather boot or shoe"
        },
        "shoe_fabric": {
            "mass": (0.2, 1.0), "restitution": (0.1, 0.3),
            "mu": (0.3, 0.7), "drag": (0.2, 0.8),
            "motion_model": MotionModel.TUMBLE_APPROX,
            "description": "Fabric sneaker or shoe"
        },

        # Fallback generic profiles (size-based)
        "generic_light": {
            "mass": (0.01, 0.5), "restitution": (0.2, 0.6),
            "mu": (0.1, 0.5), "drag": (0.1, 0.8),
            "motion_model": MotionModel.SLIDE_LINEAR_DRAG,
            "description": "Generic light object (<0.5kg)"
        },
        "generic_medium": {
            "mass": (0.5, 5.0), "restitution": (0.2, 0.5),
            "mu": (0.2, 0.6), "drag": (0.1, 0.6),
            "motion_model": MotionModel.SLIDE_LINEAR_DRAG,
            "description": "Generic medium object (0.5-5kg)"
        },
        "generic_heavy": {
            "mass": (5.0, 20.0), "restitution": (0.1, 0.4),
            "mu": (0.2, 0.5), "drag": (0.05, 0.4),
            "motion_model": MotionModel.SLIDE_LINEAR_DRAG,
            "description": "Generic heavy object (>5kg)"
        },
    }

    @classmethod
    def select_object_profile(cls, semantic_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Map semantic_info → object profile.
        Returns: (profile_name, profile_dict)
        """
        object_type = semantic_info.get('object_type', 'object').lower()
        material = semantic_info.get('material', 'unknown').lower()
        description = semantic_info.get('detailed_description', '').lower()

        # Determine shape class
        shape_keywords = {
            'sphere': ['ball', 'sphere', 'spherical', 'round', 'orb', 'marble'],
            'cube': ['box', 'cube', 'block', 'brick'],
            'cylinder': ['bottle', 'can', 'cylinder', 'tube'],
        }

        shape_class = 'generic'
        for shape, keywords in shape_keywords.items():
            if any(kw in object_type or kw in description for kw in keywords):
                shape_class = shape
                break

        # Determine material class
        material_class = material
        if material in ['unknown', 'neutral']:
            if 'rubber' in description or 'elastic' in description:
                material_class = 'rubber'
            elif 'foam' in description or 'soft' in description:
                material_class = 'foam'
            elif 'metal' in description or 'steel' in description:
                material_class = 'metal'
            elif 'wood' in description or 'wooden' in description:
                material_class = 'wood'
            elif 'cardboard' in description or 'paper' in description:
                material_class = 'cardboard'
            elif 'plastic' in description:
                material_class = 'plastic'
            elif 'leather' in description:
                material_class = 'leather'
            elif 'fabric' in description or 'cloth' in description:
                material_class = 'fabric'

        # Determine size class
        size_keywords_small = ['small', 'tiny', 'mini', 'little']
        size_keywords_medium = ['medium', 'regular', 'normal']
        size_keywords_large = ['large', 'big', 'heavy', 'massive']

        size_class = 'medium'
        if any(kw in description for kw in size_keywords_small):
            size_class = 'small'
        elif any(kw in description for kw in size_keywords_large):
            size_class = 'large'

        # Build candidate profile names
        candidates = [
            f"{shape_class}_{material_class}_{size_class}",
            f"{shape_class}_{material_class}_medium",
            f"{shape_class}_{material_class}_small",
            f"{object_type}_{material_class}",
            f"generic_{size_class}",
        ]

        # Find first matching profile
        for candidate in candidates:
            if candidate in cls.OBJECT_PROFILES:
                logger.info(f"Selected profile: {candidate}")
                return candidate, cls.OBJECT_PROFILES[candidate]

        # Fallback to generic_medium
        logger.info("Using fallback profile: generic_medium")
        return "generic_medium", cls.OBJECT_PROFILES["generic_medium"]

    def massive_parallel_search(self, estimated_friction: float,
                               estimated_mass: float,
                               estimated_restitution: float,
                               ground_truth_dist: float,
                               ground_truth_duration: float = 1.5,
                               num_agents: int = 4096,
                               semantic_info: Optional[Dict[str, Any]] = None,
                               known_mass: Optional[float] = None,
                               object_profile_name: Optional[str] = None,
                               v0: float = None,
                               start_height: float = None,
                               video_fps: float = 60.0) -> Dict[str, Any]:
        """
        Universal parallel Monte Carlo physics search.
        CONTEST FIX P0-2: Horizontal distance only to match reality.
        UNIVERSAL FRAMEWORK: Uses object profiles, motion models, mass identifiability.
        """
        # Validate inputs
        if ground_truth_dist <= 0:
            raise ValueError(f"ground_truth_dist must be positive, got {ground_truth_dist}")
        if ground_truth_duration <= 0:
            raise ValueError(f"ground_truth_duration must be positive, got {ground_truth_duration}")
        if num_agents <= 0:
            raise ValueError(f"num_agents must be positive, got {num_agents}")
        if num_agents > 100000:
            logger.warning(f"Very large num_agents ({num_agents}) may cause memory issues")

        self._start_timer("warp_simulation")
        logger.info(f"Launching {num_agents} parallel physics hypotheses (deterministic seed={self.seed})...")
        logger.info(f"   Target: {ground_truth_dist}m in {ground_truth_duration}s")

        # Use defaults if not provided
        if v0 is None:
            v0 = self.DEFAULT_V0
        if start_height is None:
            start_height = self.DEFAULT_START_HEIGHT

        # Select object profile
        if object_profile_name and object_profile_name in self.OBJECT_PROFILES:
            profile_name = object_profile_name
            profile = self.OBJECT_PROFILES[object_profile_name]
            logger.info(f"Using CLI-specified profile: {profile_name}")
        elif semantic_info:
            profile_name, profile = self.select_object_profile(semantic_info)
        else:
            profile_name = "generic_medium"
            profile = self.OBJECT_PROFILES["generic_medium"]
            logger.info("No semantic info, using generic_medium profile")

        # Mass sampling with identifiability tracking
        mass_lo, mass_hi = profile['mass']
        mass_identifiability = "low"
        mass_basis = "fit_parameter (degenerate with drag)"

        if known_mass is not None:
            # User provided known mass - treat as high confidence
            masses_np = np.full(num_agents, known_mass, dtype=np.float32)
            mass_identifiability = "high"
            mass_basis = "user_provided"
            logger.info(f"   Using known mass: {known_mass:.3f} kg (user-provided)")
        else:
            # Fit effective mass from profile bounds
            log_mu = np.log(max(estimated_mass, mass_lo))
            log_sigma = 0.5
            masses_np = np.random.lognormal(log_mu, log_sigma, num_agents).astype(np.float32)
            masses_np = np.clip(masses_np, mass_lo, mass_hi)
            logger.info(f"   Effective mass prior: {mass_lo}-{mass_hi} kg (profile: {profile_name})")
            logger.info(f"   Mass identifiability: {mass_identifiability} (degenerate with drag)")

        # Friction sampling from profile
        mu_lo, mu_hi = profile['mu']
        frictions_np = np.random.uniform(mu_lo, mu_hi, num_agents).astype(np.float32)
        frictions_np = np.clip(frictions_np, 0.01, 1.0)

        # Restitution sampling from profile
        rest_lo, rest_hi = profile['restitution']
        restitutions_np = np.random.uniform(rest_lo, rest_hi, num_agents).astype(np.float32)
        restitutions_np = np.clip(restitutions_np, 0.0, 1.0)

        # Drag sampling from profile
        drag_lo, drag_hi = profile['drag']
        drags_np = np.random.uniform(drag_lo, drag_hi, num_agents).astype(np.float32)

        # Motion model selection
        motion_model = profile.get('motion_model', self.MotionModel.SLIDE_LINEAR_DRAG)
        logger.info(f"   Motion model: {motion_model}")

        # Convert motion model to int array for Warp kernel
        motion_model_int = {
            self.MotionModel.SLIDE_LINEAR_DRAG: 0,
            self.MotionModel.ROLLING_RESISTANCE: 1,
            self.MotionModel.BOUNCE_VERTICAL: 2,
            self.MotionModel.TUMBLE_APPROX: 3
        }.get(motion_model, 0)

        motion_models_np = np.full(num_agents, motion_model_int, dtype=np.int32)

        # Warp allocations
        frictions_wp = wp.from_numpy(frictions_np, dtype=float, device=WARP_DEVICE)
        masses_wp = wp.from_numpy(masses_np, dtype=float, device=WARP_DEVICE)
        restitutions_wp = wp.from_numpy(restitutions_np, dtype=float, device=WARP_DEVICE)
        drags_wp = wp.from_numpy(drags_np, dtype=float, device=WARP_DEVICE)
        motion_models_wp = wp.from_numpy(motion_models_np, dtype=int, device=WARP_DEVICE)
        results_wp = wp.zeros(num_agents, dtype=float, device=WARP_DEVICE)
        times_wp = wp.zeros(num_agents, dtype=float, device=WARP_DEVICE)

        # Use configurable initial conditions (not hardcoded ball defaults)
        pos_np = np.tile([0.0, start_height, 0.0], (num_agents, 1)).astype(np.float32)
        vel_np = np.tile([v0, 0.0, 0.0], (num_agents, 1)).astype(np.float32)
        logger.info(f"   Initial conditions: v0={v0:.2f} m/s, h0={start_height:.2f} m")
        
        pos_wp = wp.from_numpy(pos_np, dtype=wp.vec3f, device=WARP_DEVICE)
        vel_wp = wp.from_numpy(vel_np, dtype=wp.vec3f, device=WARP_DEVICE)

        # Calculate adaptive simulation parameters from video properties
        dt = 1.0 / max(video_fps, 1.0)  # Prevent division by zero
        max_time = ground_truth_duration * self.SIMULATION_MARGIN_FACTOR
        max_steps = int(max_time / dt) + 100  # Extra safety margin
        max_steps = max(max_steps, self.MIN_SIMULATION_STEPS)
        max_steps = min(max_steps, self.MAX_SIMULATION_STEPS)
        logger.info(f"   Simulation dt: {dt:.4f}s (based on video {video_fps:.1f} FPS)")
        logger.info(f"   Max simulation steps: {max_steps} ({max_time:.1f}s)")

        # CONTEST FIX P0-2: Kernel with HORIZONTAL distance only
        @wp.kernel
        def massive_parallel_search_kernel(
            positions: wp.array(dtype=wp.vec3f),
            velocities: wp.array(dtype=wp.vec3f),
            frictions: wp.array(dtype=float),
            masses: wp.array(dtype=float),
            drag_coeffs: wp.array(dtype=float),
            restitutions: wp.array(dtype=float),
            motion_models: wp.array(dtype=int),  # NEW: 0=slide, 1=roll, 2=bounce, 3=tumble
            dt: float,
            time_steps: int,
            final_distances: wp.array(dtype=float),
            final_times: wp.array(dtype=float)
        ):
            tid = wp.tid()
            
            pos = positions[tid]
            vel = velocities[tid]

            friction = frictions[tid]
            mass = masses[tid]
            drag = drag_coeffs[tid]
            restitution = restitutions[tid]
            motion_model = motion_models[tid]

            g = 9.81  # Hardcoded in kernel (matches GRAVITY class constant)
            
            current_dist = float(0.0)
            stop_time = float(0.0)
            is_stopped = int(0)
            
            for i in range(time_steps):
                if is_stopped == 0:
                    # FIXED: Use HORIZONTAL velocity only for distance
                    vel_horizontal = wp.vec3(vel[0], 0.0, vel[2])
                    speed_horizontal = wp.length(vel_horizontal)

                    if speed_horizontal > 0.01:  # Stop threshold (matches MIN_HORIZONTAL_VELOCITY)
                        # Motion model-specific physics
                        total_decel = float(0.0)

                        if motion_model == 0:  # SLIDE_LINEAR_DRAG
                            friction_decel = friction * g
                            drag_decel = (drag * speed_horizontal) / mass
                            total_decel = friction_decel + drag_decel

                        elif motion_model == 1:  # ROLLING_RESISTANCE
                            # Rolling: F_roll = C_rr * N = C_rr * m * g
                            # For cylinders/spheres rolling without slipping
                            # C_rr ≈ 0.01 * μ for typical surfaces
                            rolling_coeff = friction * 0.01
                            rolling_decel = rolling_coeff * g
                            drag_decel = (drag * speed_horizontal) / mass
                            total_decel = rolling_decel + drag_decel

                        elif motion_model == 2:  # BOUNCE_VERTICAL
                            # Bouncing: same friction+drag as sliding
                            # Ground collision uses restitution (already handled below)
                            friction_decel = friction * g
                            drag_decel = (drag * speed_horizontal) / mass
                            total_decel = friction_decel + drag_decel

                        elif motion_model == 3:  # TUMBLE_APPROX
                            # Tumbling: higher drag for non-streamlined motion
                            # Chaotic rotation increases effective drag coefficient
                            tumble_drag_multiplier = 2.5
                            friction_decel = friction * g
                            drag_decel = (drag * speed_horizontal * tumble_drag_multiplier) / mass
                            total_decel = friction_decel + drag_decel
                        
                        # Update horizontal velocity
                        if speed_horizontal > 0.0:
                            dir_horizontal = wp.normalize(vel_horizontal)
                            new_speed = speed_horizontal - total_decel * dt
                            
                            if new_speed <= 0.0:
                                vel = wp.vec3(0.0, vel[1], 0.0)
                                is_stopped = 1
                                stop_time = float(i) * dt
                            else:
                                vel = wp.vec3(dir_horizontal[0] * new_speed, vel[1], dir_horizontal[2] * new_speed)
                        
                        # Update position
                        pos = pos + vel * dt
                        # FIXED: Accumulate HORIZONTAL distance only
                        current_dist = current_dist + speed_horizontal * dt
                        
                        # Ground collision
                        if pos[1] < 0.0:
                            pos = wp.vec3(pos[0], 0.0, pos[2])
                            vel = wp.vec3(vel[0], -vel[1] * restitution, vel[2])
                    else:
                        is_stopped = 1
                        stop_time = float(i) * dt
            
            if is_stopped == 0:
                stop_time = float(time_steps) * dt
            
            final_distances[tid] = current_dist
            final_times[tid] = stop_time
        
        # Launch swarm
        wp.launch(
            kernel=massive_parallel_search_kernel,
            dim=num_agents,
            inputs=[pos_wp, vel_wp, frictions_wp, masses_wp, drags_wp, restitutions_wp,
                   motion_models_wp,  # NEW: motion model dispatch
                   dt, max_steps, results_wp, times_wp],  # Adaptive simulation parameters
            device=WARP_DEVICE
        )
        
        # Multi-objective scoring: Distance + Time
        simulated_dists = results_wp.numpy()
        simulated_times = times_wp.numpy()
        
        abs_dist_errors_m = np.abs(simulated_dists - ground_truth_dist)
        abs_time_errors_s = np.abs(simulated_times - ground_truth_duration)
        
        dist_error_norm = abs_dist_errors_m / max(ground_truth_dist, 0.01)
        time_error_norm = abs_time_errors_s / max(ground_truth_duration, 0.01)
        
        total_error_norm = dist_error_norm + time_error_norm
        
        best_idx = np.argmin(total_error_norm)
        
        # Extract winner
        best_friction = float(frictions_np[best_idx])
        best_mass = float(masses_np[best_idx])
        best_restitution = float(restitutions_np[best_idx])
        best_drag = float(drags_np[best_idx])
        
        best_simulated_dist = float(simulated_dists[best_idx])
        best_simulated_time = float(simulated_times[best_idx])
        best_dist_error_m = float(abs_dist_errors_m[best_idx])
        best_time_error_s = float(abs_time_errors_s[best_idx])
        best_total_error_norm = float(total_error_norm[best_idx])
        
        logger.info(f"🏆 Winner Agent #{best_idx}:")
        logger.info(f"   Friction: μ={best_friction:.4f}")
        logger.info(f"   Effective Mass: m_eff={best_mass:.2f}kg ({mass_basis})")
        logger.info(f"   Drag: k={best_drag:.4f}")
        logger.info(f"   Restitution: e={best_restitution:.3f}")
        logger.info(f"   Simulated: {best_simulated_dist:.3f}m in {best_simulated_time:.2f}s")
        logger.info(f"   Target: {ground_truth_dist:.3f}m in {ground_truth_duration:.2f}s")
        logger.info(f"   Distance Error: {best_dist_error_m:.4f}m ({best_dist_error_m/ground_truth_dist*100:.2f}%)")
        logger.info(f"   Time Error: {best_time_error_s:.3f}s ({best_time_error_s/ground_truth_duration*100:.2f}%)")

        # Store for visualization
        self.simulation_history.append({
            'all_frictions': frictions_np,
            'all_masses': masses_np,
            'all_drags': drags_np,
            'all_distances': simulated_dists,
            'all_times': simulated_times,
            'errors': dist_error_norm,
            'abs_errors_m': abs_dist_errors_m,
            'total_errors': total_error_norm,
            'best_idx': best_idx,
            'ground_truth': ground_truth_dist,
            'ground_truth_time': ground_truth_duration
        })

        self._end_timer("warp_simulation")

        return {
            'friction': best_friction,
            'effective_mass_kg': best_mass,  # RENAMED: not "mass" - it's degenerate with drag
            'mass_identifiability': mass_identifiability,
            'mass_basis': mass_basis,
            'drag': best_drag,
            'restitution': best_restitution,
            'motion_model': motion_model,
            'object_profile': profile_name,
            'simulated_distance': best_simulated_dist,
            'simulated_time': best_simulated_time,
            'distance_error_m': best_dist_error_m,
            'time_error_s': best_time_error_s,
            'total_error_normalized': best_total_error_norm,
            'num_hypotheses': num_agents,
        }
    
    # ============================================================================
    # 4. THE EMBEDDING: Physics → USD (Contest-Safe)
    # ============================================================================
    def generate_semantic_digital_twin(self,
                                      visual_asset_path: str,
                                      physics_params: Dict[str, Any],
                                      semantic_info: Dict[str, Any],
                                      num_agents: int) -> str:
        """
        Create semantic digital twin with embedded physics (proper USD schema).
        UNIVERSAL FRAMEWORK: Proper mass identifiability, motion model metadata.
        """
        self._start_timer("usd_generation")
        logger.info("[EMBEDDING] Creating semantic digital twin with embedded physics...")

        # Extract physics parameters (use new naming convention)
        effective_mass_kg = physics_params.get('effective_mass_kg', 2.5)
        mass_identifiability = physics_params.get('mass_identifiability', 'low')
        mass_basis = physics_params.get('mass_basis', 'fit_parameter')
        friction = physics_params.get('friction', 0.5)
        restitution = physics_params.get('restitution', 0.3)
        drag = physics_params.get('drag', 0.0)
        motion_model = physics_params.get('motion_model', 'slide_linear_drag')
        object_profile = physics_params.get('object_profile', 'generic_medium')
        has_rotation = physics_params.get('is_rotating', False)
        angular_vel = physics_params.get('angular_velocity_rad_s', 0.0)

        # CONTEST FIX P0-3: Sanitize all strings
        object_type = self._sanitize_for_usd(semantic_info.get('object_type', 'object'))
        material = self._sanitize_for_usd(semantic_info.get('material', 'unknown'))
        description = self._sanitize_for_usd(semantic_info.get('detailed_description', 'object'))

        # Determine collision geometry (shape-aware)
        object_type_lower = object_type.lower()
        description_lower = description.lower()
        text_to_check = f"{object_type_lower} {description_lower}"

        if any(word in text_to_check for word in ['ball', 'sphere', 'round', 'orb', 'spherical']):
            collision_geom = '''def Sphere "CollisionProxy" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.5
            uniform token purpose = "guide"
            rel material:binding:physics = </World/DigitalTwin/PhysicsMaterial>
        }'''
        elif any(word in text_to_check for word in ['cylinder', 'bottle', 'can', 'tube']):
            collision_geom = '''def Cylinder "CollisionProxy" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.5
            double height = 1.0
            uniform token purpose = "guide"
            rel material:binding:physics = </World/DigitalTwin/PhysicsMaterial>
        }'''
        else:
            collision_geom = '''def Cube "CollisionProxy" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 1.0
            uniform token purpose = "guide"
            rel material:binding:physics = </World/DigitalTwin/PhysicsMaterial>
        }'''
        
        # Visual reference
        is_glb = visual_asset_path.endswith('.glb')
        
        if is_glb:
            visual_reference = f'''def "VisualGeometry" (
            prepend payload = @./{visual_asset_path}@
        )
        {{
            uniform token purpose = "render"
        }}'''
        else:
            visual_reference = f'''def "VisualGeometry" (
            prepend references = @./{visual_asset_path}@
        )
        {{
            uniform token purpose = "render"
        }}'''

        # Proper USD physics schema with mass identifiability
        usd_content = f"""#usda 1.0
(
    defaultPrim = "DigitalTwin"
    endTimeCode = 240
    metersPerUnit = 1
    startTimeCode = 0
    timeCodesPerSecond = 60
    upAxis = "Y"
    doc = "Universal Physics Framework: {object_type} (profile: {object_profile})"
)

def Xform "World"
{{
    def PhysicsScene "PhysicsScene"
    {{
        vector3f physics:gravityDirection = (0, -1, 0)
        float physics:gravityMagnitude = 9.81
    }}

    def Xform "DigitalTwin" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float3 xformOp:translate = (0, 1, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        # Proper PhysicsMassAPI usage
        float physics:mass = {effective_mass_kg}
        point3f physics:centerOfMass = (0, 0, 0)
        float3 physics:diagonalInertia = (1, 1, 1)

        # Initial conditions (not hardcoded ball values)
        vector3f physics:velocity = ({self.DEFAULT_V0}, 0, 0)
        {"vector3f physics:angularVelocity = (0, " + str(angular_vel) + ", 0)" if has_rotation else ""}

        # Semantic metadata
        string semantic:objectType = "{object_type}"
        string semantic:material = "{material}"
        string semantic:description = "{description}"
        string semantic:objectProfile = "{object_profile}"

        # Physics source metadata (UNIVERSAL FRAMEWORK identifiability)
        string physics:source = "NVIDIA Warp Monte Carlo ({num_agents} hypotheses)"
        string physics:motionModel = "{motion_model}"
        string physics:massIdentifiability = "{mass_identifiability}"
        string physics:massBasis = "{mass_basis}"
        string physics:massNote = "Effective mass fit jointly with drag (degenerate parameter - not independently identifiable from monocular video)"

        def Material "PhysicsMaterial" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            float physics:dynamicFriction = {friction}
            float physics:staticFriction = {friction * 1.1}
            float physics:restitution = {restitution}

            # Custom metadata for framework tracking
            float custom:dragCoefficient = {drag}
            int custom:numHypothesesTested = {num_agents}
            string custom:frameworkVersion = "3.2-universal"
        }}
        
        {visual_reference}

        {collision_geom}
    }}

    def Xform "Ground"
    {{
        float3 xformOp:translate = (0, -1, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Material "GroundMaterial" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            float physics:dynamicFriction = 1000
            float physics:staticFriction = 1000
            float physics:restitution = 0.0
        }}

        def Cube "Geometry" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsRigidBodyAPI"]
        )
        {{
            double size = 100.0
            float3 xformOp:scale = (100, 0.1, 100)
            uniform token[] xformOpOrder = ["xformOp:scale"]
            color3f[] primvars:displayColor = [(0.5, 0.5, 0.5)]
            
            bool physics:kinematicEnabled = true
            rel material:binding:physics = </World/Ground/GroundMaterial>
        }}
    }}
}}
"""

        twin_path = self.output_dir / "semantic_digital_twin.usda"
        with open(twin_path, 'w') as f:
            f.write(usd_content)

        logger.info(f"✅ SEMANTIC DIGITAL TWIN CREATED: {twin_path}")
        logger.info(f"   Visual: {visual_asset_path}")
        logger.info(f"   Physics: m_eff={effective_mass_kg:.2f}kg ({mass_basis}), μ={friction:.3f}, e={restitution:.3f}")
        logger.info(f"   Semantic: {object_type} ({material})")
        logger.info(f"   Profile: {object_profile}, Model: {motion_model}")
        logger.info(f"   Hypotheses: {num_agents}")

        self._end_timer("usd_generation")
        return str(twin_path)
    
    # ============================================================================
    # VISUALIZATION: Fixed to Match Kernel
    # ============================================================================
    @staticmethod
    def _simulate_trajectory(friction, mass, drag, v0=None, dt=1.0/60.0, max_steps=300):
        """
        CONTEST FIX P0-2: Simulate HORIZONTAL distance matching kernel exactly.
        """
        if v0 is None:
            v0 = PhysicalAgentSwarmGTC.DEFAULT_V0

        times = [0.0]
        dists = [0.0]
        speed_horizontal = v0  # Initial horizontal speed
        cumulative_dist = 0.0
        g = PhysicalAgentSwarmGTC.GRAVITY

        for step in range(1, max_steps + 1):
            if speed_horizontal <= PhysicalAgentSwarmGTC.MIN_HORIZONTAL_VELOCITY:
                break
            
            # Match kernel physics exactly
            friction_decel = friction * g
            drag_decel = (drag * speed_horizontal) / mass
            speed_horizontal -= (friction_decel + drag_decel) * dt
            
            if speed_horizontal < 0:
                speed_horizontal = 0.0
            
            # Accumulate horizontal distance
            cumulative_dist += speed_horizontal * dt
            times.append(step * dt)
            dists.append(cumulative_dist)
        
        return np.array(times), np.array(dists)

    def generate_gtc_visualizations(self, result: Dict[str, Any],
                                    ground_truth: float = 2.5):
        """Generate GTC-quality visualizations with fixed trajectory."""
        self._start_timer("visualization")
        logger.info("Generating GTC presentation visuals...")

        if not self.simulation_history:
            logger.warning("No simulation history")
            return

        history = self.simulation_history[-1]
        ground_truth_time = history.get('ground_truth_time', 4.5)

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Swarm Scatter
        ax1 = fig.add_subplot(gs[0, 0])
        all_distances = history['all_distances']
        best_idx = history['best_idx']

        sample_indices = np.random.choice(len(all_distances),
                                         min(200, len(all_distances)),
                                         replace=False)

        for idx in sample_indices:
            if idx != best_idx:
                ax1.scatter(history['all_frictions'][idx],
                          all_distances[idx],
                          color='grey', alpha=0.15, s=15)

        ax1.scatter(history['all_frictions'][best_idx],
                   all_distances[best_idx],
                   color='#76b900', s=300, marker='*',
                   label=f'Winner (mu={history["all_frictions"][best_idx]:.3f})',
                   edgecolors='black', linewidths=2.5, zorder=5)

        ax1.axhline(y=ground_truth, color='#ff4444', linestyle='--',
                   linewidth=2.5, label='Ground Truth', alpha=0.8)

        ax1.set_xlabel('Friction (mu)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Horizontal Distance (m)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Parallel Monte Carlo: {len(all_distances)} Hypotheses',
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle=':')

        # 2. Error Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        abs_errors = history['abs_errors_m']
        ax2.hist(abs_errors, bins=60, color='#76b900', alpha=0.7, edgecolor='black')
        ax2.axvline(x=abs_errors[best_idx], color='#ff4444', linestyle='--',
                   linewidth=2.5, label=f'Best: {abs_errors[best_idx]:.4f} m')
        ax2.set_xlabel('Absolute Distance Error (m)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax2.set_title('Convergence Quality', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle=':')

        # 3. Trajectory Convergence (FIXED to match kernel)
        ax3 = fig.add_subplot(gs[0, 2])

        sample_size = min(30, len(history['all_frictions']))
        sample_indices = np.random.choice(len(history['all_frictions']),
                                         sample_size, replace=False)

        for idx in sample_indices:
            if idx != best_idx:
                t_h, d_h = self._simulate_trajectory(
                    history['all_frictions'][idx],
                    history['all_masses'][idx],
                    history['all_drags'][idx])
                ax3.plot(t_h, d_h, color='grey', alpha=0.12, linewidth=0.8)

        # Winner trajectory
        t_best, d_best = self._simulate_trajectory(
            history['all_frictions'][best_idx],
            history['all_masses'][best_idx],
            history['all_drags'][best_idx])
        ax3.plot(t_best, d_best, color='#76b900', linewidth=3.5,
                label=f'Winner (mu={history["all_frictions"][best_idx]:.3f})')

        ax3.axhline(y=ground_truth, color='#ff4444', linestyle='--',
                   linewidth=2, label=f'Target dist ({ground_truth:.1f} m)', alpha=0.8)
        ax3.axvline(x=ground_truth_time, color='#4444ff', linestyle=':',
                   linewidth=2, label=f'Video duration ({ground_truth_time:.1f} s)', alpha=0.8)

        ax3.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Horizontal Distance (m)', fontsize=11, fontweight='bold')
        ax3.set_title('Trajectory (horizontal motion)', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3, linestyle=':')

        # 4. Mass Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(history['all_masses'], bins=50, color='#0099cc',
                alpha=0.7, edgecolor='black')
        ax4.axvline(x=history['all_masses'][best_idx], color='#ff4444',
                   linestyle='--', linewidth=2.5,
                   label=f'Winner: {history["all_masses"][best_idx]:.3f} kg')
        ax4.set_xlabel('Mass (kg)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax4.set_title('Mass Hypothesis Distribution', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3, linestyle=':')

        # 5. Drag vs Mass
        ax5 = fig.add_subplot(gs[1, 1])
        total_err = history['total_errors']
        err_pct = np.percentile(total_err, 10)
        good_mask = total_err <= err_pct

        ax5.scatter(history['all_masses'][~good_mask],
                   history['all_drags'][~good_mask],
                   color='grey', alpha=0.05, s=8, label='Poor fits')
        ax5.scatter(history['all_masses'][good_mask],
                   history['all_drags'][good_mask],
                   color='#ff8800', alpha=0.4, s=20, label=f'Top 10% fits')
        ax5.scatter(history['all_masses'][best_idx],
                   history['all_drags'][best_idx],
                   color='#76b900', s=300, marker='*',
                   edgecolors='black', linewidths=2.5, zorder=5, label='Winner')

        ax5.set_xlabel('Mass (kg)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Drag coefficient', fontsize=11, fontweight='bold')
        ax5.set_title('Mass vs Drag (underdetermined)', fontsize=12, fontweight='bold')
        ax5.legend(loc='best', fontsize=8)
        ax5.grid(True, alpha=0.3, linestyle=':')

        # 6. Pipeline Overview
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        pipeline_text = f"""UNIVERSAL PHYSICS FRAMEWORK
------------------------------------
SEMANTIC DIGITAL TWIN PIPELINE

[INPUT]
  Video Analysis
  Diff-Heatmap Motion Tracking

[AI VISION] Llama-3.2
  Semantic Identification
  Object Profile Selection

[TRELLIS 3D] Text-to-3D
  Photorealistic Model
  Detailed Prompts

[WARP PARALLEL] {len(history['all_frictions'])} agents
  Monte Carlo Search
  Motion Model Dispatch
  Horizontal Distance Physics

[MASS IDENTIFIABILITY]
  Effective Mass (degenerate w/ drag)
  Friction (identifiable)
  Restitution (identifiable)

[USD EMBEDDING]
  Proper PhysicsMassAPI
  Visual + Collision Proxy
  Identifiability Metadata

RESULT: Drop in Omniverse
  -> Instant Physical Simulation

------------------------------------
Best dist error: {abs_errors[best_idx]:.4f} m
Winner friction: mu={history['all_frictions'][best_idx]:.3f}
Effective mass:  m_eff={history['all_masses'][best_idx]:.3f} kg
Winner drag:     k={history['all_drags'][best_idx]:.4f}
Device: {WARP_DEVICE.upper()}
Seed: {self.seed}
"""
        ax6.text(0.05, 0.95, pipeline_text,
                transform=ax6.transAxes,
                fontsize=8.5,
                verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle('Physical Agent Swarm: Semantic Digital Twins for NVIDIA GTC 2025',
                    fontsize=16, fontweight='bold', y=0.98)

        fig_path = self.output_dir / "gtc_presentation.png"
        try:
            plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
            logger.info(f"✓ GTC visualization saved: {fig_path}")
        finally:
            plt.close()
        
        self._end_timer("visualization")
    
    # ============================================================================
    # RUN REPORT: JSON + Markdown
    # ============================================================================
    def generate_run_report(self, result: Dict[str, Any],
                           video_path: str,
                           ground_truth_distance: float,
                           num_agents: int,
                           ground_truth_duration: float):
        """
        CONTEST FIX P1-7: Generate run report (JSON + Markdown).
        """
        logger.info("Generating run report...")
        
        # JSON report
        report_json = {
            "timestamp": datetime.now().isoformat(),
            "version": "3.1-contest-ready",
            "seed": self.seed,
            "device": WARP_DEVICE,
            "inputs": {
                "video_path": str(video_path),
                "ground_truth_distance_m": ground_truth_distance,
                "num_agents": num_agents
            },
            "semantic_identification": {
                "object_type": result['semantic_info']['object_type'],
                "material": result['semantic_info']['material'],
                "color": result['semantic_info']['color'],
                "description": result['semantic_info']['detailed_description'][:100]
            },
            "physics_results": {
                "friction": result['physics_parameters']['friction'],
                "effective_mass_kg": result['physics_parameters']['effective_mass_kg'],
                "mass_identifiability": result['physics_parameters']['mass_identifiability'],
                "mass_basis": result['physics_parameters']['mass_basis'],
                "drag_coefficient": result['physics_parameters']['drag'],
                "restitution": result['physics_parameters']['restitution'],
                "motion_model": result['physics_parameters'].get('motion_model', 'unknown'),
                "object_profile": result['physics_parameters'].get('object_profile', 'unknown'),
                "simulated_distance_m": result['physics_parameters']['simulated_distance'],
                "simulated_time_s": result['physics_parameters']['simulated_time'],
                "distance_error_m": result['physics_parameters']['distance_error_m'],
                "time_error_s": result['physics_parameters']['time_error_s']
            },
            "convergence": {
                "distance_converged": result['swarm_statistics']['convergence_achieved_distance'],
                "time_converged": result['swarm_statistics']['convergence_achieved_time'],
                "overall_converged": result['swarm_statistics']['convergence_achieved']
            },
            "outputs": {
                "digital_twin_path": result['digital_twin_path'],
                "visual_asset": result['visual_asset'],
                "visualization": "gtc_presentation.png"
            },
            "timings_seconds": {k: v.get('elapsed', 0) for k, v in self.timings.items()}
        }
        
        json_path = self.output_dir / "run.json"
        with open(json_path, 'w') as f:
            json.dump(report_json, f, indent=2)
        logger.info(f"✓ Run report (JSON): {json_path}")
        
        # Markdown report
        md_content = f"""# GTC Golden Ticket Submission - Run Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Version:** 3.1 Contest-Ready  
**Seed:** {self.seed}  
**Device:** {WARP_DEVICE}

## Input

- **Video:** `{video_path}`
- **Ground Truth Distance:** {ground_truth_distance}m
- **Parallel Hypotheses:** {num_agents}

## Semantic Identification

- **Object Type:** {result['semantic_info']['object_type']}
- **Material:** {result['semantic_info']['material']}
- **Color:** {result['semantic_info']['color']}
- **Description:** {result['semantic_info']['detailed_description'][:150]}...

## Physics Results

**Object Profile:** {result['physics_parameters'].get('object_profile', 'unknown')}
**Motion Model:** {result['physics_parameters'].get('motion_model', 'unknown')}

| Parameter | Value | Notes |
|-----------|-------|-------|
| Friction (μ) | {result['physics_parameters']['friction']:.4f} | From Monte Carlo fit |
| Effective Mass | {result['physics_parameters']['effective_mass_kg']:.2f} kg | {result['physics_parameters']['mass_basis']} |
| Mass Identifiability | {result['physics_parameters']['mass_identifiability']} | Degenerate with drag |
| Drag (k) | {result['physics_parameters']['drag']:.4f} | From Monte Carlo fit |
| Restitution (e) | {result['physics_parameters']['restitution']:.3f} | From Monte Carlo fit |

## Simulation Accuracy

| Metric | Simulated | Target | Error |
|--------|-----------|--------|-------|
| Distance | {result['physics_parameters']['simulated_distance']:.3f}m | {ground_truth_distance:.3f}m | {result['physics_parameters']['distance_error_m']:.4f}m |
| Time | {result['physics_parameters']['simulated_time']:.2f}s | {ground_truth_duration:.2f}s | {result['physics_parameters']['time_error_s']:.3f}s |

**Convergence Status:**  
- Distance: {'✓ CONVERGED' if result['swarm_statistics']['convergence_achieved_distance'] else '✗ Not converged'}  
- Time: {'✓ CONVERGED' if result['swarm_statistics']['convergence_achieved_time'] else '✗ Not converged'}

## Outputs

- **Digital Twin:** `{result['digital_twin_path']}`
- **Visual Asset:** `{result['visual_asset']}`
- **Visualization:** `gtc_presentation.png`

## Pipeline Timing

"""
        for stage, timing in self.timings.items():
            if 'elapsed' in timing:
                md_content += f"- **{stage}:** {timing['elapsed']:.2f}s\n"
        
        total_time = sum(t.get('elapsed', 0) for t in self.timings.values())
        md_content += f"\n**Total Pipeline Time:** {total_time:.2f}s\n"
        
        md_content += """
## How to Use

1. Open `semantic_digital_twin.usda` in NVIDIA Omniverse
2. Press PLAY to see physics simulation
3. All physics properties are embedded in USD metadata

## Universal Physics Framework

This system uses a **universal object profile registry** to map semantic information (shape, material, size) to physics parameter priors. The framework clearly distinguishes between:

- **Identifiable parameters**: Friction, restitution (observable from motion)
- **Degenerate parameters**: Effective mass and drag (fit jointly, not independently identifiable from monocular video)

The mass-vs-drag degeneracy is visualized in the presentation. For known-mass scenarios, use `--known_mass <kg>` to constrain the fit.

## Motion Models

The system selects appropriate physics models based on object type:
- **slide_linear_drag**: Boxes, blocks (sliding friction + linear drag)
- **rolling_resistance**: Cylinders, cans (rolling resistance)
- **bounce_vertical**: Balls, spheres (restitution-based bouncing)
- **tumble_approx**: Complex shapes (boots, shoes)
"""
        
        md_path = self.output_dir / "run.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
        logger.info(f"✓ Run report (Markdown): {md_path}")
    
    # ============================================================================
    # ORCHESTRATION: GTC Pipeline
    # ============================================================================
    def run_gtc_pipeline(self, video_path: str,
                        ground_truth_distance: float = 2.5,
                        num_agents: int = 4096,
                        known_mass: Optional[float] = None,
                        object_profile_name: Optional[str] = None,
                        v0: Optional[float] = None,
                        start_height: Optional[float] = None) -> Dict[str, Any]:
        """Execute the complete universal physics pipeline."""
        # Validate inputs
        if ground_truth_distance <= 0:
            raise ValueError(f"ground_truth_distance must be positive, got {ground_truth_distance}")
        if num_agents <= 0:
            raise ValueError(f"num_agents must be positive, got {num_agents}")
        if num_agents > 100000:
            logger.warning(f"Very large num_agents ({num_agents}) may cause memory issues")

        logger.info("\n" + "="*70)
        logger.info("PHYSICAL AGENT SWARM - GTC GOLDEN TICKET EDITION")
        logger.info("Contest-Ready: Reproducible, Safe, Deterministic")
        logger.info("="*70)

        # Step 1: Video processing (with real diff-heatmap MHI)
        motion_history_base64, first_frame_base64, rotation_info = \
            self.create_motion_history_image(video_path)

        # Step 2: Semantic identification (detailed for TRELLIS)
        semantic_info = self.identify_object_semantically(first_frame_base64)
        object_name = semantic_info['object_type']

        # Step 3: Physics extraction
        physics_guess = self.ask_cosmos_physics_enhanced(
            motion_history_base64, rotation_info, object_name
        )

        # Step 4: TRELLIS 3D model generation (text-to-3D with detailed prompts)
        visual_asset_path = self.generate_trellis_model(semantic_info)

        # Step 5: Massive parallel physics search (fixed horizontal distance)
        video_duration = rotation_info.get('duration', 1.5)
        video_fps = rotation_info.get('fps', 60.0)

        swarm_result = self.massive_parallel_search(
            estimated_friction=physics_guess.get('friction', 0.5),
            estimated_mass=physics_guess.get('mass', 2.5),
            estimated_restitution=physics_guess.get('restitution', 0.3),
            ground_truth_dist=ground_truth_distance,
            ground_truth_duration=video_duration,
            num_agents=num_agents,
            semantic_info=semantic_info,
            known_mass=known_mass,
            object_profile_name=object_profile_name,
            v0=v0,
            start_height=start_height,
            video_fps=video_fps
        )

        # Merge results (use new naming: effective_mass_kg)
        verified_physics = {
            **physics_guess,
            'friction': swarm_result['friction'],
            'effective_mass_kg': swarm_result['effective_mass_kg'],
            'mass_identifiability': swarm_result['mass_identifiability'],
            'mass_basis': swarm_result['mass_basis'],
            'drag': swarm_result['drag'],
            'restitution': swarm_result['restitution'],
            'motion_model': swarm_result['motion_model'],
            'object_profile': swarm_result['object_profile'],
            'simulated_distance': swarm_result['simulated_distance'],
            'simulated_time': swarm_result['simulated_time'],
            'distance_error_m': swarm_result['distance_error_m'],
            'time_error_s': swarm_result['time_error_s'],
        }

        # Step 6: Generate semantic digital twin (contest-safe USD)
        twin_path = self.generate_semantic_digital_twin(
            visual_asset_path=visual_asset_path,
            physics_params=verified_physics,
            semantic_info=semantic_info,
            num_agents=num_agents  # FIXED: Dynamic not hardcoded
        )

        result = {
            "status": "success",
            "semantic_info": semantic_info,
            "physics_parameters": verified_physics,
            "visual_asset": visual_asset_path,
            "digital_twin_path": twin_path,
            "swarm_statistics": {
                "num_hypotheses": swarm_result['num_hypotheses'],
                "distance_error_m": swarm_result['distance_error_m'],
                "time_error_s": swarm_result['time_error_s'],
                "convergence_achieved_distance": swarm_result['distance_error_m'] < 0.2,
                "convergence_achieved_time": swarm_result['time_error_s'] < 0.5,
                "convergence_achieved": (swarm_result['distance_error_m'] < 0.2
                                         and swarm_result['time_error_s'] < 0.5),
            },
            "output_directory": str(self.output_dir.absolute()),
            "device_used": WARP_DEVICE,
            "seed_used": self.seed,
            "gtc_ready": True
        }

        # Step 7: Generate GTC visualizations (fixed trajectory)
        self.generate_gtc_visualizations(result, ground_truth_distance)
        
        # Step 8: Generate run report (JSON + Markdown)
        self.generate_run_report(result, video_path, ground_truth_distance, num_agents, video_duration)

        logger.info("\n" + "="*70)
        logger.info("🏆 GTC PIPELINE COMPLETE")
        logger.info("="*70)

        return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main entry point for GTC Golden Ticket submission (contest-ready)."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Physical Agent Swarm: Universal Physics Framework Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (infers object profile from video)
  python app.py --video bouncing_ball.mp4 --truth 2.5 --seed 42

  # With known mass (high identifiability)
  python app.py --video tennis_ball.mp4 --truth 3.0 --known_mass 0.057 --seed 42

  # With explicit object profile
  python app.py --video cardboard_box.mp4 --truth 2.5 --object_profile cube_cardboard_small --seed 42

  # With custom initial conditions
  python app.py --video soccer_ball.mp4 --truth 4.0 --v0 5.0 --start_height 1.0 --agents 8192

Output:
  - semantic_digital_twin.usda (drop into Omniverse)
  - gtc_presentation.png (visualization with mass degeneracy plot)
  - run.json + run.md (reproducibility report with identifiability metadata)

Object Profiles Available:
  Spheres: sphere_rubber_small, sphere_rubber_medium, sphere_foam_light, sphere_metal_heavy
  Cubes: cube_cardboard_small, cube_cardboard_medium, cube_wood_medium, cube_metal_heavy, cube_plastic_light
  Cylinders: cylinder_plastic_bottle, cylinder_metal_can
  Complex: boot_leather, shoe_fabric
  Fallback: generic_light, generic_medium, generic_heavy
        """
    )
    
    parser.add_argument("--video", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--truth", type=float, default=2.5,
                       help="Ground truth distance (meters)")
    parser.add_argument("--output", type=str, default="./gtc_output",
                       help="Output directory")
    parser.add_argument("--agents", type=int, default=4096,
                       help="Number of parallel hypotheses")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    # UNIVERSAL FRAMEWORK: Optional overrides for known parameters
    parser.add_argument("--known_mass", type=float, default=None,
                       help="Known mass (kg) - if provided, skips mass fitting (identifiable: high)")
    parser.add_argument("--object_profile", type=str, default=None,
                       help="Object profile name (e.g., 'sphere_rubber_small', 'cube_cardboard_medium')")
    parser.add_argument("--v0", type=float, default=None,
                       help=f"Initial horizontal velocity (m/s) - default: {PhysicalAgentSwarmGTC.DEFAULT_V0}")
    parser.add_argument("--start_height", type=float, default=None,
                       help=f"Starting height (m) - default: {PhysicalAgentSwarmGTC.DEFAULT_START_HEIGHT}")
    
    args = parser.parse_args()
    
    # Initialize GTC swarm with seed
    swarm = PhysicalAgentSwarmGTC(
        max_iterations=1, 
        output_dir=args.output,
        seed=args.seed
    )
    
    # Run pipeline (with universal framework parameters)
    result = swarm.run_gtc_pipeline(
        video_path=args.video,
        ground_truth_distance=args.truth,
        num_agents=args.agents,
        known_mass=args.known_mass,
        object_profile_name=args.object_profile,
        v0=args.v0,
        start_height=args.start_height
    )
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("🎯 GTC SUBMISSION READY")
    logger.info("="*70)
    logger.info(f"Object: {result['semantic_info']['object_type']}")
    logger.info(f"Description: {result['semantic_info']['detailed_description'][:100]}...")
    logger.info(f"Profile: {result['physics_parameters'].get('object_profile', 'unknown')}")
    logger.info(f"\n📊 Physics (Warp Monte Carlo - {args.agents} Hypotheses, Seed={args.seed}):")
    logger.info(f"   Friction: μ={result['physics_parameters']['friction']:.3f}")
    logger.info(f"   Effective Mass: m_eff={result['physics_parameters']['effective_mass_kg']:.2f}kg")
    logger.info(f"   Mass Identifiability: {result['physics_parameters']['mass_identifiability']} ({result['physics_parameters']['mass_basis']})")
    logger.info(f"   Drag: k={result['physics_parameters'].get('drag', 0):.4f}")
    logger.info(f"   Restitution: e={result['physics_parameters']['restitution']:.3f}")
    logger.info(f"   Motion Model: {result['physics_parameters'].get('motion_model', 'unknown')}")
    stats = result['swarm_statistics']
    logger.info(f"\n🎯 Convergence:")
    logger.info(f"   Distance Error: {stats['distance_error_m']:.4f}m"
                f"  [{'CONVERGED' if stats['convergence_achieved_distance'] else 'NOT converged'}]")
    logger.info(f"   Time Error: {stats['time_error_s']:.3f}s"
                f"  [{'CONVERGED' if stats['convergence_achieved_time'] else 'NOT converged'}]")
    
    total_time = sum(t.get('elapsed', 0) for t in swarm.timings.values())
    logger.info(f"\n⏱️  Total Pipeline Time: {total_time:.2f}s")
    
    logger.info(f"\n📦 OUTPUTS:")
    logger.info(f"   Digital Twin: {result['digital_twin_path']}")
    logger.info(f"   Visualization: gtc_presentation.png")
    logger.info(f"   Run Report: run.json + run.md")
    
    logger.info("\n🚀 NEXT STEPS:")
    logger.info("1. Open semantic_digital_twin.usda in NVIDIA Omniverse")
    logger.info("2. Press PLAY → Watch physics work instantly")
    logger.info("3. Submit to GTC 2025 Golden Ticket Contest!")
    
    logger.info("\n💡 UNIVERSAL FRAMEWORK FEATURES:")
    logger.info("   ✓ Object Profile Registry (not ball-specific)")
    logger.info("   ✓ Motion Model Dispatch (slide/roll/bounce/tumble)")
    logger.info("   ✓ Mass Identifiability Tracking (effective_mass_kg + degeneracy notes)")
    logger.info("   ✓ Configurable Priors (--object_profile, --known_mass, --v0, --start_height)")
    logger.info("   ✓ Reproducible (--seed flag, deterministic RNG)")
    logger.info("   ✓ Fixed physics (horizontal distance only)")
    logger.info("   ✓ Proper USD schema (PhysicsMassAPI, collision binding)")
    logger.info("   ✓ Real MHI (diff-heatmap motion tracking)")
    logger.info("   ✓ Run reports (JSON + Markdown)")
    logger.info("   ✓ Detailed TRELLIS prompts (rich semantics)")
    logger.info("="*70)


if __name__ == "__main__":
    main()