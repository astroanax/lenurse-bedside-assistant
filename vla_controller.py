"""
Vision-Language-Action (VLA) Controller for SO-101 5 DOF Robot
Uses pi0.5 VLA model from shehiin/pi05_pick_red_cube_lora for real-time action generation

This module interfaces with LeRobot to generate robotic actions directly from:
- Visual observations (camera feed)
- Natural language instructions (voice commands)
- Current robot state

Unlike traditional trajectory-based systems, the VLA model generates actions
on-the-fly without requiring pre-recorded movements.
"""

import os
import numpy as np
import base64
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# LeRobot imports for VLA policy
try:
    from lerobot.common.policies.pi_zero.modeling_pi_zero import PiZeroPolicy
    from huggingface_hub import hf_hub_download
    import torch
    LEROBOT_AVAILABLE = True
except ImportError:
    print("WARNING: LeRobot not installed. Running in simulation mode.")
    print("Install with: pip install lerobot torch torchvision")
    LEROBOT_AVAILABLE = False


class VLAController:
    """
    Vision-Language-Action controller using pi0.5 model
    
    Translates visual observations + language commands into robot actions
    """
    
    # HuggingFace model identifier
    VLA_MODEL_ID = "shehiin/pi05_pick_red_cube_lora"
    
    # SO-101 5 DOF joint limits (in servo units: 0-4095)
    JOINT_LIMITS = {
        'shoulder_pan': (500, 3500),
        'shoulder_lift': (500, 3500),
        'elbow_flex': (500, 3500),
        'wrist_flex': (500, 3500),
        'wrist_roll': (500, 3500),
        'gripper': (1024, 3072),  # Open to closed
    }
    
    # Home position for safety
    HOME_POSITION = [2048, 2048, 2048, 2048, 2048, 1024]
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize VLA controller
        
        Args:
            device: torch device ('cpu' or 'cuda')
        """
        self.device = device
        self.model = None
        self.is_initialized = False
        
        # Current robot state
        self.current_position = self.HOME_POSITION.copy()
        self.current_gripper_state = "open"
        
        # Action history for smoothing
        self.action_history: List[np.ndarray] = []
        self.max_history_len = 5
        
        # VLA inference settings
        self.action_horizon = 10  # Number of future actions to predict
        self.control_frequency = 10  # Hz
        
        if LEROBOT_AVAILABLE:
            self._load_vla_model()
        else:
            print("[VLA] Running in SIMULATION mode - using trajectory fallback")
    
    def _load_vla_model(self):
        """Load the pi0.5 VLA model from HuggingFace"""
        try:
            print(f"[VLA] Loading pi0.5 model from {self.VLA_MODEL_ID}...")
            
            # Download model weights from HuggingFace
            model_path = hf_hub_download(
                repo_id=self.VLA_MODEL_ID,
                filename="policy.safetensors",
                cache_dir="./models/vla_cache"
            )
            
            # Initialize policy
            self.model = PiZeroPolicy.from_pretrained(
                self.VLA_MODEL_ID,
                device=self.device
            )
            self.model.eval()
            
            self.is_initialized = True
            print(f"[VLA] ✓ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"[VLA] Warning: Could not load VLA model: {e}")
            print("[VLA] Falling back to trajectory-based control")
            self.is_initialized = False
    
    def predict_action(
        self, 
        image: np.ndarray, 
        instruction: str,
        robot_state: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Generate robot action from visual observation and language instruction
        
        Args:
            image: Camera image (H, W, 3) in BGR format
            instruction: Natural language instruction (e.g., "pick up the water bottle")
            robot_state: Current robot joint positions
        
        Returns:
            Dictionary with predicted actions and metadata
        """
        if not self.is_initialized or not LEROBOT_AVAILABLE:
            # Fallback to trajectory-based control
            return self._fallback_action_generation(instruction)
        
        try:
            # Preprocess image for VLA model
            image_tensor = self._preprocess_image(image)
            
            # Encode language instruction
            instruction_embedding = self._encode_instruction(instruction)
            
            # Get current robot state
            if robot_state is None:
                robot_state = {
                    'joint_positions': self.current_position,
                    'gripper_state': self.current_gripper_state
                }
            
            # Run VLA inference
            with torch.no_grad():
                vla_output = self.model(
                    images=image_tensor,
                    instructions=instruction_embedding,
                    robot_state=robot_state
                )
            
            # Extract predicted actions
            predicted_actions = vla_output['actions'].cpu().numpy()
            
            # Apply action smoothing
            smoothed_actions = self._smooth_actions(predicted_actions)
            
            # Convert VLA output to servo commands
            servo_commands = self._vla_to_servo_commands(smoothed_actions)
            
            # Safety checks
            safe_commands = self._apply_safety_limits(servo_commands)
            
            return {
                'success': True,
                'servo_commands': safe_commands,
                'action_type': 'vla_generated',
                'confidence': float(vla_output.get('confidence', 0.9)),
                'instruction': instruction,
                'model': self.VLA_MODEL_ID
            }
            
        except Exception as e:
            print(f"[VLA] Error during inference: {e}")
            return self._fallback_action_generation(instruction)
    
    def _fallback_action_generation(self, instruction: str) -> Dict[str, any]:
        """
        Fallback to trajectory-based control when VLA is unavailable
        Maps instructions to pre-recorded trajectory files
        """
        # Map common instructions to trajectory files
        trajectory_map = {
            'water': 'pick_water_bottle.json',
            'bottle': 'pick_water_bottle.json',
            'drink': 'pick_water_bottle.json',
            'fruit': 'pick_fruit.json',
            'apple': 'pick_fruit.json',
            'orange': 'pick_fruit.json',
            'light': 'toggle_light.json',
            'lamp': 'toggle_light.json',
        }
        
        # Find matching trajectory
        instruction_lower = instruction.lower()
        trajectory_file = None
        
        for keyword, traj_file in trajectory_map.items():
            if keyword in instruction_lower:
                trajectory_file = traj_file
                break
        
        if trajectory_file:
            print(f"[VLA] 📋 Using trajectory: {trajectory_file}")
            return {
                'success': True,
                'action_type': 'trajectory_fallback',
                'trajectory_file': trajectory_file,
                'instruction': instruction,
                'note': 'VLA unavailable - using trajectory'
            }
        else:
            print(f"[VLA] ⚠ No matching action for: {instruction}")
            return {
                'success': False,
                'action_type': 'unknown',
                'instruction': instruction,
                'error': 'Could not map instruction to action'
            }
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess camera image for VLA model input"""
        # Convert BGR to RGB
        image_rgb = image[:, :, ::-1]
        
        # Resize to model input size (typically 224x224 or 256x256)
        from cv2 import resize
        image_resized = resize(image_rgb, (224, 224))
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def _encode_instruction(self, instruction: str) -> torch.Tensor:
        """Encode language instruction for VLA model"""
        # For pi0.5, instructions are typically encoded using CLIP or similar
        # This is a placeholder - actual implementation depends on model architecture
        return instruction  # Model handles encoding internally
    
    def _smooth_actions(self, actions: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to predicted actions"""
        self.action_history.append(actions)
        
        # Keep only recent history
        if len(self.action_history) > self.max_history_len:
            self.action_history.pop(0)
        
        # Moving average filter
        if len(self.action_history) > 1:
            smoothed = np.mean(self.action_history, axis=0)
        else:
            smoothed = actions
        
        return smoothed
    
    def _vla_to_servo_commands(self, vla_actions: np.ndarray) -> List[int]:
        """
        Convert VLA model output to SO-101 servo commands
        
        VLA outputs normalized joint angles [-1, 1]
        Convert to servo range [0, 4095]
        """
        servo_commands = []
        
        for i, action in enumerate(vla_actions[:6]):  # 6 DOF (5 joints + gripper)
            # Map [-1, 1] to joint limits
            joint_min, joint_max = list(self.JOINT_LIMITS.values())[i]
            servo_pos = int((action + 1) / 2 * (joint_max - joint_min) + joint_min)
            servo_commands.append(servo_pos)
        
        return servo_commands
    
    def _apply_safety_limits(self, commands: List[int]) -> List[int]:
        """Apply joint limits and velocity constraints for safety"""
        safe_commands = []
        
        for i, cmd in enumerate(commands):
            joint_min, joint_max = list(self.JOINT_LIMITS.values())[i]
            
            # Clamp to joint limits
            safe_cmd = max(joint_min, min(joint_max, cmd))
            
            # Velocity limiting (max change per step)
            max_delta = 200  # Max servo units per control step
            current = self.current_position[i]
            delta = safe_cmd - current
            
            if abs(delta) > max_delta:
                safe_cmd = current + np.sign(delta) * max_delta
            
            safe_commands.append(int(safe_cmd))
        
        # Update current position
        self.current_position = safe_commands
        
        return safe_commands
    
    def reset(self):
        """Reset controller state"""
        self.action_history = []
        self.current_position = self.HOME_POSITION.copy()
        print("[VLA] Controller reset to home position")
    
    def get_status(self) -> Dict:
        """Get current VLA controller status"""
        return {
            'initialized': self.is_initialized,
            'model': self.VLA_MODEL_ID if self.is_initialized else 'trajectory_fallback',
            'device': self.device,
            'lerobot_available': LEROBOT_AVAILABLE,
            'current_position': self.current_position,
            'mode': 'vla' if self.is_initialized else 'trajectory'
        }


# Singleton instance
_vla_controller = None


def get_vla_controller(device: str = "cpu") -> VLAController:
    """Get or create the VLA controller singleton"""
    global _vla_controller
    if _vla_controller is None:
        _vla_controller = VLAController(device=device)
    return _vla_controller


def reset_vla_controller():
    """Reset the VLA controller singleton"""
    global _vla_controller
    if _vla_controller:
        _vla_controller.reset()
    _vla_controller = None
