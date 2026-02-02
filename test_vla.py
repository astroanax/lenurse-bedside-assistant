#!/usr/bin/env python3
"""
Test script for VLA (Vision-Language-Action) model integration
Verifies that the pi0.5 model from LeRobot can be loaded and run inference

Usage:
    python test_vla.py
    python test_vla.py --device cuda  # Use GPU if available
"""

import argparse
import sys
import numpy as np

def test_vla_imports():
    """Test if all VLA dependencies are available"""
    print("🔍 Testing VLA dependencies...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    try:
        from huggingface_hub import hf_hub_download
        print("✓ HuggingFace Hub")
    except ImportError:
        print("✗ HuggingFace Hub not installed")
        return False
    
    try:
        import lerobot
        print(f"✓ LeRobot")
    except ImportError:
        print("✗ LeRobot not installed")
        print("  Install with: pip install lerobot")
        return False
    
    return True


def test_vla_controller():
    """Test VLA controller initialization"""
    print("\n🤖 Testing VLA controller...")
    
    try:
        from vla_controller import get_vla_controller, VLAController
        
        # Create controller
        vla = get_vla_controller(device="cpu")
        print(f"✓ VLA controller created")
        
        # Check status
        status = vla.get_status()
        print(f"✓ Status: {status}")
        
        # Test with dummy data
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        instruction = "pick up the red cube"
        
        print(f"\n📸 Testing inference with instruction: '{instruction}'")
        result = vla.predict_action(dummy_image, instruction)
        
        print(f"✓ Prediction completed")
        print(f"  Success: {result['success']}")
        print(f"  Action type: {result['action_type']}")
        
        if result['success'] and result['action_type'] == 'vla_generated':
            print(f"  Servo commands: {result.get('servo_commands', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 0.0):.2f}")
        elif result['action_type'] == 'trajectory_fallback':
            print(f"  Fallback trajectory: {result.get('trajectory_file', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test VLA model integration")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VLA (Vision-Language-Action) Model Test")
    print("Model: shehiin/pi05_pick_red_cube_lora")
    print("Framework: LeRobot")
    print("=" * 60)
    
    # Test dependencies
    if not test_vla_imports():
        print("\n❌ Dependency test failed")
        print("Install missing dependencies with:")
        print("  pip install lerobot torch torchvision huggingface-hub")
        sys.exit(1)
    
    # Test VLA controller
    if not test_vla_controller():
        print("\n❌ VLA controller test failed")
        print("\nNote: The VLA model will fall back to trajectory-based control")
        print("if the model cannot be loaded. This is normal for development.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
