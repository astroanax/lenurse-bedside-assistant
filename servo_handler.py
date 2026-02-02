"""
Servo Handler - Bridge between trajectory_agent and actual servo hardware
Initializes and controls the robotic arm servos
"""

import sys
import os
from typing import Optional

# Setup environment for pygame (headless mode)
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Add servo SDK to path
SERVO_SDK_PATH = (
    "/home/astroanax/dev/vla-pinn-rl/hackathon/drm_hack/STServo_Python/stservo-env"
)
if SERVO_SDK_PATH not in sys.path:
    sys.path.append(SERVO_SDK_PATH)

try:
    import pygame

    pygame.init()
    PYGAME_AVAILABLE = True
except ImportError:
    print("WARNING: pygame not available")
    PYGAME_AVAILABLE = False

try:
    from scservo_sdk import *

    SERVO_SDK_AVAILABLE = True
except ImportError:
    print("WARNING: Servo SDK not available. Servo commands will be simulated.")
    SERVO_SDK_AVAILABLE = False
    # Define stub for COMM_SUCCESS
    COMM_SUCCESS = 0


# Servo configuration (from record_replay.py)
DEVICENAME = "/dev/ttyACM0"
BAUDRATE = 1000000

SERVO_LIMITS = {
    1: [500, 3500],
    2: [500, 3500],
    3: [500, 3500],
    4: [500, 3500],
    5: [500, 3500],
    6: [500, 3500],
}


class ServoController:
    """Controls the physical servo hardware"""

    def __init__(self):
        self.connected = False
        self.port = None
        self.servo = None
        self.current_positions = [2048] * 6

    def connect(self) -> bool:
        """
        Connect to servo hardware

        Returns:
            True if connected successfully, False otherwise
        """
        if not SERVO_SDK_AVAILABLE:
            print("Servo SDK not available - running in simulation mode")
            self.connected = False
            return False

        try:
            # Initialize port
            self.port = PortHandler(DEVICENAME)

            if not self.port.openPort():
                print(f"Error: Could not open servo port {DEVICENAME}")
                return False

            if not self.port.setBaudRate(BAUDRATE):
                print(f"Error: Could not set baud rate to {BAUDRATE}")
                self.port.closePort()
                return False

            # Initialize servo interface
            self.servo = sms_sts(self.port)

            # Read current positions
            self._read_all_positions()

            self.connected = True
            print(f"✓ Servo controller connected to {DEVICENAME}")
            print(f"✓ Current positions: {self.current_positions}")
            return True

        except Exception as e:
            print(f"Error connecting to servos: {e}")
            if self.port:
                self.port.closePort()
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from servo hardware"""
        if self.port:
            self.port.closePort()
        self.connected = False
        print("Servo controller disconnected")

    def _read_all_positions(self):
        """Read current positions from all servos"""
        if not self.connected or not self.servo:
            return

        for i in range(6):
            try:
                pos, res, err = self.servo.ReadPos(i + 1)
                if res == COMM_SUCCESS:
                    self.current_positions[i] = pos
            except Exception as e:
                print(f"Warning: Could not read servo {i + 1} position: {e}")

    def move_servo(self, servo_id: int, position: int, speed: int = 800, acc: int = 40):
        """
        Move a single servo to target position

        Args:
            servo_id: Servo ID (1-6)
            position: Target position (servo units)
            speed: Movement speed
            acc: Acceleration
        """
        # Validate servo_id
        if servo_id < 1 or servo_id > 6:
            print(f"Error: Invalid servo_id {servo_id}")
            return

        # Apply limits
        limits = SERVO_LIMITS.get(servo_id, [500, 3500])
        position = max(limits[0], min(limits[1], int(position)))

        if not self.connected or not self.servo:
            # Simulation mode
            print(f"[SIM] Servo {servo_id} → {position} (speed={speed})")
            self.current_positions[servo_id - 1] = position
            return

        try:
            # Move servo
            self.servo.WritePosEx(servo_id, position, speed, acc)
            self.current_positions[servo_id - 1] = position

        except Exception as e:
            print(f"Error moving servo {servo_id}: {e}")

    def get_positions(self):
        """Get current servo positions"""
        return list(self.current_positions)

    def is_connected(self) -> bool:
        """Check if connected to hardware"""
        return self.connected


# Singleton instance
_servo_controller: Optional[ServoController] = None


def get_servo_controller() -> ServoController:
    """Get or create singleton servo controller"""
    global _servo_controller
    if _servo_controller is None:
        _servo_controller = ServoController()
    return _servo_controller


def create_servo_handler():
    """
    Create a servo handler function for TrajectoryReplayer

    Returns:
        Callable that can be used as servo_handler for TrajectoryReplayer
    """
    controller = get_servo_controller()

    # Try to connect (will fall back to simulation if fails)
    if not controller.is_connected():
        controller.connect()

    def handler(servo_id: int, position: int, speed: int, acc: int):
        """Servo handler callback for trajectory replayer"""
        controller.move_servo(servo_id, position, speed, acc)

    return handler
