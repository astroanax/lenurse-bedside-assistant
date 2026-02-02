#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "pygame",
#     "pyserial",
# ]
# ///
"""
Simple trajectory recorder and replayer for robotic arm
Saves trajectories to JSON files
"""

import sys
import os
import time
import json
import argparse

os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

sys.path.append(
    "/home/astroanax/dev/vla-pinn-rl/hackathon/drm_hack/STServo_Python/stservo-env"
)
from scservo_sdk import *

# =============================================================================
# CONFIG
# =============================================================================

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

# Positions [base, shoulder, elbow, wrist, wrist_rot, gripper]
HOME_POS = [2048, 1628, 1988, 2856, 2048, 2048]

# Movement
SPEED = 800
ACC = 40
DEADZONE = 0.15
STEP = 8
RECORD_FPS = 10

# =============================================================================
# UTILITY
# =============================================================================


def apply_deadzone(val):
    if abs(val) < DEADZONE:
        return 0.0
    sign = 1 if val > 0 else -1
    return sign * (abs(val) - DEADZONE) / (1.0 - DEADZONE)


# =============================================================================
# ARM
# =============================================================================


class Arm:
    def __init__(self, port, servo):
        self.port = port
        self.servo = servo
        self.pos = [2048] * 6

    def read_positions(self):
        for i in range(6):
            try:
                pos, res, err = self.servo.ReadPos(i + 1)
                if res == COMM_SUCCESS:
                    self.pos[i] = pos
            except:
                pass
        return self.pos

    def move(self, servo_id, position, speed=SPEED):
        limits = SERVO_LIMITS.get(servo_id, [500, 3500])
        position = max(limits[0], min(limits[1], int(position)))
        try:
            self.servo.WritePosEx(servo_id, position, speed, ACC)
            self.pos[servo_id - 1] = position
        except:
            pass

    def move_all(self, positions, speed=SPEED):
        for i, pos in enumerate(positions):
            if pos is not None:
                self.move(i + 1, pos, speed)
                time.sleep(0.01)

    def move_delta(self, servo_id, delta):
        if delta == 0:
            return
        current = self.pos[servo_id - 1]
        self.move(servo_id, current + delta)

    def get_positions(self):
        return list(self.pos)


# =============================================================================
# RECORDER
# =============================================================================


class TrajectoryRecorder:
    def __init__(self):
        self.recording = False
        self.trajectory = []
        self.start_time = 0
        self.last_record = 0

    def start(self):
        self.trajectory = []
        self.start_time = time.time()
        self.recording = True
        print("\n>>> RECORDING STARTED")

    def stop(self):
        if not self.recording:
            return None
        self.recording = False
        print(">>> RECORDING STOPPED")
        return {
            "duration": time.time() - self.start_time,
            "num_points": len(self.trajectory),
            "trajectory": self.trajectory,
        }

    def record_point(self, positions):
        if not self.recording:
            return

        now = time.time()
        if now - self.last_record < 1.0 / RECORD_FPS:
            return
        self.last_record = now

        self.trajectory.append(
            {
                "t": round(now - self.start_time, 3),
                "pos": positions,
            }
        )


# =============================================================================
# RECORD MODE
# =============================================================================


def record_mode(arm, joystick):
    recorder = TrajectoryRecorder()
    prev_x = False
    prev_y = False
    prev_a = False
    last_display = 0

    print("\n" + "=" * 60)
    print("TRAJECTORY RECORDING MODE")
    print("=" * 60)
    print("Controls:")
    print("  Left Stick  = Base + Shoulder (servos 1, 2)")
    print("  Right Stick = Elbow + Wrist (servos 3, 4)")
    print("  LT/RT       = Wrist rotation (servo 5)")
    print("  L1/R1       = Gripper close/open (servo 6)")
    print("")
    print("  A = Go to home position")
    print("  X = Start/Stop recording")
    print("  Y = Save trajectory")
    print("  START = Quit")
    print("=" * 60)

    trajectory_data = None

    try:
        while True:
            pygame.event.pump()

            # Quit
            if joystick.get_button(7):  # START
                break

            # A = Go to home
            a = joystick.get_button(0)
            if a and not prev_a:
                print("Moving to home position...")
                arm.move_all(HOME_POS, 600)
            prev_a = a

            # X = Start/Stop recording
            x = joystick.get_button(2)
            if x and not prev_x:
                if not recorder.recording:
                    recorder.start()
                else:
                    trajectory_data = recorder.stop()
                    print(
                        f"Recorded {trajectory_data['num_points']} points over {trajectory_data['duration']:.2f}s"
                    )
            prev_x = x

            # Y = Save
            y = joystick.get_button(3)
            if y and not prev_y:
                if trajectory_data:
                    filename = f"trajectory_{time.strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, "w") as f:
                        json.dump(trajectory_data, f, indent=2)
                    print(f">>> SAVED: {filename}")
                    trajectory_data = None
                else:
                    print("No trajectory to save!")
            prev_y = y

            # Teleop controls
            lx = apply_deadzone(joystick.get_axis(0))
            ly = apply_deadzone(-joystick.get_axis(1))
            rx = apply_deadzone(joystick.get_axis(3))
            ry = apply_deadzone(-joystick.get_axis(4))
            lt = (joystick.get_axis(2) + 1) / 2
            rt = (joystick.get_axis(5) + 1) / 2

            if lx:
                arm.move_delta(1, lx * STEP)
            if ly:
                arm.move_delta(2, ly * STEP)
            if rx:
                arm.move_delta(3, rx * STEP)
            if ry:
                arm.move_delta(4, ry * STEP)
            if rt > 0.1:
                arm.move_delta(5, rt * STEP)
            elif lt > 0.1:
                arm.move_delta(5, -lt * STEP)

            # Gripper
            if joystick.get_button(5):  # R1 = open
                arm.move_delta(6, STEP)
            if joystick.get_button(4):  # L1 = close
                arm.move_delta(6, -STEP)

            # Record current position
            recorder.record_point(arm.get_positions())

            # Display
            if time.time() - last_display > 0.3:
                os.system("clear")
                print("=" * 60)
                if recorder.recording:
                    print(f"*** RECORDING *** Points: {len(recorder.trajectory)}")
                    print("Press X to stop")
                elif trajectory_data:
                    print(
                        f"Last recording: {trajectory_data['num_points']} points, {trajectory_data['duration']:.2f}s"
                    )
                    print("Press Y to save")
                else:
                    print("Press X to start recording")
                print(f"Position: {arm.pos}")
                print("=" * 60)
                last_display = time.time()

            time.sleep(0.02)

    except KeyboardInterrupt:
        pass


# =============================================================================
# REPLAY MODE
# =============================================================================


def replay_mode(arm, trajectory_file):
    print(f"\nLoading trajectory: {trajectory_file}")

    try:
        with open(trajectory_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{trajectory_file}' not found!")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{trajectory_file}'!")
        return

    trajectory = data.get("trajectory", [])
    duration = data.get("duration", 0)

    print(f"Points: {len(trajectory)}")
    print(f"Duration: {duration:.2f}s")
    print("\nPress ENTER to start replay...")
    input()

    print("REPLAYING...")
    start_time = time.time()

    for i, point in enumerate(trajectory):
        target_time = point["t"]
        positions = point["pos"]

        # Wait until we should execute this point
        while time.time() - start_time < target_time:
            time.sleep(0.005)

        # Move to position
        arm.move_all(positions, SPEED)

        print(f"Point {i + 1}/{len(trajectory)}: t={target_time:.2f}s", end="\r")

    print(f"\n>>> REPLAY COMPLETE!")
    print(f"Final position: {arm.pos}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Record and replay robotic arm trajectories"
    )
    parser.add_argument(
        "--replay", type=str, metavar="FILE", help="Replay trajectory from JSON file"
    )
    args = parser.parse_args()

    # Initialize pygame for controller
    pygame.init()
    pygame.joystick.init()

    joystick = None
    if not args.replay:
        if pygame.joystick.get_count() == 0:
            print("Error: No controller detected!")
            return 1
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Controller: {joystick.get_name()}")

    # Initialize servo connection
    port = PortHandler(DEVICENAME)
    if not port.openPort():
        print("Error: Could not open servo port!")
        return 1
    if not port.setBaudRate(BAUDRATE):
        print("Error: Could not set baud rate!")
        return 1

    servo = sms_sts(port)
    arm = Arm(port, servo)
    arm.read_positions()
    print(f"Arm connected. Current position: {arm.pos}")

    try:
        if args.replay:
            # Replay mode
            replay_mode(arm, args.replay)
        else:
            # Record mode
            record_mode(arm, joystick)
    finally:
        port.closePort()
        pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
