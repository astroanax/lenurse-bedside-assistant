"""
Hazard Detection System using YOLO11
Monitors camera feed for emergencies: blood, falls, distress, etc.
"""

import cv2
# import numpy as np  # Disabled due to illegal instruction on ARM64
from datetime import datetime
import threading
import time

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Run: pip install ultralytics")


class HazardDetector:
    """Detects hazards in video feed using YOLO11"""

    # Hazard keywords to trigger alerts
    HAZARD_CLASSES = [
        "person",  # For fall detection (unusual position)
        "knife",
        "scissors",
        "bottle",  # Spilled bottles
        "cup",  # Spilled cups
    ]

    def __init__(self, model_name="yolo11n.pt"):
        """Initialize YOLO11 model"""
        self.model = None
        self.running = False
        self.last_frame = None
        self.last_detections = []
        self.hazard_detected = False
        self.hazard_type = None
        self.hazard_callback = None
        self.last_alert_time = {}  # Cooldown for each hazard type
        
        # Red color detection for blood - disabled due to numpy illegal instruction on ARM64
        # self.blood_hsv_lower = np.array([0, 100, 100])
        # self.blood_hsv_upper = np.array([10, 255, 255])
        self.blood_hsv_lower = None
        self.blood_hsv_upper = None

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_name)
                print(f"✓ YOLO11 model loaded: {model_name}")
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                self.model = None
        else:
            print("YOLO11 not available - hazard detection disabled")

    def set_hazard_callback(self, callback):
        """Set callback function for hazard alerts"""
        self.hazard_callback = callback

    def detect_blood(self, frame):
        """Detect red (blood) in frame using HSV color space"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.blood_hsv_lower, self.blood_hsv_upper)

        # Calculate percentage of red pixels
        red_pixels = cv2.countNonZero(mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        red_percentage = (red_pixels / total_pixels) * 100

        # Threshold for blood detection (>10% red) - reduced false positives
        if red_percentage > 10.0:
            return True, red_percentage
        return False, red_percentage

    def detect_fall(self, detections):
        """Detect if person is in unusual position (fallen)"""
        for det in detections:
            if det["class"] == "person":
                # If person bounding box is wider than tall = possible fall
                width = det["bbox"][2] - det["bbox"][0]
                height = det["bbox"][3] - det["bbox"][1]
                aspect_ratio = width / height if height > 0 else 0

                if aspect_ratio > 1.5:  # Horizontal orientation
                    return True, aspect_ratio
        return False, 0

    def process_frame(self, frame):
        """Process frame with YOLO11 and hazard detection"""
        if frame is None:
            return frame, []

        self.last_frame = frame.copy()
        detections = []
        annotated_frame = frame.copy()

        # Check for blood - DISABLED due to numpy illegal instruction on ARM64
        # is_blood, red_pct = self.detect_blood(frame)
        # if is_blood:
        #     self.trigger_hazard("BLOOD", f"Blood detected ({red_pct:.1f}% red)")
        #     cv2.putText(
        #         annotated_frame,
        #         "!!! BLOOD DETECTED !!!",
        #         (10, 30),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (0, 0, 255),
        #         3,
        #     )

        # YOLO detection
        if self.model is not None:
            try:
                results = self.model(frame, verbose=False)

                if results and len(results) > 0:
                    result = results[0]

                    # Parse detections
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy()

                            class_name = self.model.names[cls_id]

                            detection = {
                                "class": class_name,
                                "confidence": conf,
                                "bbox": xyxy.tolist(),
                            }
                            detections.append(detection)

                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, xyxy)
                            color = (0, 255, 0)  # Green

                            # Check if hazardous object
                            if class_name in ["knife", "scissors"]:
                                color = (0, 0, 255)  # Red for dangerous
                                self.trigger_hazard(
                                    "SHARP_OBJECT", f"{class_name} detected"
                                )

                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                            # Label
                            label = f"{class_name} {conf:.2f}"
                            label_size = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                            )[0]
                            cv2.rectangle(
                                annotated_frame,
                                (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1),
                                color,
                                -1,
                            )
                            cv2.putText(
                                annotated_frame,
                                label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )

                    # Check for fall
                    is_fall, aspect = self.detect_fall(detections)
                    if is_fall:
                        self.trigger_hazard(
                            "FALL", f"Person fallen (aspect: {aspect:.2f})"
                        )
                        cv2.putText(
                            annotated_frame,
                            "!!! FALL DETECTED !!!",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3,
                        )

            except Exception as e:
                print(f"YOLO detection error: {e}")

        self.last_detections = detections
        return annotated_frame, detections

    def trigger_hazard(self, hazard_type, description):
        """Trigger hazard alert with cooldown to prevent spam"""
        current_time = time.time()
        
        # Check cooldown (10 seconds between same hazard type)
        if hazard_type in self.last_alert_time:
            if current_time - self.last_alert_time[hazard_type] < 10.0:
                return  # Skip if within cooldown period
        
        self.hazard_detected = True
        self.hazard_type = hazard_type
        self.last_alert_time[hazard_type] = current_time

        if self.hazard_callback:
            self.hazard_callback(
                {
                    "type": hazard_type,
                    "description": description,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def reset_hazard(self):
        """Reset hazard state"""
        self.hazard_detected = False
        self.hazard_type = None

    def get_status(self):
        """Get detector status"""
        return {
            "running": self.running,
            "model_loaded": self.model is not None,
            "hazard_detected": self.hazard_detected,
            "hazard_type": self.hazard_type,
            "last_detections": self.last_detections,
        }


# Global detector instance
_detector = None


def get_detector():
    """Get or create global detector instance"""
    global _detector
    if _detector is None:
        _detector = HazardDetector()
    return _detector


def reset_detector():
    """Reset global detector"""
    global _detector
    if _detector:
        _detector.running = False
    _detector = None
