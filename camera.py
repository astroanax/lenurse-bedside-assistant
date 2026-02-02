"""
Camera Handler for Bedside Healthcare Assistant
Handles OpenCV camera operations with device selection, MJPEG streaming, and YOLO11 hazard detection
"""

import cv2
import threading
import time
from typing import Optional, List, Tuple, Generator
from hazard_detector import get_detector


class CameraHandler:
    """Manages camera operations for the inspection system"""

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        enable_hazard_detection: bool = False,  # Disabled by default to avoid illegal instruction on ARM64
    ):
        """
        Initialize camera handler

        Args:
            device_id: Camera device index (default 0)
            width: Frame width
            height: Frame height
            enable_hazard_detection: Enable YOLO11 hazard detection
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.current_frame: Optional[bytes] = None
        self.raw_frame: Optional[any] = None
        self.annotated_frame: Optional[any] = None  # Frame with YOLO bounding boxes
        self.lock = threading.Lock()
        self._capture_thread: Optional[threading.Thread] = None
        self.enable_hazard_detection = enable_hazard_detection
        self.hazard_detector = get_detector() if enable_hazard_detection else None

    def start(self) -> bool:
        """
        Start camera capture

        Returns:
            True if camera started successfully, False otherwise
        """
        if self.is_running:
            return True

        # Try V4L2 backend first (Linux)
        try:
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        except:
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            return False

        # Try to read a test frame to verify it works
        ret, _ = self.cap.read()
        if not ret:
            self.cap.release()
            self.cap = None
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        self.is_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        return True

    def stop(self):
        """Stop camera capture"""
        self.is_running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None

    def _capture_loop(self):
        """Background thread for continuous frame capture with YOLO11 hazard detection"""
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Process frame through YOLO11 hazard detector if enabled
                if self.enable_hazard_detection and self.hazard_detector:
                    annotated_frame, detections = self.hazard_detector.process_frame(
                        frame
                    )
                else:
                    annotated_frame = frame

                with self.lock:
                    self.raw_frame = frame.copy()
                    self.annotated_frame = annotated_frame.copy()
                    # Encode ANNOTATED frame (with bounding boxes) to JPEG for streaming
                    _, buffer = cv2.imencode(
                        ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    self.current_frame = buffer.tobytes()
            time.sleep(0.03)  # ~30 FPS

    def get_frame(self) -> Optional[bytes]:
        """
        Get current JPEG-encoded frame

        Returns:
            JPEG bytes or None if no frame available
        """
        with self.lock:
            return self.current_frame

    def get_raw_frame(self) -> Optional[any]:
        """
        Get current raw OpenCV frame (BGR numpy array)

        Returns:
            OpenCV frame or None if no frame available
        """
        with self.lock:
            if self.raw_frame is not None:
                return self.raw_frame.copy()
            return None

    def capture_still(self) -> Optional[any]:
        """
        Capture a still frame for inspection

        Returns:
            OpenCV frame (BGR numpy array) or None
        """
        return self.get_raw_frame()

    def generate_mjpeg(self) -> Generator[bytes, None, None]:
        """
        Generate MJPEG stream for HTTP streaming

        Yields:
            MJPEG frame bytes
        """
        while self.is_running:
            frame = self.get_frame()
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)

    def switch_device(self, device_id: int) -> bool:
        """
        Switch to a different camera device

        Args:
            device_id: New camera device index

        Returns:
            True if switch successful, False otherwise
        """
        self.stop()
        self.device_id = device_id
        return self.start()

    @staticmethod
    def list_available_cameras(max_devices: int = 10) -> List[dict]:
        """
        List all available camera devices

        Args:
            max_devices: Maximum number of devices to check

        Returns:
            List of dicts with device info
        """
        cameras = []
        for i in range(max_devices):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                if cap.isOpened():
                    # Try to read a frame to verify it's a real capture device
                    ret, _ = cap.read()
                    if ret:
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cameras.append(
                            {"id": i, "name": f"Camera {i} (/dev/video{i})", "resolution": f"{width}x{height}"}
                        )
                    cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {e}")
                pass
        return cameras

    @property
    def is_connected(self) -> bool:
        """Check if camera is connected and running"""
        return self.is_running and self.cap is not None and self.cap.isOpened()


# Singleton instance for the application
_camera_instance: Optional[CameraHandler] = None


def get_camera(device_id: int = 0) -> CameraHandler:
    """
    Get or create the singleton camera instance

    Args:
        device_id: Camera device index (only used on first call)

    Returns:
        CameraHandler instance
    """
    global _camera_instance
    if _camera_instance is None:
        _camera_instance = CameraHandler(device_id=device_id)
    return _camera_instance


def reset_camera():
    """Reset the camera singleton"""
    global _camera_instance
    if _camera_instance:
        _camera_instance.stop()
        _camera_instance = None
