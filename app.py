"""
Bedside Healthcare Assistant Robot - Flask Backend
Main application server with API routes
Voice-controlled robotic arm for assistance with daily activities
"""

import os
import cv2
import base64
import json
from datetime import datetime
from flask import (
    Flask,
    render_template,
    Response,
    jsonify,
    request,
    stream_with_context,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import local modules
from camera import CameraHandler, get_camera, reset_camera
from jarvis_agent import (
    get_orchestrator,
    get_translations,
    TRANSLATIONS,
)
from voice_processor import get_fallback_processor
from trajectory_agent import get_replayer, HOME_POS
from vla_controller import get_vla_controller  # Vision-Language-Action model
from hazard_detector import get_detector
from family_contacts import get_family_contacts, call_mom, call_dad

app = Flask(__name__)

# Application state
app_state = {
    "selected_task": None,  # Current task: "water_bottle", "fruit", "light", etc.
    "logs": [],
    "language": "en",  # Current language
    "dark_mode": False,  # Dark mode setting
    "jarvis_conversation": [],  # JARVIS conversation history
    "robot_state": "IDLE",  # Current robot state for display
    "robot_task": None,  # Current task being executed
}


def add_log(message: str, level: str = "info"):
    """Add entry to system log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {"timestamp": timestamp, "message": message, "level": level}
    app_state["logs"].append(entry)
    # Keep only last 100 logs
    if len(app_state["logs"]) > 100:
        app_state["logs"] = app_state["logs"][-100:]
    print(f"[{timestamp}] [{level.upper()}] {message}")


def update_robot_status(state: str, task_name=None):
    """Update robot status for display"""
    app_state["robot_state"] = state
    if task_name is not None:
        app_state["robot_task"] = task_name


# =============================================================================
# PAGE ROUTES
# =============================================================================


@app.route("/")
def index():
    """Serve main GUI"""
    return render_template("index.html")


# =============================================================================
# CAMERA API
# =============================================================================


@app.route("/api/cameras")
def list_cameras():
    """List available camera devices

    Returns a list of available camera devices on the system.
    Each camera is represented by its device ID (e.g., 0 for /dev/video0).
    """
    cameras = CameraHandler.list_available_cameras()
    return jsonify({"cameras": cameras})


@app.route("/api/camera/select", methods=["POST"])
def select_camera():
    """Switch to a different camera device

    This allows you to select which camera device to use.
    Camera devices are typically /dev/video0, /dev/video1, etc.

    To find available cameras, use /api/cameras endpoint first.
    Then POST to this endpoint with {"device_id": X} where X is the number.

    Example: {"device_id": 0} will use /dev/video0
             {"device_id": 2} will use /dev/video2
    """
    data = request.get_json()
    device_id = data.get("device_id", 0)

    add_log(f"📷 Switching camera to /dev/video{device_id}...")

    camera = get_camera()
    camera.stop()
    reset_camera()

    camera = get_camera(device_id)
    success = camera.start()

    if success:
        add_log(f"✅ Camera /dev/video{device_id} connected", "success")
        return jsonify({"success": True, "device_id": device_id})
    else:
        add_log(f"❌ Failed to connect to camera /dev/video{device_id}", "error")
        return jsonify({"success": False, "error": "Failed to connect to camera"}), 400


@app.route("/api/camera/status")
def camera_status():
    """Get camera connection status"""
    camera = get_camera()
    return jsonify({"connected": camera.is_connected, "device_id": camera.device_id})


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream"""
    camera = get_camera()
    if not camera.is_connected:
        camera.start()

    if not camera.is_connected:
        # Return a placeholder image
        return Response(
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + _get_no_camera_image()
            + b"\r\n",
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return Response(
        camera.generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def _get_no_camera_image():
    """Generate a 'no camera' placeholder image"""
    import numpy as np

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # Dark gray

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "NO CAMERA CONNECTED"
    text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
    text_x = (640 - text_size[0]) // 2
    text_y = (480 + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, 0.8, (100, 100, 100), 2)

    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


# =============================================================================
# HAZARD DETECTION API
# =============================================================================


@app.route("/api/hazard/status")
def hazard_status():
    """Get hazard detection status"""
    detector = get_detector()
    status = detector.get_status()
    return jsonify({"success": True, "hazard_detection": status})


@app.route("/api/hazard/reset", methods=["POST"])
def hazard_reset():
    """Reset hazard alert state"""
    detector = get_detector()
    detector.reset_hazard()
    add_log("Hazard alert reset")
    return jsonify({"success": True, "message": "Hazard alert reset"})


# Global hazard alert callback
def on_hazard_detected(hazard_info):
    """Callback when hazard is detected"""
    hazard_type = hazard_info.get("type", "UNKNOWN")
    description = hazard_info.get("description", "")

    add_log(f"🚨 HAZARD DETECTED: {hazard_type} - {description}", "error")

    # TODO: Trigger emergency siren sound on frontend
    # TODO: Optionally notify family contacts


# Set up hazard callback during initialization
def setup_hazard_detection():
    """Set up hazard detection callback"""
    detector = get_detector()
    detector.set_hazard_callback(on_hazard_detected)
    add_log("Hazard detection system initialized")


# =============================================================================
# FAMILY CONTACTS API
# =============================================================================


@app.route("/api/family/call/mom", methods=["POST"])
def api_call_mom():
    """Call mom"""
    add_log("Calling mom...")
    result = call_mom()

    if result["success"]:
        add_log(f"📞 {result['message']}", "success")
    else:
        add_log(f"❌ {result['message']}", "error")

    return jsonify(result)


@app.route("/api/family/call/dad", methods=["POST"])
def api_call_dad():
    """Call dad"""
    add_log("Calling dad...")
    result = call_dad()

    if result["success"]:
        add_log(f"📞 {result['message']}", "success")
    else:
        add_log(f"❌ {result['message']}", "error")

    return jsonify(result)


@app.route("/api/family/contacts")
def get_contacts():
    """Get family contacts"""
    contacts = get_family_contacts()
    return jsonify({"success": True, "contacts": contacts.get_contacts()})


@app.route("/api/family/contacts", methods=["POST"])
def update_contact():
    """Update a family contact"""
    data = request.get_json() or {}
    relation = data.get("relation")  # "mom" or "dad"
    phone = data.get("phone")
    name = data.get("name")

    if not relation or not phone:
        return jsonify({"success": False, "error": "relation and phone required"}), 400

    contacts = get_family_contacts()
    contacts.set_contact(relation, phone, name)

    add_log(f"Updated contact: {relation} -> {phone}")
    return jsonify({"success": True, "message": f"Contact updated: {relation}"})


@app.route("/api/family/history")
def get_call_history():
    """Get call history"""
    contacts = get_family_contacts()
    history = contacts.get_call_history(limit=20)
    return jsonify({"success": True, "history": history})


# =============================================================================
# ROBOT API
# =============================================================================


@app.route("/api/robot/status")
def robot_status():
    """Get robot status"""
    status = {
        "state": app_state.get("robot_state", "IDLE").lower(),
        "state_display": app_state.get("robot_state", "IDLE"),
        "current_task": app_state.get("robot_task"),
        "is_busy": app_state.get("robot_state", "IDLE") not in ["IDLE", "READY"],
    }
    return jsonify(status)


@app.route("/api/robot/servo-positions")
def get_servo_positions():
    """Get current servo positions for digital twin visualization"""
    try:
        traj_controller = get_replayer()

        # Get current positions from servo handler if available
        if hasattr(traj_controller, "servo_handler") and traj_controller.servo_handler:
            try:
                positions = HOME_POS
            except:
                positions = HOME_POS
        else:
            positions = HOME_POS

        # Convert servo positions (0-4095) to joint angles in radians
        def servo_to_radians(pos):
            return ((pos - 2048) / 2048.0) * 3.14159

        joint_angles = {
            "shoulder_pan": servo_to_radians(positions[0]),
            "shoulder_lift": servo_to_radians(positions[1]),
            "elbow_flex": servo_to_radians(positions[2]),
            "wrist_flex": servo_to_radians(positions[3]),
            "wrist_roll": servo_to_radians(positions[4]),
            "gripper": servo_to_radians(positions[5]),
        }

        return jsonify(
            {
                "success": True,
                "servo_positions": positions,
                "joint_angles": joint_angles,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/robot/home", methods=["POST"])
def robot_home():
    """Move robot to home position"""
    add_log("Robot moving to home position")
    update_robot_status("MOVING_HOME")

    # TODO: Add actual home movement logic
    # For now, just update status
    update_robot_status("IDLE")

    return jsonify({"success": True})


@app.route("/api/robot/reset", methods=["POST"])
def robot_reset():
    """Reset robot error state"""
    update_robot_status("IDLE")
    app_state["robot_task"] = None
    add_log("Robot reset")
    return jsonify({"success": True})


# =============================================================================
# STATE API
# =============================================================================


@app.route("/api/select", methods=["POST"])
def select_task():
    """Select task for execution"""
    data = request.get_json()
    task_name = data.get("task")

    valid_tasks = ["water_bottle", "fruit", "light", "medicine", "phone"]

    if task_name and task_name.lower() in valid_tasks:
        app_state["selected_task"] = task_name.lower()
        add_log(f"Selected task: {task_name.upper()}")
        return jsonify({"success": True, "task": task_name})
    else:
        return jsonify({"success": False, "error": "Invalid task type"}), 400


@app.route("/api/state")
def get_state():
    """Get current application state"""
    camera = get_camera()

    return jsonify(
        {
            "selected_task": app_state["selected_task"],
            "robot_state": app_state["robot_state"],
            "camera_connected": camera.is_connected,
        }
    )


@app.route("/api/logs")
def get_logs():
    """Get system logs"""
    return jsonify({"logs": app_state["logs"]})


@app.route("/api/reset", methods=["POST"])
def reset_state():
    """Reset application state"""
    app_state["selected_task"] = None
    app_state["robot_task"] = None
    update_robot_status("IDLE")
    add_log("System reset")
    return jsonify({"success": True})


# =============================================================================
# JARVIS API
# =============================================================================


@app.route("/api/jarvis/chat", methods=["POST"])
def jarvis_chat():
    """Send a message to JARVIS and get a response"""
    data = request.get_json() or {}
    user_input = data.get("message", "")
    language = data.get("language", app_state.get("language", "en"))

    if not user_input:
        return jsonify({"success": False, "error": "No message provided"}), 400

    # Check if API key is configured
    if not os.getenv("OPENROUTER_API_KEY"):
        error_msg = "OPENROUTER_API_KEY not configured. Please add it to .env file."
        add_log(error_msg, "error")
        return jsonify({"success": False, "error": error_msg}), 503

    add_log(f"💬 User ({language}): {user_input[:50]}...")

    try:
        camera = get_camera()
        orchestrator = get_orchestrator(camera, None)

        # Process the input
        result = orchestrator.process_input(user_input, language)

        # Store conversation
        app_state["jarvis_conversation"].append(
            {
                "role": "user",
                "content": user_input,
                "language": language,
                "timestamp": datetime.now().isoformat(),
            }
        )
        app_state["jarvis_conversation"].append(
            {
                "role": "assistant",
                "content": result.get("response", ""),
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Log JARVIS response
        add_log(f"JARVIS: {result.get('response', '')[:50]}...")

        # Log arm action if any
        if result.get("arm_action"):
            add_log(f"ARM: {result['arm_action'].get('message', '')}", "success")

        # Update robot status based on arm command
        arm_command = result.get("arm_command")
        if arm_command == "pick":
            target = result.get("arm_target", "object")
            update_robot_status(f"PICKING", target.upper())
        elif arm_command == "place":
            update_robot_status("PLACING")
            app_state["robot_task"] = None
        elif arm_command == "snap":
            update_robot_status("GREETING")
        elif arm_command == "call_mom":
            update_robot_status("CALLING")
            call_result = call_mom()
            add_log(f"📞 {call_result.get('message', 'Calling mom...')}", "success")
        elif arm_command == "call_dad":
            update_robot_status("CALLING")
            call_result = call_dad()
            add_log(f"📞 {call_result.get('message', 'Calling dad...')}", "success")
        elif arm_command == "none" or not arm_command:
            if app_state.get("robot_state") in ["GREETING", "CALLING"]:
                update_robot_status("IDLE")

        return jsonify(
            {
                "success": True,
                "response": result.get("response", ""),
                "arm_action": result.get("arm_action"),
                "is_greeting": result.get("is_greeting", False),
            }
        )

    except Exception as e:
        add_log(f"JARVIS error: {str(e)}", "error")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jarvis/conversation")
def get_jarvis_conversation():
    """Get JARVIS conversation history"""
    return jsonify(
        {
            "success": True,
            "conversation": app_state["jarvis_conversation"][-20:],
        }
    )


@app.route("/api/jarvis/clear", methods=["POST"])
def clear_jarvis_conversation():
    """Clear JARVIS conversation history"""
    app_state["jarvis_conversation"] = []
    add_log("JARVIS conversation cleared")
    return jsonify({"success": True})


# =============================================================================
# VOICE API - Speech-to-Text, Translation, Text-to-Speech
# =============================================================================


@app.route("/api/voice/translate", methods=["POST"])
def translate_text():
    """Translate text between languages"""
    data = request.get_json() or {}
    text = data.get("text", "")
    source_lang = data.get("source_lang", "en")
    target_lang = data.get("target_lang", "en")

    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400

    try:
        processor = get_fallback_processor()

        if target_lang == "en":
            translated = processor.translate_to_english(text, source_lang)
        else:
            translated = processor.translate_from_english(text, target_lang)

        return jsonify(
            {
                "success": True,
                "original_text": text,
                "translated_text": translated,
                "source_lang": source_lang,
                "target_lang": target_lang,
            }
        )
    except Exception as e:
        add_log(f"Translation error: {str(e)}", "error")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/voice/tts", methods=["POST"])
def text_to_speech():
    """Convert text to speech audio - Returns base64 encoded MP3 audio"""
    data = request.get_json() or {}
    text = data.get("text", "")
    language = data.get("language", "en")

    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400

    try:
        processor = get_fallback_processor()
        audio_base64 = processor.generate_speech(text, language)

        if audio_base64:
            return jsonify(
                {"success": True, "audio": audio_base64, "language": language}
            )
        else:
            return jsonify(
                {
                    "success": False,
                    "error": "TTS not available - use browser speech synthesis",
                }
            ), 503
    except Exception as e:
        add_log(f"TTS error: {str(e)}", "error")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/voice/chat", methods=["POST"])
def voice_chat():
    """Full voice chat pipeline with translation and TTS"""
    data = request.get_json() or {}
    text = data.get("text", "")
    user_language = data.get("language", app_state.get("language", "en"))

    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400

    # Check if API key is configured
    if not os.getenv("OPENROUTER_API_KEY"):
        error_msg = "OPENROUTER_API_KEY not configured. Please add it to .env file."
        add_log(error_msg, "error")
        return jsonify({"success": False, "error": error_msg}), 503

    add_log(f"🎤 Voice input ({user_language}): {text[:50]}...")

    try:
        processor = get_fallback_processor()

        # Translate to English if needed
        english_text = text
        if user_language != "en":
            english_text = processor.translate_to_english(text, user_language)

        # Process with JARVIS
        camera = get_camera()
        orchestrator = get_orchestrator(camera, None)
        jarvis_result = orchestrator.process_input(english_text, "en")
        english_response = jarvis_result.get("response", "I didn't understand that.")

        # Translate response back
        translated_response = english_response
        if user_language != "en":
            translated_response = processor.translate_from_english(
                english_response, user_language
            )

        # Generate TTS audio
        audio_base64 = processor.generate_speech(translated_response, user_language)

        # Store conversation
        app_state["jarvis_conversation"].append(
            {
                "role": "user",
                "content": text,
                "language": user_language,
                "timestamp": datetime.now().isoformat(),
            }
        )
        app_state["jarvis_conversation"].append(
            {
                "role": "assistant",
                "content": translated_response,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Update robot status
        arm_command = jarvis_result.get("arm_command")
        if arm_command == "pick":
            target = jarvis_result.get("arm_target", "object")
            update_robot_status("PICKING", target.upper())
        elif arm_command == "place":
            update_robot_status("PLACING")
        elif arm_command == "snap":
            update_robot_status("GREETING")
        elif arm_command == "call_mom":
            update_robot_status("CALLING")
            call_result = call_mom()
            add_log(f"📞 {call_result.get('message', 'Calling mom...')}", "success")
        elif arm_command == "call_dad":
            update_robot_status("CALLING")
            call_result = call_dad()
            add_log(f"📞 {call_result.get('message', 'Calling dad...')}", "success")

        # Return response even without audio (browser will use speech synthesis)
        return jsonify(
            {
                "success": True,
                "response": translated_response,
                "audio": audio_base64,  # May be None, browser will fallback to its TTS
                "arm_action": jarvis_result.get("arm_action"),
                "language": user_language,
                "use_browser_tts": audio_base64 is None,  # Signal to use browser TTS
            }
        )

    except Exception as e:
        add_log(f"Voice chat error: {str(e)}", "error")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/voice/status")
def voice_status():
    """Check if voice processing is available"""
    try:
        processor = get_fallback_processor()
        has_translate = processor.translate_client is not None
        has_tts = processor.tts_client is not None

        return jsonify(
            {
                "available": has_translate or has_tts,
                "translation": has_translate,
                "tts": has_tts,
            }
        )
    except Exception as e:
        return jsonify({"available": False, "translation": False, "tts": False})


# =============================================================================
# SETTINGS API
# =============================================================================


@app.route("/api/settings")
def get_settings():
    """Get current settings"""
    return jsonify(
        {
            "language": app_state["language"],
            "dark_mode": app_state["dark_mode"],
            "translations": get_translations(app_state["language"]),
        }
    )


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update settings"""
    data = request.get_json() or {}

    if "language" in data:
        lang = data["language"]
        if lang in TRANSLATIONS:
            app_state["language"] = lang
            add_log(f"Language changed to: {lang}")

    if "dark_mode" in data:
        app_state["dark_mode"] = bool(data["dark_mode"])
        add_log(f"Dark mode: {'enabled' if app_state['dark_mode'] else 'disabled'}")

    return jsonify(
        {
            "success": True,
            "language": app_state["language"],
            "dark_mode": app_state["dark_mode"],
            "translations": get_translations(app_state["language"]),
        }
    )


# =============================================================================
# TRAJECTORY REPLAY
# =============================================================================


@app.route("/api/vla/status", methods=["GET"])
def vla_status():
    """Get VLA (Vision-Language-Action) controller status"""
    try:
        vla = get_vla_controller()
        status = vla.get_status()
        return jsonify({"success": True, "vla": status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/vla/predict", methods=["POST"])
def vla_predict_action():
    """
    Generate robotic action using VLA model
    Accepts: camera image + natural language instruction
    Returns: predicted servo commands
    """
    data = request.get_json() or {}
    instruction = data.get("instruction", "")
    
    if not instruction:
        return jsonify({"success": False, "error": "instruction required"}), 400
    
    try:
        camera = get_camera()
        frame = camera.get_raw_frame()
        
        if frame is None:
            return jsonify({"success": False, "error": "No camera frame available"}), 400
        
        vla = get_vla_controller()
        result = vla.predict_action(frame, instruction)
        
        add_log(f"🤖 VLA prediction: {instruction[:50]}...")
        
        return jsonify({
            "success": result['success'],
            "action": result,
            "mode": "vla" if vla.is_initialized else "trajectory_fallback"
        })
    
    except Exception as e:
        add_log(f"VLA prediction error: {str(e)}", "error")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/vla/reset", methods=["POST"])
def vla_reset():
    """Reset VLA controller to home position"""
    try:
        vla = get_vla_controller()
        vla.reset()
        add_log("VLA controller reset to home")
        return jsonify({"success": True, "message": "VLA reset to home"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/trajectories/list", methods=["GET"])
def list_trajectories():
    """List available trajectory files"""
    replayer = get_replayer()
    trajectories = replayer.list_trajectories()
    return jsonify({"success": True, "trajectories": trajectories})


@app.route("/api/trajectories/replay", methods=["POST"])
def replay_trajectory():
    """Start trajectory replay"""
    data = request.get_json()
    trajectory_file = data.get("trajectory_file")

    if not trajectory_file:
        return jsonify({"success": False, "error": "trajectory_file required"}), 400

    replayer = get_replayer()
    success = replayer.replay(trajectory_file)

    if success:
        add_log(f"Starting trajectory replay: {trajectory_file}")
        return jsonify(
            {"success": True, "message": f"Replay started: {trajectory_file}"}
        )
    else:
        error_msg = replayer.last_error or "Unknown error"
        add_log(f"Failed to start replay: {error_msg}", "error")
        return jsonify({"success": False, "error": error_msg}), 400


@app.route("/api/trajectories/status", methods=["GET"])
def trajectory_status():
    """Get trajectory replay status"""
    replayer = get_replayer()
    status = replayer.get_status()
    return jsonify({"success": True, "status": status})


@app.route("/api/trajectories/stop", methods=["POST"])
def stop_trajectory():
    """Stop current trajectory replay"""
    replayer = get_replayer()
    replayer.stop_replay()
    add_log("Trajectory replay stopped")
    return jsonify({"success": True, "message": "Replay stopped"})


@app.route("/api/trajectories/upload", methods=["POST"])
def upload_trajectory():
    """Upload a new trajectory file"""
    from werkzeug.utils import secure_filename

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not file.filename.endswith(".json"):
        return jsonify({"success": False, "error": "File must be JSON"}), 400

    filename = secure_filename(file.filename)
    trajectories_dir = "trajectories"
    os.makedirs(trajectories_dir, exist_ok=True)

    filepath = os.path.join(trajectories_dir, filename)
    file.save(filepath)

    add_log(f"Trajectory uploaded: {filename}")
    return jsonify(
        {"success": True, "message": f"Uploaded: {filename}", "filename": filename}
    )


# =============================================================================
# STARTUP
# =============================================================================


def initialize():
    """Initialize application"""
    add_log("Bedside Healthcare Assistant Robot v1.0 starting...")
    add_log("Voice-controlled assistance for daily activities")

    # Set up hazard detection
    setup_hazard_detection()

    # List available cameras first
    add_log("Scanning for cameras...")
    cameras = CameraHandler.list_available_cameras(max_devices=10)
    if cameras:
        add_log(f"Found {len(cameras)} camera(s): {[c['name'] for c in cameras]}")
        # Try to start the first available camera
        for cam in cameras:
            reset_camera()
            camera = get_camera(cam['id'])
            if camera.start():
                add_log(f"✓ Camera /dev/video{cam['id']} connected successfully")
                break
        else:
            add_log("Failed to start any camera", "warning")
    else:
        add_log(
            "No cameras detected - use /api/camera/select to connect manually",
            "warning",
        )

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key and len(api_key) > 10:
        add_log("✓ OpenRouter API key configured")
        add_log("✓ JARVIS LLM agent ready for conversation")
    else:
        add_log("⚠ WARNING: OPENROUTER_API_KEY not set in .env", "warning")
        add_log("⚠ LLM chat will not work without API key - set OPENROUTER_API_KEY in .env file", "error")

    # Initialize servo handler for trajectory replay
    try:
        from servo_handler import get_servo_controller
        from trajectory_agent import get_replayer

        servo_ctrl = get_servo_controller()
        if servo_ctrl.connect():
            add_log("✓ Servo controller connected to /dev/ttyACM0")
        else:
            add_log("Servo controller running in simulation mode", "warning")

        replayer = get_replayer()
        add_log("✓ Trajectory replayer initialized")

    except Exception as e:
        add_log(f"Error initializing servo controller: {e}", "warning")
        add_log("Trajectory replay will run in simulation mode", "warning")

    # Initialize VLA (Vision-Language-Action) controller
    try:
        vla = get_vla_controller()
        status = vla.get_status()
        if status['initialized']:
            add_log(f"✓ VLA Model loaded: {status['model']}")
            add_log("✓ pi0.5 Vision-Language-Action controller ready")
        else:
            add_log("VLA controller using trajectory fallback mode", "warning")
    except Exception as e:
        add_log(f"VLA controller initialization: {e}", "warning")

    add_log("System ready - Say hello to your assistant!")


if __name__ == "__main__":
    initialize()
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
