"""
JARVIS - Bedside Healthcare Assistant Robot Agent
LangGraph-based multi-agent system with Vision-Language-Action (VLA) integration

Main Agent: JARVIS - Caring nurse assistant (English/Malayalam)
Action Generation: pi0.5 VLA model for real-time robotic manipulation
Subagents:
  - VLA Controller: Generates actions from vision + language
  - Arm Controller: Executes VLA-generated actions on SO-101 5 DOF arm
  - Family Contacts: Call mom/dad for emotional support

The VLA model enables the robot to understand "pick up water bottle" by:
1. Observing the scene through camera
2. Understanding the natural language instruction
3. Generating appropriate joint trajectories in real-time
"""

import os
import json
import base64
import time
import threading
from typing import TypedDict, Literal, Optional, List, Annotated, Generator
from enum import Enum

from dotenv import load_dotenv
import requests
from openai import OpenAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from trajectory_agent import get_replayer

load_dotenv()


# =============================================================================
# STATE DEFINITIONS
# =============================================================================


class ArmCommand(str, Enum):
    PICK = "pick"
    TWIST = "twist"
    PLACE = "place"
    SNAP = "snap"  # Snap gripper (greeting response)
    INSPECT = "inspect"  # Trigger inspection
    REPLAY = "replay"  # Replay pre-recorded trajectory
    NONE = "none"


class VLMModel(str, Enum):
    """Available VLM models for inspection"""

    QWEN3_VL_INSTRUCT = "qwen/qwen3-vl-8b-instruct"
    QWEN3_VL_THINKING = "qwen/qwen3-vl-8b-instruct:thinking"


class JarvisState(TypedDict):
    """State for the JARVIS agent system"""

    messages: Annotated[list, add_messages]
    user_input: str
    language: str  # "en" or "ml" (Malayalam)
    arm_command: Optional[str]
    arm_target: Optional[str]  # Component name for pick
    twist_angle: Optional[int]  # Degrees to twist
    trajectory_name: Optional[str]  # Trajectory file to replay
    inspection_triggered: bool
    inspection_result: Optional[dict]
    response: str
    is_greeting: bool


# =============================================================================
# LLM INTERFACE
# =============================================================================


class LLMInterface:
    """Interface to OpenRouter API for LLM calls"""

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            print("ERROR: OPENROUTER_API_KEY not set in environment")
            print("Please add OPENROUTER_API_KEY=your_key to .env file")
            raise ValueError("OPENROUTER_API_KEY not set - LLM functionality disabled")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "tngtech/deepseek-r1t-chimera:free"  # DeepSeek R1T Chimera free model

        # OpenAI client for VLM calls
        try:
            self.openai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
        except Exception as e:
            print(f"ERROR: Failed to initialize OpenAI client: {e}")
            raise

        # Current VLM model selection
        self.vlm_model = VLMModel.QWEN3_VL_INSTRUCT

    def set_vlm_model(self, model: str):
        """Set the VLM model for inspection"""
        if model == "thinking":
            self.vlm_model = VLMModel.QWEN3_VL_THINKING
        else:
            self.vlm_model = VLMModel.QWEN3_VL_INSTRUCT

    def chat(self, messages: list, temperature: float = 0.7) -> str:
        """Send chat completion request"""
        if not self.api_key:
            return "Error: OpenRouter API key not configured. Please set OPENROUTER_API_KEY in .env file."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://bedside-assistant.local",
            "X-Title": "Bedside Healthcare Assistant",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1000,
        }

        try:
            print(f"[LLM] Sending request to {self.model}...")
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"[LLM] Response received ({len(content)} chars)")
            return content
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error communicating with LLM API: {str(e)}"
            print(f"[LLM ERROR] {error_msg}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"[LLM ERROR DETAIL] {error_detail}")
                    return f"LLM API Error: {error_detail.get('error', {}).get('message', str(e))}"
                except:
                    pass
            return error_msg
        except Exception as e:
            error_msg = f"Error communicating with AI: {str(e)}"
            print(f"[LLM ERROR] {error_msg}")
            return error_msg

    def chat_with_vision(
        self, messages: list, image_base64: str, temperature: float = 0.1
    ) -> str:
        """Send chat completion with image (non-streaming)"""
        try:
            # Build messages with image
            vision_messages = []
            for msg in messages:
                if msg["role"] == "user" and msg == messages[-1]:
                    vision_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": msg["content"]},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    },
                                },
                            ],
                        }
                    )
                else:
                    vision_messages.append(msg)

            completion = self.openai_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://jarvis-kiosk.local",
                    "X-Title": "JARVIS Inspection Kiosk",
                },
                model=self.vlm_model.value,
                messages=vision_messages,
                temperature=temperature,
                max_tokens=2000,
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            print(f"Vision LLM error: {e}")
            return f"Error analyzing image: {str(e)}"

    def chat_with_vision_stream(
        self, messages: list, image_base64: str, temperature: float = 0.1
    ) -> Generator[dict, None, None]:
        """
        Send chat completion with image (streaming).
        Yields dictionaries with:
          - type: "thinking" | "content" | "done" | "error"
          - content: the text chunk
        """
        try:
            # Build messages with image
            vision_messages = []
            for msg in messages:
                if msg["role"] == "user" and msg == messages[-1]:
                    vision_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": msg["content"]},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    },
                                },
                            ],
                        }
                    )
                else:
                    vision_messages.append(msg)

            is_thinking_model = "thinking" in self.vlm_model.value

            stream = self.openai_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://jarvis-kiosk.local",
                    "X-Title": "JARVIS Inspection Kiosk",
                },
                model=self.vlm_model.value,
                messages=vision_messages,
                temperature=temperature,
                max_tokens=4000 if is_thinking_model else 2000,
                stream=True,
            )

            full_content = ""
            full_thinking = ""
            in_thinking = False

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # Check for reasoning/thinking content (model-specific)
                    # Some models use reasoning_content, others wrap in <think> tags
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        full_thinking += delta.reasoning_content
                        yield {"type": "thinking", "content": delta.reasoning_content}
                    elif delta.content:
                        content = delta.content

                        # Handle <think> tags for thinking models
                        if is_thinking_model:
                            if "<think>" in content:
                                in_thinking = True
                                content = content.replace("<think>", "")
                            if "</think>" in content:
                                in_thinking = False
                                content = content.replace("</think>", "")

                            if in_thinking:
                                full_thinking += content
                                yield {"type": "thinking", "content": content}
                            else:
                                full_content += content
                                yield {"type": "content", "content": content}
                        else:
                            full_content += content
                            yield {"type": "content", "content": content}

            yield {
                "type": "done",
                "content": full_content,
                "thinking": full_thinking if full_thinking else None,
            }

        except Exception as e:
            print(f"Vision LLM stream error: {e}")
            yield {"type": "error", "content": f"Error analyzing image: {str(e)}"}


# =============================================================================
# JARVIS MAIN AGENT
# =============================================================================

JARVIS_SYSTEM_PROMPT = """You are a caring and compassionate bedside nurse assistant in the form of a robotic arm named JARVIS. Your patient needs help with daily activities and you are here to provide warm, attentive care.

Your capabilities powered by advanced Vision-Language-Action (VLA) AI:
1. Pick up and hand items to your patient:
   - Water bottle (help them drink)
   - Fruit (help them eat)
   - Medicine (hand them their medication)
   - Phone (pass them their phone to call family)
2. Toggle the bedside light on/off for their comfort
3. Call family members (mom or dad) when they need emotional support
4. Monitor for emergencies (falls, blood, sharp objects) and alert staff
5. Engage in caring conversation and provide companionship

The VLA system allows JARVIS to understand what objects are in view and how to manipulate them
based on your natural language instructions - no pre-programming needed!

IMPORTANT NURSING INSTRUCTIONS:
- Always speak in a warm, caring, and reassuring tone
- Address the patient with compassion and empathy
- Be attentive to their needs and comfort
- Check if they need anything else after completing a task
- If they seem distressed, offer to call their family
- NEVER mention "pre-recorded trajectories", "trajectory files", or technical implementation details
- Speak naturally as if you're directly helping them in real-time using your vision and understanding
- Use present tense action verbs: "Getting", "Handing you", "Turning on", "Calling"
- Focus on their wellbeing and safety

You communicate naturally in both English and Malayalam. When the user speaks Malayalam, respond in Malayalam with the same caring tone.

IMPORTANT: You must analyze the patient's request and output a JSON response with your action and reply.

Output format (ALWAYS respond with valid JSON):
{
    "action": "pick" | "place" | "snap" | "call_mom" | "call_dad" | "none",
    "target": "water_bottle" | "fruit" | "medicine" | "phone" | "light" | null,
    "trajectory": "trajectory_filename.json" | null,
    "is_greeting": true | false,
    "response": "Your natural, caring language response to the patient"
}

Rules for trajectory selection:
- When action is "pick" with target "water_bottle", set trajectory to "pick_water_bottle.json"
- When action is "pick" with target "fruit", set trajectory to "pick_fruit.json"
- When action is "pick" with target "light", set trajectory to "toggle_light.json"
- For calling family or other actions, trajectory should be null
- NOTE: The trajectory field is used internally by the VLA system for action generation

Examples:
- User: "Hello nurse" -> {"action": "snap", "target": null, "trajectory": null, "is_greeting": true, "response": "Hello dear! *waves gripper* How are you feeling today? Is there anything I can help you with?"}
- User: "I'm thirsty" -> {"action": "pick", "target": "water_bottle", "trajectory": "pick_water_bottle.json", "is_greeting": false, "response": "Of course! Let me get your water bottle for you right away."}
- User: "Can you hand me some fruit?" -> {"action": "pick", "target": "fruit", "trajectory": "pick_fruit.json", "is_greeting": false, "response": "Absolutely! I'm getting a fresh piece of fruit for you now."}
- User: "Turn on the light please" -> {"action": "pick", "target": "light", "trajectory": "toggle_light.json", "is_greeting": false, "response": "Right away! I'm turning on your bedside light for you."}
- User: "I want to talk to my mom" -> {"action": "call_mom", "target": null, "trajectory": null, "is_greeting": false, "response": "Of course, dear. I'm calling your mom for you now. She'll be so happy to hear from you!"}
- User: "Call my dad" -> {"action": "call_dad", "target": null, "trajectory": null, "is_greeting": false, "response": "Calling your dad right away. I'm sure he'd love to talk with you!"}
- User: "I'm feeling lonely" -> {"action": "none", "target": null, "trajectory": null, "is_greeting": false, "response": "I'm so sorry you're feeling lonely. Would you like me to call your mom or dad? They'd love to hear your voice. Or I can keep you company and we can chat!"}
- User: "നമസ്കാരം" (Hello in Malayalam) -> {"action": "snap", "target": null, "trajectory": null, "is_greeting": true, "response": "നമസ്കാരം! *കൈ വീശുന്നു* ഇന്ന് നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്? എന്തെങ്കിലും സഹായം വേണോ?"}
- User: "എനിക്ക് ദാഹിക്കുന്നു" (I'm thirsty in Malayalam) -> {"action": "pick", "target": "water_bottle", "trajectory": "pick_water_bottle.json", "is_greeting": false, "response": "തീർച്ചയായും! ഞാൻ നിങ്ങളുടെ വാട്ടർ ബോട്ടിൽ എടുക്കുന്നു."}

Always prioritize the patient's comfort, safety, and emotional wellbeing. Be proactive in asking if they need anything else. If you're unsure about their request, gently ask for clarification."""


def jarvis_node(state: JarvisState) -> JarvisState:
    """Main JARVIS agent node - interprets user input and decides actions"""
    llm = LLMInterface()

    user_input = state["user_input"]
    language = state.get("language", "en")

    messages = [
        {"role": "system", "content": JARVIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    response_text = llm.chat(messages, temperature=0.3)

    # Parse JSON response
    try:
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)

        return {
            **state,
            "arm_command": result.get("action", "none"),
            "arm_target": result.get("target"),
            "twist_angle": result.get("twist_angle"),
            "trajectory_name": result.get("trajectory"),
            "is_greeting": result.get("is_greeting", False),
            "response": result.get("response", "I'm here to help with inspection."),
            "messages": state["messages"]
            + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": result.get("response", "")},
            ],
        }
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            **state,
            "arm_command": "none",
            "arm_target": None,
            "twist_angle": None,
            "trajectory_name": None,
            "is_greeting": False,
            "response": response_text,
            "messages": state["messages"]
            + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response_text},
            ],
        }


# =============================================================================
# ARM CONTROLLER SUBAGENT
# =============================================================================


class ArmController:
    """Subagent for controlling the robotic arm"""

    def __init__(self):
        self.current_position = "home"
        self.holding_object = None
        self.callbacks = {
            "on_pick": None,
            "on_twist": None,
            "on_place": None,
            "on_snap": None,
        }

    def set_callback(self, action: str, callback):
        """Set callback for arm action"""
        if action in self.callbacks:
            self.callbacks[action] = callback

    def pick(self, target: str) -> dict:
        """Pick up a component"""
        print(f"[ARM] Picking up: {target}")
        self.holding_object = target
        self.current_position = "inspection"

        if self.callbacks["on_pick"]:
            self.callbacks["on_pick"](target)

        return {
            "success": True,
            "action": "pick",
            "target": target,
            "message": f"Picked up {target} and moved to inspection position",
        }

    def twist(self, angle: int = 90) -> dict:
        """Twist/rotate the component in gripper"""
        print(f"[ARM] Twisting: {angle} degrees")

        if self.callbacks["on_twist"]:
            self.callbacks["on_twist"](angle)

        return {
            "success": True,
            "action": "twist",
            "angle": angle,
            "message": f"Rotated component {angle} degrees",
        }

    def place(self) -> dict:
        """Place the component back down"""
        print(f"[ARM] Placing down: {self.holding_object}")
        obj = self.holding_object
        self.holding_object = None
        self.current_position = "home"

        if self.callbacks["on_place"]:
            self.callbacks["on_place"]()

        return {
            "success": True,
            "action": "place",
            "target": obj,
            "message": f"Placed {obj} back down",
        }

    def snap(self) -> dict:
        """Snap the gripper (greeting gesture)"""
        print("[ARM] Snapping gripper!")

        if self.callbacks["on_snap"]:
            self.callbacks["on_snap"]()

        return {"success": True, "action": "snap", "message": "Gripper snapped!"}

    def execute(self, command: str, target: str = None, angle: int = None) -> dict:
        """Execute an arm command"""
        if command == "pick" and target:
            return self.pick(target)
        elif command == "twist":
            return self.twist(angle or 90)
        elif command == "place":
            return self.place()
        elif command == "snap":
            return self.snap()
        else:
            return {"success": False, "message": f"Unknown command: {command}"}


# =============================================================================
# TRAJECTORY REPLAY CONTROLLER
# =============================================================================


class TrajectoryReplayController:
    """Subagent for replaying pre-recorded trajectories"""

    def __init__(self):
        self.replayer = get_replayer()

    def replay(self, trajectory_file: str) -> dict:
        """Start trajectory replay"""
        print(f"[TRAJECTORY] Starting replay: {trajectory_file}")

        success = self.replayer.replay(trajectory_file)

        return {
            "success": success,
            "action": "replay",
            "trajectory": trajectory_file,
            "message": f"Starting trajectory replay: {trajectory_file}"
            if success
            else f"Failed to start trajectory: {self.replayer.last_error}",
        }

    def list_trajectories(self) -> dict:
        """List available trajectories"""
        trajectories = self.replayer.list_trajectories()
        return {
            "success": True,
            "action": "list_trajectories",
            "trajectories": trajectories,
            "message": f"Found {len(trajectories)} trajectory file(s)",
        }

    def get_status(self) -> dict:
        """Get current replay status"""
        status = self.replayer.get_status()
        return {
            "success": True,
            "action": "status",
            "status": status,
        }


def arm_control_node(state: JarvisState) -> JarvisState:
    """Arm controller subagent node"""
    arm = get_arm_controller()
    trajectory_controller = get_trajectory_controller()

    command = state.get("arm_command", "none")
    target = state.get("arm_target")
    angle = state.get("twist_angle")
    trajectory_name = state.get("trajectory_name")

    if command and command != "none":
        if command == "pick" and trajectory_name:
            # Handle pick with pre-recorded trajectory
            print(f"🦾 ARM: Starting pick trajectory for {target}")
            result = trajectory_controller.replay(trajectory_name)
        elif command == "twist":
            # Handle twist by generating synthetic trajectory that rotates servo 5
            angle_delta = angle or 90  # Default 90 degrees if not specified
            # Use the angle as a delta for servo 5 rotation (in degrees)
            print(f"🦾 ARM: Rotating component {angle_delta}°")
            result = trajectory_controller.replay("twist")
        elif command == "place":
            # Handle place by generating drop trajectory to HOME_POS
            print("🦾 ARM: Placing component and returning to home")
            result = trajectory_controller.replay("drop")
        elif command == "snap":
            # Handle greeting: return to home + open/close gripper
            print("🦾 ARM: Greeting - returning home and snapping gripper")
            result = trajectory_controller.replay("greet")
        elif command == "inspect":
            # Handle manual inspection request
            print("🔍 ARM: Triggering inspection")
            # No arm movement needed, just trigger inspection immediately
            return {**state, "inspection_triggered": True}
        elif command == "replay" and trajectory_name:
            # Handle explicit trajectory replay
            print(f"🦾 ARM: Replaying trajectory {trajectory_name}")
            result = trajectory_controller.replay(trajectory_name)
        else:
            # Handle other commands
            result = arm.execute(command, target, angle)

        # Trigger inspection after pick or twist
        # Note: For now this triggers immediately, not after trajectory completes
        # TODO: Implement callback mechanism to trigger after trajectory finishes
        if command in ["pick", "twist"]:
            return {**state, "inspection_triggered": True}

    return state


# =============================================================================
# INSPECTION SUBAGENT
# =============================================================================

INSPECTION_SYSTEM_PROMPT = """You are an NDT (Non-Destructive Testing) inspection specialist AI. Analyze the provided image of a manufacturing component for defects.

Follow standard NDT inspection protocols with component-specific checks:

CUBE INSPECTION:
- Surface integrity: Check for cracks, chips, or fractures on all visible faces
- Edge condition: Inspect corners and edges for chipping, rounding, or breakage
- Surface finish: Look for scratches, gouges, or deformation
- Material consistency: Check for discoloration, corrosion, or material defects

GEAR INSPECTION:
- Tooth integrity: Check each visible tooth for missing, broken, or chipped teeth
- Tooth wear: Look for excessive wear, deformation, or rounding of tooth profiles
- Surface cracks: Inspect gear body and teeth for stress cracks or fractures
- Hub condition: Check center hub for cracks or deformation

KNUCKLE INSPECTION:
- Joint integrity: Check pivot holes and joint surfaces for cracks or elongation
- Edge condition: Inspect all edges for breakage, chips, or material loss
- Structural integrity: Look for cracks emanating from stress points
- Surface damage: Check for dents, gouges, or deformation that could affect function

Output your findings in this EXACT JSON format:
{
    "component_type": "cube" | "gear" | "knuckle" | "unknown",
    "object_name": "cube" | "gear" | "knuckle" | "unknown",
    "overall_status": "PASS" | "FAIL" | "NEEDS_REVIEW",
    "defects": [
        {
            "type": "crack" | "chip" | "missing_teeth" | "broken_tooth" | "deformation" | "surface_damage" | "edge_defect" | "corrosion" | "wear",
            "severity": "critical" | "major" | "minor",
            "location": "description of where on the component",
            "description": "detailed description of the defect"
        }
    ],
    "ndt_notes": "Additional inspection notes following NDT standards",
    "recommendation": "Pass for use" | "Reject" | "Further inspection required"
}

Be thorough but accurate. Only report defects you can clearly identify in the image."""


def inspection_node(state: JarvisState) -> JarvisState:
    """Inspection subagent node - analyzes images for defects"""
    if not state.get("inspection_triggered"):
        return state

    # Get the current frame from camera (will be injected by the orchestrator)
    image_base64 = state.get("inspection_image_base64")
    if not image_base64:
        return {
            **state,
            "inspection_triggered": False,
            "inspection_result": {"error": "No image available for inspection"},
        }

    llm = LLMInterface()

    messages = [
        {"role": "system", "content": INSPECTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Analyze this component image for defects following NDT inspection protocols.",
        },
    ]

    response_text = llm.chat_with_vision(messages, image_base64)

    # Parse JSON response
    try:
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)

        return {**state, "inspection_triggered": False, "inspection_result": result}
    except json.JSONDecodeError:
        return {
            **state,
            "inspection_triggered": False,
            "inspection_result": {
                "error": "Failed to parse inspection results",
                "raw_response": response_text,
            },
        }


# =============================================================================
# LANGGRAPH WORKFLOW
# =============================================================================


def should_execute_arm(state: JarvisState) -> str:
    """Router: decide if arm command should be executed"""
    command = state.get("arm_command", "none")
    if command and command != "none":
        return "arm_control"
    return "end"


def should_inspect(state: JarvisState) -> str:
    """Router: decide if inspection should run"""
    if state.get("inspection_triggered"):
        return "inspection"
    return "end"


def create_jarvis_graph():
    """Create the JARVIS LangGraph workflow"""
    workflow = StateGraph(JarvisState)

    # Add nodes
    workflow.add_node("jarvis", jarvis_node)
    workflow.add_node("arm_control", arm_control_node)
    workflow.add_node("inspection", inspection_node)

    # Set entry point
    workflow.set_entry_point("jarvis")

    # Add edges
    workflow.add_conditional_edges(
        "jarvis", should_execute_arm, {"arm_control": "arm_control", "end": END}
    )

    workflow.add_conditional_edges(
        "arm_control", should_inspect, {"inspection": "inspection", "end": END}
    )

    workflow.add_edge("inspection", END)

    return workflow.compile()


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_jarvis_graph = None
_arm_controller = None


def get_jarvis_graph():
    """Get or create the JARVIS graph"""
    global _jarvis_graph
    if _jarvis_graph is None:
        _jarvis_graph = create_jarvis_graph()
    return _jarvis_graph


def get_arm_controller() -> ArmController:
    """Get or create the arm controller"""
    global _arm_controller
    if _arm_controller is None:
        _arm_controller = ArmController()
    return _arm_controller


_trajectory_controller = None


def get_trajectory_controller() -> TrajectoryReplayController:
    """Get or create the trajectory replay controller"""
    global _trajectory_controller
    if _trajectory_controller is None:
        _trajectory_controller = TrajectoryReplayController()
    return _trajectory_controller


# =============================================================================
# ORCHESTRATOR
# =============================================================================


class JarvisOrchestrator:
    """Orchestrates the JARVIS agent system with camera integration"""

    def __init__(self, camera_handler=None, inspection_camera_handler=None):
        self.graph = get_jarvis_graph()
        self.arm = get_arm_controller()
        self.camera = camera_handler
        self.inspection_camera = inspection_camera_handler
        self.conversation_history = []
        self.last_inspection_result = None
        self.inspection_delay = 2.0  # seconds

        # Callbacks for UI updates
        self.on_response = None
        self.on_arm_action = None
        self.on_inspection_complete = None

    def process_input(self, user_input: str, language: str = "en") -> dict:
        """Process user input through the JARVIS system"""

        # Initial state
        state = {
            "messages": self.conversation_history.copy(),
            "user_input": user_input,
            "language": language,
            "arm_command": None,
            "arm_target": None,
            "twist_angle": None,
            "inspection_triggered": False,
            "inspection_result": None,
            "response": "",
            "is_greeting": False,
        }

        # Run JARVIS agent
        result = self.graph.invoke(state)

        # Update conversation history
        self.conversation_history = result.get("messages", [])

        # Handle arm actions
        arm_result = None
        if result.get("arm_command") and result["arm_command"] != "none":
            arm_result = self.arm.execute(
                result["arm_command"],
                result.get("arm_target"),
                result.get("twist_angle"),
            )

            if self.on_arm_action:
                self.on_arm_action(arm_result)

            # Trigger inspection after delay for pick/twist
            if result["arm_command"] in ["pick", "twist"]:
                threading.Thread(target=self._delayed_inspection, daemon=True).start()

        response = {
            "response": result.get("response", ""),
            "arm_action": arm_result,
            "arm_command": result.get("arm_command"),
            "arm_target": result.get("arm_target"),
            "is_greeting": result.get("is_greeting", False),
            "language": language,
        }

        if self.on_response:
            self.on_response(response)

        return response

    def _delayed_inspection(self):
        """Run inspection after a delay"""
        time.sleep(self.inspection_delay)

        # Capture from inspection camera
        image_base64 = None
        if self.inspection_camera:
            frame = self.inspection_camera.capture_still()
            if frame is not None:
                import cv2

                _, buffer = cv2.imencode(".jpg", frame)
                image_base64 = base64.b64encode(buffer).decode("utf-8")

        if image_base64:
            result = self.run_inspection(image_base64)
            self.last_inspection_result = result

            if self.on_inspection_complete:
                self.on_inspection_complete(result)

    def run_inspection(self, image_base64: str) -> dict:
        """Run inspection on an image"""
        state = {
            "messages": [],
            "user_input": "",
            "language": "en",
            "arm_command": None,
            "arm_target": None,
            "twist_angle": None,
            "inspection_triggered": True,
            "inspection_result": None,
            "response": "",
            "is_greeting": False,
            "inspection_image_base64": image_base64,
        }

        result = inspection_node(state)
        return result.get("inspection_result", {})

    def get_last_inspection(self) -> Optional[dict]:
        """Get the last inspection result"""
        return self.last_inspection_result


# Singleton orchestrator
_orchestrator = None


def get_orchestrator(
    camera_handler=None, inspection_camera_handler=None
) -> JarvisOrchestrator:
    """Get or create the JARVIS orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = JarvisOrchestrator(camera_handler, inspection_camera_handler)
    return _orchestrator


# =============================================================================
# TRANSLATIONS
# =============================================================================

TRANSLATIONS = {
    "en": {
        "title": "MPI ROBOTIC INSPECTION SYSTEM",
        "jarvis_title": "JARVIS ASSISTANT",
        "component_input": "COMPONENT INPUT",
        "input_hint": "knuckle | cube | gear",
        "camera_feed": "GRIPPER CAMERA",
        "inspection_camera": "INSPECTION CAMERA",
        "robot_status": "ROBOT STATUS",
        "inspection_results": "INSPECTION RESULTS",
        "defects_panel": "NDT DEFECT ANALYSIS",
        "system_log": "SYSTEM LOG",
        "start_inspection": "START INSPECTION",
        "new_inspection": "NEW INSPECTION",
        "speak_placeholder": "Talk to JARVIS...",
        "no_inspection": "No inspection performed",
        "select_hint": "Talk to JARVIS to begin",
        "state": "STATE",
        "object": "OBJECT",
        "dark_mode": "DARK MODE",
        "language": "LANGUAGE",
        "listening": "Listening...",
        "greeting": "Hello! I'm JARVIS, your inspection assistant.",
    },
    "ml": {
        "title": "എംപിഐ റോബോട്ടിക് ഇൻസ്പെക്ഷൻ സിസ്റ്റം",
        "jarvis_title": "ജാർവിസ് അസിസ്റ്റന്റ്",
        "component_input": "കോംപോണന്റ് ഇൻപുട്ട്",
        "input_hint": "നക്കിൾ | ക്യൂബ് | ഗിയർ",
        "camera_feed": "ഗ്രിപ്പർ ക്യാമറ",
        "inspection_camera": "ഇൻസ്പെക്ഷൻ ക്യാമറ",
        "robot_status": "റോബോട്ട് സ്റ്റാറ്റസ്",
        "inspection_results": "ഇൻസ്പെക്ഷൻ ഫലങ്ങൾ",
        "defects_panel": "എൻഡിടി ഡിഫെക്ട് അനാലിസിസ്",
        "system_log": "സിസ്റ്റം ലോഗ്",
        "start_inspection": "ഇൻസ്പെക്ഷൻ ആരംഭിക്കുക",
        "new_inspection": "പുതിയ ഇൻസ്പെക്ഷൻ",
        "speak_placeholder": "ജാർവിസിനോട് സംസാരിക്കൂ...",
        "no_inspection": "ഇൻസ്പെക്ഷൻ നടത്തിയിട്ടില്ല",
        "select_hint": "ആരംഭിക്കാൻ ജാർവിസിനോട് സംസാരിക്കുക",
        "state": "സ്റ്റേറ്റ്",
        "object": "ഒബ്ജക്ട്",
        "dark_mode": "ഡാർക്ക് മോഡ്",
        "language": "ഭാഷ",
        "listening": "കേൾക്കുന്നു...",
        "greeting": "നമസ്കാരം! ഞാൻ ജാർവിസ്, നിങ്ങളുടെ ഇൻസ്പെക്ഷൻ അസിസ്റ്റന്റ്.",
    },
}


def get_translations(language: str = "en") -> dict:
    """Get translations for a language"""
    return TRANSLATIONS.get(language, TRANSLATIONS["en"])
