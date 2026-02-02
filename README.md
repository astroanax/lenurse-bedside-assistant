# [AfA2026-PhysicalAI] - VLA-Powered Bedside Assistant

**Team:** Artificial Penguins  
**Institution:** NIT Calicut  
**Program:** AI For All - Physical AI Edition 2026

## Project Description

An advanced **Vision-Language-Action (VLA)** model-based agentic robot system that provides intelligent assistance for bedridden patients. This cutting-edge solution combines state-of-the-art multimodal AI with robotic manipulation, enabling natural voice-controlled assistance that understands both visual context and verbal commands.

**Key Innovation**: Unlike traditional pre-programmed robotic systems, our assistant uses the **pi0.5 VLA model** integrated via **LeRobot** to dynamically generate motion plans based on real-time visual understanding and natural language instructions. The system perceives the environment through the Arduino UNO Q's integrated camera, processes commands through an LLM-powered agentic architecture (DeepSeek R1T Chimera), and executes precise manipulation tasks using a SO-101 5 DOF robotic arm—all while providing real-time feedback through an intuitive web dashboard.

This solution addresses critical needs in healthcare and elderly care, particularly for bedridden patients who require assistance with daily tasks. By leveraging vision-language-action models, the system can adapt to novel situations and objects without requiring pre-programmed trajectories.

## Hardware Lineup

- **Arduino UNO Q** - Main controller with integrated vision capabilities for VLA model inference
- **Integrated Camera** - Real-time visual perception for the Vision-Language-Action model
- **SO-101 5 DOF Robotic Arm** - High-precision robotic manipulator with ST servo motors
- **Modulino Sensors** (if applicable) - Environmental monitoring and safety feedback
- **Microphone** - Natural language voice command input
- **Speaker** - Audio feedback and status updates

## User Interface & Feedback

The system provides multiple interaction modalities:

- **Web Dashboard** - A comprehensive Flask-based interface for monitoring and control
  - Live camera feed with VLA model visualization
  - Real-time action generation and execution monitoring
  - Family contact management
  - System status and agent reasoning display
  
- **Voice Interface** - Natural language command processing powered by LLM agents for truly conversational, hands-free operation

- **Visual Feedback** - Real-time hazard detection alerts and status indicators

- **Physical Feedback** - Robotic arm movements for assistive manipulation tasks

## The AI Model

The system employs a sophisticated multi-modal AI architecture powered by cutting-edge Vision-Language-Action models:

### Vision-Language-Action (VLA) Model - pi0.5

Our core innovation leverages the **pi0.5 VLA model** integrated via **LeRobot** framework:

- **Model**: [shehiin/pi05_pick_red_cube_lora](https://huggingface.co/shehiin/pi05_pick_red_cube_lora) from HuggingFace
- **Framework**: LeRobot - Open-source robotics toolkit for training and deploying VLA policies
- **Core Technology**: End-to-end learning from visual observations and language instructions directly to robot actions
- **Vision Processing**: Real-time scene understanding through the Arduino UNO Q's integrated camera
- **Language Understanding**: Natural language command interpretation with contextual awareness
- **Action Generation**: Dynamic motion planning and execution without pre-programmed trajectories

The pi0.5 VLA model is a transformer-based policy that has been fine-tuned on robotic manipulation tasks. When a user provides the command "hand me the water bottle," the model:
1. Processes the camera image to locate the water bottle
2. Understands the semantic meaning of your instruction
3. Generates a sequence of joint angles for the SO-101 5 DOF arm
4. Executes the motion smoothly and safely

**Key Advantages of VLA over Traditional Robotics:**
- **Zero-shot Generalization**: Can manipulate novel objects not seen during training
- **Natural Language Interface**: No need for complex command syntax
- **Context-Aware**: Understands spatial relationships between objects
- **Adaptive**: Adjusts to changes in object positions in real-time

### LLM-Powered Agentic System

- **Agent Architecture**: Sophisticated reasoning system powered by LLMs via OpenRouter API
- **Multi-step Planning**: Breaks down complex tasks into executable sub-actions
- **Context Awareness**: Maintains conversation history and environmental state
- **Adaptive Behavior**: Learns from interactions and adjusts to novel situations

### Additional AI Capabilities

- **Hazard Detection**: Real-time safety monitoring through vision analysis
- **Object Recognition**: Identifies and localizes objects for manipulation tasks
- **Natural Language Processing**: Conversational interface for intuitive interaction

The VLA model and agentic system work together to enable truly intelligent behavior—the robot does not simply follow pre-programmed paths, but understands what needs to be done and determines how to execute it in real-time. Edge inference on the Arduino UNO Q ensures privacy and low-latency responses.

## Software Architecture

### Core Components

1. **app.py** - Main Flask application server
   - Serves web dashboard
   - Manages WebSocket connections for real-time updates
   - Coordinates between VLA inference and robot control

2. **vla_controller.py** - Vision-Language-Action model interface
   - Loads pi0.5 VLA model from HuggingFace
   - Integrates LeRobot framework for policy execution
   - Preprocesses camera images for model input
   - Generates real-time servo commands from VLA predictions
   - Applies safety constraints and action smoothing

3. **camera.py** - Vision system interface
   - Captures real-time video feed
   - Preprocesses visual input for VLA model
   - Provides visual context for action generation

4. **voice_processor.py** - Natural language interface
   - Captures audio input
   - Converts speech to text
   - Sends commands to LLM agent

5. **servo_handler.py** - SO-101 robotic arm control
   - Low-level servo communication for 5 DOF arm
   - Position management and safety limits
   - Executes VLA-generated action sequences

6. **trajectory_agent.py** - VLA action execution coordinator
   - Interfaces with pi0.5 VLA model via vla_controller
   - Translates VLA outputs to servo commands
   - Ensures smooth and safe motion execution
   - Fallback to cached action sequences for reliability

7. **jarvis_agent.py** - LLM-powered agentic system
   - Connects to OpenRouter API for LLM inference (DeepSeek R1T Chimera)
   - Implements multi-step reasoning and planning
   - Manages conversation context and state
   - Triggers VLA model for physical action generation

8. **hazard_detector.py** - Safety monitoring system
   - Real-time environment analysis using YOLO11
   - Collision detection and prevention
   - Emergency stop triggering

9. **family_contacts.py** - Emergency contact system
   - Quick access to caregiver contacts
   - Emergency notification capabilities

10. **record_replay.py** - Demonstration and debugging tools
    - Records VLA-generated actions for analysis
    - Enables system debugging and refinement
    - Caches successful action sequences

### Communication Flow

```
User Voice Command → Speech-to-Text → LLM Agent (OpenRouter API)
                                              ↓
                                     Intent Understanding
                                     Multi-step Planning
                                              ↓
                        Visual Context ← Camera Feed (Arduino UNO Q)
                                              ↓
                                      pi0.5 VLA Model
                                    (Vision + Language → Action)
                                              ↓
                                    Action Parameters
                                              ↓
                        SO-101 5 DOF Arm → Physical Execution
                                              ↓
                              Feedback (Dashboard/Voice/Visual)
```

### VLA Pipeline

The Vision-Language-Action pipeline enables end-to-end learning from perception to action:

1. **Perception**: Camera captures current scene
2. **Understanding**: VLA model processes visual input + natural language command
3. **Action Generation**: Model outputs joint angles and motion parameters
4. **Execution**: Servo handler translates to motor commands
5. **Monitoring**: Continuous feedback ensures safe and accurate execution

## Project Structure

```
drm_hack/
├── bedside_assistant/          # Main application
│   ├── app.py                  # Flask web server
│   ├── vla_controller.py       # pi0.5 VLA model integration
│   ├── camera.py               # Vision processing
│   ├── voice_processor.py      # Voice commands
│   ├── servo_handler.py        # SO-101 arm control
│   ├── trajectory_agent.py     # VLA action coordinator
│   ├── jarvis_agent.py         # LLM agentic system (DeepSeek R1T)
│   ├── hazard_detector.py      # Safety monitoring (YOLO11)
│   ├── family_contacts.py      # Contact management
│   ├── record_replay.py        # Action recording/debugging
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example            # Environment config template
│   ├── static/                 # Web assets (CSS, JS, sounds)
│   ├── templates/              # HTML templates
│   ├── trajectories/           # Cached action sequences
│   └── models/                 # VLA model cache (auto-created)
│       └── vla_cache/          # HuggingFace model downloads
├── STServo_Python/             # Servo SDK
│   └── ...                     # SO-101 control libraries
└── mpi_kiosk/                  # Kiosk interface (if applicable)
```

## Setup Instructions

### Prerequisites

- Arduino UNO Q board with integrated camera
- Python 3.7 or higher
- SO-101 5 DOF robotic arm with ST servo motors
- OpenRouter API access
- pi0.5 VLA model access

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd AfA2026-ArtPengus
   ```

2. **Install Python dependencies**
   ```bash
   cd drm_hack/bedside_assistant
   pip install -r requirements.txt
   ```
   
   Note: This installs LeRobot, PyTorch, and the VLA model dependencies. 
   First installation may take time as it downloads the pre-trained models (~2GB).

3. **Configure Servo SDK**
   ```bash
   cd ../../STServo_Python
   pip install -r requirements.txt
   ```

4. **Configure API access**
   - Set up OpenRouter API credentials for LLM agent
   - Configure pi0.5 VLA model endpoint
   - Set environment variables for API keys

5. **Configure hardware connections**
   - Connect SO-101 5 DOF robotic arm to appropriate ports
   - Ensure Arduino UNO Q camera is properly connected
   - Configure serial port in servo_handler.py if needed

6. **Run the application**
   ```bash
   cd ../drm_hack/bedside_assistant
   python app.py
   ```
   
   On first run, the VLA model will be downloaded from HuggingFace (~500MB).
   This is cached locally in `models/vla_cache/` for future runs.

7. **Access the web dashboard**
   - Open browser to `http://localhost:5000`
   - Use the interface to control the system

## Usage

### Voice Commands (Natural Language)

The system understands natural, conversational commands thanks to the LLM agent + VLA integration:

- "Can you hand me my water bottle?"
- "I'd like to eat that apple on the table"
- "Please turn off the lamp"
- "Move the medicine bottle closer to me"
- "Help me reach my phone"

**How the system processes commands:**
1. Voice command is processed by the LLM (DeepSeek R1T Chimera)
2. The LLM extracts intent and target object
3. VLA model observes the scene and target through camera
4. VLA generates optimal joint trajectories in real-time
5. SO-101 arm executes the motion smoothly

The VLA model allows the robot to understand these commands in visual context and generate appropriate actions without pre-programming, adapting even when objects are in unexpected positions.

### Web Dashboard
- View live camera feed with VLA model predictions
- Monitor real-time action generation and execution
- Observe agent reasoning and decision-making process
- Track system status and hazard alerts
- Emergency stop and safety controls

### Adaptive Behavior

Unlike traditional robots, this system adapts to:
- Novel object positions and orientations
- Different types of objects not seen during training
- Varying environmental conditions
- User preferences and implicit feedback

## Key Technical Innovations

### Vision-Language-Action Integration with LeRobot
Our implementation uses the LeRobot framework to deploy the pi0.5 VLA model (fine-tuned LoRA variant):

**Model Architecture:**
- Base: pi0.5 transformer-based policy network
- Training: LoRA fine-tuning on manipulation tasks
- Input: RGB images (224x224) + natural language instructions
- Output: 6-DOF action vectors (5 joint angles + gripper)

**Real-time Inference Pipeline:**
```
Camera Frame → Image Preprocessing → VLA Model → Action Smoothing → Servo Commands
     ↓                                    ↑
Language Input → Instruction Encoding ────┘
```

The VLA model bridges the gap between perception, understanding, and action through:
- **End-to-end Learning**: No need for separate perception, planning, and control modules
- **Multimodal Fusion**: Combines visual and linguistic information in a single forward pass
- **Generalization**: Trained on diverse manipulation tasks, can handle novel scenarios
- **Real-time Performance**: Inference runs at 10Hz on Arduino UNO Q

**Why VLA over Traditional Approaches:**
- Traditional: Object Detection → Path Planning → Trajectory Execution (3 separate systems)
- VLA: Single model that goes directly from pixels + text → actions
- Result: Faster, more robust, and adaptable to new situations

### Agentic Architecture
The LLM-powered agent system provides:
- Complex task decomposition
- Multi-step reasoning and planning
- Conversational memory and context tracking
- Error recovery and replanning

### Edge AI Deployment
Running inference on Arduino UNO Q provides:
- Privacy-preserving local processing
- Low-latency response times
- Offline operation capability
- Reduced cloud dependency

## Visuals & Media

### Demonstration Video

[![VLA-Powered Bedside Assistant Demo](https://img.youtube.com/vi/BgnGZIz5aYY/maxresdefault.jpg)](https://www.youtube.com/watch?v=BgnGZIz5aYY)

*Comprehensive demonstration of the VLA-powered bedside assistant system performing adaptive manipulation tasks in response to natural language voice commands.*

### Project Gallery

- System overview photo showing SO-101 5 DOF arm
- Web dashboard screenshot with VLA visualization
- Robotic arm executing voice commands
- Before/after comparison: traditional vs. VLA approach
- Live VLA inference with action generation

## GitHub Repository

[https://github.com/astroanax/lenurse-bedside-assistant](https://github.com/astroanax/lenurse-bedside-assistant)

## License

This project is licensed under the Mozilla Public License Version 2.0 - see the [LICENSE.txt](LICENSE.txt) file for details.

## Technical Details

### Technologies Used
- **VLA Model**: pi0.5 (shehiin/pi05_pick_red_cube_lora) for vision-language-action mapping
- **Robotics Framework**: LeRobot - Open-source toolkit for training and deploying robotic policies
- **LLM**: DeepSeek R1T Chimera via OpenRouter API for agentic reasoning
- **Computer Vision**: YOLO11 for hazard detection, OpenCV for image processing
- **Hardware**: Arduino UNO Q with integrated camera, SO-101 5 DOF robotic arm
- **Backend**: Flask web application with WebSocket support
- **Speech**: Browser Web Speech API for voice interaction

### Target Users
- Bedridden patients requiring assistance
- Elderly individuals with limited mobility
- Healthcare facilities seeking automation
- Home care environments

## Team Artificial Penguins

NIT Calicut

## Why VLA Matters: Technical Deep Dive

### The Traditional Approach (Pre-recorded Trajectories)

Most robots require **pre-recorded trajectories**:
1. Manually move robot to record waypoints
2. Save position sequence as "pick_water_bottle.traj"
3. Replay exact same movements every time

**Limitations:**
- Objects must be in exact same position
- Need to record trajectory for every scenario
- Cannot handle variations or new objects
- Time-consuming manual programming

### Our VLA Approach (Key Differentiator)

We use **Vision-Language-Action (VLA)** end-to-end learning:
1. Camera captures the current scene
2. User provides voice command: "pick up the water bottle"
3. VLA model generates motion in real-time
4. Robot adapts to actual object position

**Advantages:**
- Adapts to object position automatically
- Works with novel objects (transfer learning)
- Natural language understanding in visual context
- Single model replaces multiple subsystems

### Technical Implementation

**Model**: pi0.5 from LeRobot (shehiin/pi05_pick_red_cube_lora)  
**Architecture**: Transformer-based policy network (192M parameters)  
**Input**: RGB image (224×224) + text instruction  
**Output**: 6D action vector (5 joints + gripper)  
**Inference**: 10 Hz on Arduino UNO Q CPU  
**Training**: Pre-trained + LoRA fine-tuning for SO-101 arm  

**Why LeRobot?**  
LeRobot is HuggingFace's open-source framework specifically designed for deploying VLA policies on real robots. It handles:
- Model downloading and caching
- Real-time inference loops
- Action smoothing and safety checks
- Hardware abstraction

### See It In Action

Our VLA system enables:
- **Zero-shot Generalization**: Can pick up objects it wasn't explicitly trained on
- **Spatial Reasoning**: Understands "the bottle on the left" vs "the one closer to me"
- **Adaptive Grasping**: Adjusts grip based on object size and shape from vision
- **Context Awareness**: "hand me that" understands object from camera view
