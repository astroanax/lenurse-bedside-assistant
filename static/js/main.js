/**
 * Bedside Healthcare Assistant Robot - Frontend
 * Handles voice assistant chat, camera feeds, and robot control
 */

// =============================================================================
// STATE
// =============================================================================

const state = {
    selectedTask: null,
    robotState: 'IDLE',
    cameraConnected: false,
    darkMode: false,
    language: 'en',
    translations: {},
    jarvisConversation: [],
    isRecording: false,
    trajectoryStatus: null
};

// =============================================================================
// DOM ELEMENTS
// =============================================================================

const elements = {
    // Status
    systemStatus: document.getElementById('systemStatus'),
    
    // Settings
    btnDarkMode: document.getElementById('btnDarkMode'),
    selectLanguage: document.getElementById('selectLanguage'),
    
    // JARVIS Chat
    jarvisMessages: document.getElementById('jarvisMessages'),
    jarvisInput: document.getElementById('jarvisInput'),
    btnJarvisVoice: document.getElementById('btnJarvisVoice'),
    btnJarvisSend: document.getElementById('btnJarvisSend'),
    btnClearChat: document.getElementById('btnClearChat'),
    
    // Camera
    cameraInput: document.getElementById('cameraInput'),
    btnCameraConnect: document.getElementById('btnCameraConnect'),
    cameraStatus: document.getElementById('cameraStatus'),
    videoFeed: document.getElementById('videoFeed'),
    feedStatus: document.getElementById('feedStatus'),
    cameraDevice: document.getElementById('cameraDevice'),
    
    // Robot Status
    robotState: document.getElementById('robotState'),
    robotTask: document.getElementById('robotTask'),
    btnRobotReset: document.getElementById('btnRobotReset'),
    btnHome: document.getElementById('btnHome'),
    btnEmergencyStop: document.getElementById('btnEmergencyStop'),
    
    // Trajectories
    trajectoriesList: document.getElementById('trajectoriesList'),
    btnRefreshTrajectories: document.getElementById('btnRefreshTrajectories'),
    btnStopTrajectory: document.getElementById('btnStopTrajectory'),
    btnUploadTrajectory: document.getElementById('btnUploadTrajectory'),
    fileUploadTrajectory: document.getElementById('fileUploadTrajectory'),
    
    // Log
    logContainer: document.getElementById('logContainer'),
    btnClearLog: document.getElementById('btnClearLog'),
    
    // Footer
    footerTime: document.getElementById('footerTime')
};

// =============================================================================
// API CALLS
// =============================================================================

const api = {
    async jarvisChat(message, language = 'en') {
        const response = await fetch('/api/jarvis/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, language })
        });
        return response.json();
    },
    
    async voiceChat(text, language = 'en') {
        const response = await fetch('/api/voice/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, language })
        });
        return response.json();
    },
    
    async selectCamera(deviceId) {
        const response = await fetch('/api/camera/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ device_id: parseInt(deviceId) })
        });
        return response.json();
    },
    
    async getCameraStatus() {
        const response = await fetch('/api/camera/status');
        return response.json();
    },
    
    async getRobotStatus() {
        const response = await fetch('/api/robot/status');
        return response.json();
    },
    
    async robotHome() {
        const response = await fetch('/api/robot/home', { method: 'POST' });
        return response.json();
    },
    
    async robotReset() {
        const response = await fetch('/api/robot/reset', { method: 'POST' });
        return response.json();
    },
    
    async selectTask(task) {
        const response = await fetch('/api/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task })
        });
        return response.json();
    },
    
    async getLogs() {
        const response = await fetch('/api/logs');
        return response.json();
    },
    
    async resetState() {
        const response = await fetch('/api/reset', { method: 'POST' });
        return response.json();
    },
    
    async listTrajectories() {
        const response = await fetch('/api/trajectories/list');
        return response.json();
    },
    
    async replayTrajectory(filename) {
        const response = await fetch('/api/trajectories/replay', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ trajectory_file: filename })
        });
        return response.json();
    },
    
    async stopTrajectory() {
        const response = await fetch('/api/trajectories/stop', { method: 'POST' });
        return response.json();
    },
    
    async callMom() {
        const response = await fetch('/api/family/call/mom', { method: 'POST' });
        return response.json();
    },
    
    async callDad() {
        const response = await fetch('/api/family/call/dad', { method: 'POST' });
        return response.json();
    },
    
    async getHazardStatus() {
        const response = await fetch('/api/hazard/status');
        return response.json();
    },
    
    async resetHazard() {
        const response = await fetch('/api/hazard/reset', { method: 'POST' });
        return response.json();
    },
    
    async getTrajectoryStatus() {
        const response = await fetch('/api/trajectories/status');
        return response.json();
    },
    
    async uploadTrajectory(file) {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch('/api/trajectories/upload', {
            method: 'POST',
            body: formData
        });
        return response.json();
    },
    
    async getSettings() {
        const response = await fetch('/api/settings');
        return response.json();
    },
    
    async updateSettings(settings) {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        return response.json();
    }
};

// =============================================================================
// UI UPDATES
// =============================================================================

function addJarvisMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `jarvis-message ${role}`;
    
    const sender = document.createElement('span');
    sender.className = 'message-sender';
    sender.textContent = role === 'user' ? 'YOU' : 'ASSISTANT';
    
    const contentSpan = document.createElement('span');
    contentSpan.className = 'message-content';
    contentSpan.textContent = content;
    
    messageDiv.appendChild(sender);
    messageDiv.appendChild(contentSpan);
    
    elements.jarvisMessages.appendChild(messageDiv);
    elements.jarvisMessages.scrollTop = elements.jarvisMessages.scrollHeight;
}

function addLog(message, level = 'info') {
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];
    
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${level}`;
    
    const time = document.createElement('span');
    time.className = 'log-time';
    time.textContent = timeStr;
    
    const msg = document.createElement('span');
    msg.className = 'log-message';
    msg.textContent = message;
    
    logEntry.appendChild(time);
    logEntry.appendChild(msg);
    
    elements.logContainer.insertBefore(logEntry, elements.logContainer.firstChild);
    
    // Keep only last 50 logs
    while (elements.logContainer.children.length > 50) {
        elements.logContainer.removeChild(elements.logContainer.lastChild);
    }
}

function updateSystemStatus(status, connected = true) {
    const dot = elements.systemStatus.querySelector('.status-dot');
    const text = elements.systemStatus.querySelector('.status-text');
    
    dot.className = 'status-dot' + (connected ? ' connected' : ' disconnected');
    text.textContent = status;
}

function updateCameraStatus() {
    api.getCameraStatus().then(data => {
        state.cameraConnected = data.connected;
        elements.cameraStatus.textContent = data.connected ? 
            `Device: /dev/video${data.device_id}` : 'Not Connected';
        elements.cameraDevice.textContent = data.connected ?
            `/dev/video${data.device_id}` : 'Not Connected';
    });
}

function updateRobotStatus() {
    api.getRobotStatus().then(data => {
        state.robotState = data.state_display || 'IDLE';
        elements.robotState.textContent = data.state_display || 'IDLE';
        elements.robotTask.textContent = data.current_task || 'None';
        
        // Update status color
        const stateClass = data.is_busy ? 'busy' : 'idle';
        elements.robotState.className = `status-value ${stateClass}`;
    });
}

function updateTrajectories() {
    api.listTrajectories().then(data => {
        if (data.success && data.trajectories.length > 0) {
            elements.trajectoriesList.innerHTML = '';
            data.trajectories.forEach(traj => {
                const trajDiv = document.createElement('div');
                trajDiv.className = 'trajectory-item';
                trajDiv.innerHTML = `
                    <span class="trajectory-name">${traj.filename}</span>
                    <button class="btn btn-mini btn-play" data-file="${traj.filename}">PLAY</button>
                `;
                elements.trajectoriesList.appendChild(trajDiv);
            });
            
            // Add click handlers
            document.querySelectorAll('.btn-play').forEach(btn => {
                btn.addEventListener('click', () => {
                    const filename = btn.getAttribute('data-file');
                    replayTrajectory(filename);
                });
            });
        } else {
            elements.trajectoriesList.innerHTML = '<p class="info-text">No trajectories available</p>';
        }
    }).catch(err => {
        addLog('Failed to load trajectories', 'error');
    });
}

function updateLogs() {
    api.getLogs().then(data => {
        if (data.logs && data.logs.length > 0) {
            elements.logContainer.innerHTML = '';
            data.logs.reverse().forEach(log => {
                addLog(log.message, log.level);
            });
        }
    });
}

// =============================================================================
// JARVIS CHAT
// =============================================================================

async function sendJarvisMessage() {
    const message = elements.jarvisInput.value.trim();
    if (!message) return;
    
    addJarvisMessage('user', message);
    elements.jarvisInput.value = '';
    
    try {
        const result = await api.jarvisChat(message, state.language);
        if (result.success) {
            addJarvisMessage('assistant', result.response);
            if (result.arm_action) {
                addLog(result.arm_action.message, 'success');
            }
        } else {
            addLog('Error: ' + result.error, 'error');
        }
    } catch (err) {
        addLog('Failed to send message', 'error');
    }
    
    updateRobotStatus();
}

// =============================================================================
// VOICE INPUT
// =============================================================================

let voiceRecognition = null;

function startVoiceRecording() {
    // Use browser's speech recognition API directly
    if (!('webkitSpeechRecognition' in window)) {
        addLog('Voice recognition not supported in this browser', 'warning');
        return;
    }
    
    voiceRecognition = new webkitSpeechRecognition();
    voiceRecognition.lang = state.language === 'ml' ? 'ml-IN' : 'en-US';
    voiceRecognition.interimResults = false;
    voiceRecognition.maxAlternatives = 1;
    
    voiceRecognition.onstart = () => {
        state.isRecording = true;
        elements.btnJarvisVoice.classList.add('recording');
        addLog('🎤 Listening...', 'info');
    };
    
    voiceRecognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        addLog(`Heard: ${transcript}`, 'success');
        elements.jarvisInput.value = transcript;
        
        // Automatically send the message
        sendVoiceMessage(transcript);
    };
    
    voiceRecognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        addLog('Voice recognition error: ' + event.error, 'error');
        stopVoiceRecording();
    };
    
    voiceRecognition.onend = () => {
        stopVoiceRecording();
    };
    
    try {
        voiceRecognition.start();
    } catch (error) {
        addLog('Failed to start voice recognition', 'error');
        stopVoiceRecording();
    }
}

function stopVoiceRecording() {
    state.isRecording = false;
    elements.btnJarvisVoice.classList.remove('recording');
    
    if (voiceRecognition) {
        try {
            voiceRecognition.stop();
        } catch (e) {
            // Already stopped
        }
        voiceRecognition = null;
    }
}

async function sendVoiceMessage(transcript) {
    try {
        // Send to voice chat API with TTS response
        const result = await api.voiceChat(transcript, state.language);
        
        if (result.success) {
            addJarvisMessage('user', transcript);
            addJarvisMessage('assistant', result.response);
            
            // Speak the response
            if (result.use_browser_tts || !result.audio) {
                speakResponse(result.response);
            } else if (result.audio) {
                playAudioResponse(result.audio);
            }
            
            // Update robot status
            updateRobotStatus();
        } else {
            addLog('Error: ' + (result.error || 'Unknown error'), 'error');
            speakResponse("Sorry, I encountered an error.");
        }
    } catch (error) {
        addLog('Voice chat error: ' + error.message, 'error');
        speakResponse("Sorry, I'm having trouble right now.");
    }
}

// =============================================================================
// CAMERA CONTROLS
// =============================================================================

async function connectCamera() {
    const deviceId = parseInt(elements.cameraInput.value);
    addLog(`Connecting to camera /dev/video${deviceId}...`, 'info');
    
    try {
        const result = await api.selectCamera(deviceId);
        if (result.success) {
            addLog(`Camera connected: /dev/video${deviceId}`, 'success');
            updateCameraStatus();
            // Reload video feed
            elements.videoFeed.src = '/video_feed?' + new Date().getTime();
        } else {
            addLog('Failed to connect camera: ' + result.error, 'error');
        }
    } catch (err) {
        addLog('Camera connection error', 'error');
    }
}

// =============================================================================
// ROBOT CONTROLS
// =============================================================================

async function robotHome() {
    addLog('Moving robot to home position...', 'info');
    try {
        const result = await api.robotHome();
        if (result.success) {
            addLog('Robot moved to home', 'success');
            updateRobotStatus();
        }
    } catch (err) {
        addLog('Failed to move robot home', 'error');
    }
}

async function robotReset() {
    addLog('Resetting robot...', 'info');
    try {
        const result = await api.robotReset();
        if (result.success) {
            addLog('Robot reset', 'success');
            updateRobotStatus();
        }
    } catch (err) {
        addLog('Failed to reset robot', 'error');
    }
}

async function emergencyStop() {
    addLog('🛑 EMERGENCY STOP ACTIVATED', 'error');
    await robotReset();
    await api.stopTrajectory();
}

// =============================================================================
// TASK SELECTION
// =============================================================================

async function selectTask(task) {
    addLog(`Selected task: ${task}`, 'info');
    state.selectedTask = task;
    
    try {
        const result = await api.selectTask(task);
        if (result.success) {
            // Send to JARVIS to execute
            const message = `Please help me with ${task.replace('_', ' ')}`;
            addJarvisMessage('user', message);
            const jarvisResult = await api.jarvisChat(message, state.language);
            if (jarvisResult.success) {
                addJarvisMessage('assistant', jarvisResult.response);
            }
        }
    } catch (err) {
        addLog('Failed to select task', 'error');
    }
    
    updateRobotStatus();
}

// =============================================================================
// TRAJECTORY CONTROLS
// =============================================================================

async function replayTrajectory(filename) {
    addLog(`Starting trajectory: ${filename}`, 'info');
    elements.btnStopTrajectory.disabled = false;
    
    try {
        const result = await api.replayTrajectory(filename);
        if (result.success) {
            addLog(`Trajectory started: ${filename}`, 'success');
            monitorTrajectory();
        } else {
            addLog('Failed to start trajectory: ' + result.error, 'error');
        }
    } catch (err) {
        addLog('Trajectory replay error', 'error');
    }
}

async function stopTrajectory() {
    addLog('Stopping trajectory...', 'warning');
    try {
        const result = await api.stopTrajectory();
        if (result.success) {
            addLog('Trajectory stopped', 'warning');
            elements.btnStopTrajectory.disabled = true;
        }
    } catch (err) {
        addLog('Failed to stop trajectory', 'error');
    }
}

function monitorTrajectory() {
    const interval = setInterval(async () => {
        try {
            const result = await api.getTrajectoryStatus();
            state.trajectoryStatus = result.status;
            
            if (result.status.state === 'idle') {
                clearInterval(interval);
                elements.btnStopTrajectory.disabled = true;
                addLog('Trajectory completed', 'success');
            }
        } catch (err) {
            clearInterval(interval);
        }
    }, 500);
}

async function uploadTrajectory() {
    const file = elements.fileUploadTrajectory.files[0];
    if (!file) return;
    
    addLog(`Uploading trajectory: ${file.name}`, 'info');
    
    try {
        const result = await api.uploadTrajectory(file);
        if (result.success) {
            addLog(`Trajectory uploaded: ${file.name}`, 'success');
            updateTrajectories();
        } else {
            addLog('Upload failed: ' + result.error, 'error');
        }
    } catch (err) {
        addLog('Upload error', 'error');
    }
    
    elements.fileUploadTrajectory.value = '';
}

// =============================================================================
// SETTINGS
// =============================================================================

async function toggleDarkMode() {
    state.darkMode = !state.darkMode;
    document.body.classList.toggle('dark-mode', state.darkMode);
    elements.btnDarkMode.querySelector('.setting-icon').textContent = state.darkMode ? '☀' : '🌙';
    
    await api.updateSettings({ dark_mode: state.darkMode });
}

async function changeLanguage(language) {
    state.language = language;
    await api.updateSettings({ language });
    addLog(`Language changed to: ${language}`, 'info');
}

// =============================================================================
// WAKE WORD DETECTION - "NURSE"
// =============================================================================

let wakeWordRecognition = null;
let activeConversationRecognition = null;
let isInActiveConversation = false;
const WAKE_WORD = 'nurse';
const SILENCE_TIMEOUT = 5000; // 5 seconds of silence to exit conversation mode
let silenceTimer = null;

function startWakeWordDetection() {
    if (!('webkitSpeechRecognition' in window)) {
        addLog('Wake word detection not supported in this browser', 'warning');
        return;
    }
    
    wakeWordRecognition = new webkitSpeechRecognition();
    wakeWordRecognition.continuous = true;
    wakeWordRecognition.interimResults = false;
    wakeWordRecognition.lang = 'en-US'; // Wake word always in English
    
    wakeWordRecognition.onresult = (event) => {
        const lastResultIndex = event.results.length - 1;
        const transcript = event.results[lastResultIndex][0].transcript.toLowerCase().trim();
        
        console.log('Wake word detection heard:', transcript);
        
        if (transcript.includes(WAKE_WORD)) {
            addLog('🎤 Wake word detected! Starting active conversation...', 'success');
            startActiveConversation();
        }
    };
    
    wakeWordRecognition.onerror = (event) => {
        if (event.error !== 'no-speech') {
            console.error('Wake word detection error:', event.error);
        }
        // Restart on error
        setTimeout(() => {
            if (!isInActiveConversation) {
                try {
                    wakeWordRecognition.start();
                } catch (e) {
                    // Already started
                }
            }
        }, 1000);
    };
    
    wakeWordRecognition.onend = () => {
        // Automatically restart wake word detection if not in active conversation
        if (!isInActiveConversation) {
            setTimeout(() => {
                try {
                    wakeWordRecognition.start();
                } catch (e) {
                    // Already started
                }
            }, 500);
        }
    };
    
    try {
        wakeWordRecognition.start();
        addLog('🎤 Wake word detection active - Say "nurse" to start conversation', 'info');
    } catch (e) {
        console.error('Failed to start wake word detection:', e);
    }
}

function startActiveConversation() {
    // Stop wake word detection
    if (wakeWordRecognition) {
        try {
            wakeWordRecognition.stop();
        } catch (e) {}
    }
    
    isInActiveConversation = true;
    document.body.classList.add('conversation-active');
    addLog('💬 Active conversation mode - Listening...', 'info');
    
    // Start continuous listening
    activeConversationRecognition = new webkitSpeechRecognition();
    activeConversationRecognition.continuous = true;
    activeConversationRecognition.interimResults = true;
    activeConversationRecognition.lang = state.language === 'ml' ? 'ml-IN' : 'en-US';
    
    activeConversationRecognition.onresult = (event) => {
        const lastResultIndex = event.results.length - 1;
        const result = event.results[lastResultIndex];
        const transcript = result[0].transcript;
        
        // Reset silence timer on any speech
        clearTimeout(silenceTimer);
        
        if (result.isFinal) {
            console.log('Final transcript:', transcript);
            
            // Process the user's input
            processConversationInput(transcript);
            
            // Set silence timer - if no speech for 5 seconds, exit conversation mode
            silenceTimer = setTimeout(() => {
                endActiveConversation();
            }, SILENCE_TIMEOUT);
        } else {
            // Show interim results
            console.log('Interim:', transcript);
        }
    };
    
    activeConversationRecognition.onerror = (event) => {
        if (event.error !== 'no-speech') {
            console.error('Active conversation error:', event.error);
        }
        // On error, restart or end conversation
        setTimeout(() => {
            if (isInActiveConversation) {
                endActiveConversation();
            }
        }, 1000);
    };
    
    activeConversationRecognition.onend = () => {
        // Restart if still in active conversation
        if (isInActiveConversation) {
            setTimeout(() => {
                try {
                    activeConversationRecognition.start();
                } catch (e) {
                    console.error('Failed to restart active conversation:', e);
                    endActiveConversation();
                }
            }, 500);
        }
    };
    
    try {
        activeConversationRecognition.start();
    } catch (e) {
        console.error('Failed to start active conversation:', e);
        endActiveConversation();
    }
}

async function processConversationInput(transcript) {
    addLog(`You: ${transcript}`, 'info');
    
    try {
        // Send to voice chat API (handles translation + TTS)
        const result = await api.voiceChat(transcript, state.language);
        
        if (result.success) {
            addJarvisMessage('user', transcript);
            addJarvisMessage('assistant', result.response);
            
            // Play audio response - prefer browser TTS for reliability
            if (result.use_browser_tts || !result.audio) {
                // Use browser TTS (more reliable)
                addLog('Using browser text-to-speech', 'info');
                speakResponse(result.response);
            } else if (result.audio) {
                // Use server-generated audio if available
                playAudioResponse(result.audio);
            }
            
            addLog(`Nurse: ${result.response.substring(0, 50)}...`, 'success');
            
            // Update robot status if needed
            updateRobotStatus();
        } else {
            addLog('Voice processing error: ' + (result.error || 'Unknown error'), 'error');
            speakResponse("Sorry, I encountered an error. Please check the system logs.");
        }
    } catch (error) {
        console.error('Voice processing error:', error);
        addLog('Error processing voice input: ' + error.message, 'error');
        speakResponse("Sorry, I'm having trouble connecting. Please check if the API key is configured.");
    }
}

function endActiveConversation() {
    isInActiveConversation = false;
    document.body.classList.remove('conversation-active');
    
    // Stop active conversation recognition
    if (activeConversationRecognition) {
        try {
            activeConversationRecognition.stop();
        } catch (e) {}
        activeConversationRecognition = null;
    }
    
    clearTimeout(silenceTimer);
    
    addLog('💬 Conversation ended - Listening for wake word...', 'info');
    
    // Restart wake word detection
    setTimeout(() => {
        startWakeWordDetection();
    }, 1000);
}

function playAudioResponse(base64Audio) {
    try {
        const audio = new Audio('data:audio/mp3;base64,' + base64Audio);
        audio.play();
    } catch (error) {
        console.error('Audio playback error:', error);
    }
}

function speakResponse(text) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = state.language === 'ml' ? 'ml-IN' : 'en-US';
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    }
}

// =============================================================================
// FAMILY CONTACTS
// =============================================================================

async function callMom() {
    addLog('📞 Calling Mom...', 'info');
    try {
        const result = await api.callMom();
        if (result.success) {
            addLog(`✅ ${result.message}`, 'success');
        } else {
            addLog(`❌ ${result.message}`, 'error');
        }
    } catch (error) {
        addLog('Failed to call mom', 'error');
    }
}

async function callDad() {
    addLog('📞 Calling Dad...', 'info');
    try {
        const result = await api.callDad();
        if (result.success) {
            addLog(`✅ ${result.message}`, 'success');
        } else {
            addLog(`❌ ${result.message}`, 'error');
        }
    } catch (error) {
        addLog('Failed to call dad', 'error');
    }
}

// =============================================================================
// HAZARD DETECTION & ALERTS
// =============================================================================

function showHazardAlert(hazardType, description) {
    const alertDiv = document.getElementById('hazardAlert');
    const typeElement = document.getElementById('hazardType');
    const descElement = document.getElementById('hazardDescription');
    
    typeElement.textContent = hazardType.toUpperCase();
    descElement.textContent = description;
    
    alertDiv.style.display = 'block';
    
    // Play siren sound if available
    playSiren();
    
    // Auto-dismiss after 30 seconds
    setTimeout(() => {
        dismissHazardAlert();
    }, 30000);
}

function dismissHazardAlert() {
    const alertDiv = document.getElementById('hazardAlert');
    alertDiv.style.display = 'none';
    api.resetHazard();
    stopSiren();
}

function playSiren() {
    // Try to play siren sound file
    const siren = new Audio('/static/sounds/emergency_siren.mp3');
    siren.loop = true;
    siren.volume = 0.7;
    siren.play().catch(() => {
        // Fallback: Use browser beep
        console.warn('Siren sound file not found');
    });
    window.sirenAudio = siren;
}

function stopSiren() {
    if (window.sirenAudio) {
        window.sirenAudio.pause();
        window.sirenAudio.currentTime = 0;
        window.sirenAudio = null;
    }
}

async function checkHazardStatus() {
    try {
        const result = await api.getHazardStatus();
        if (result.success && result.hazard_detection) {
            const hazard = result.hazard_detection;
            if (hazard.hazard_detected && hazard.hazard_type) {
                // Show alert if not already shown
                const alertDiv = document.getElementById('hazardAlert');
                if (alertDiv.style.display === 'none') {
                    showHazardAlert(hazard.hazard_type, 'Emergency detected - please check patient');
                }
            }
        }
    } catch (error) {
        console.error('Failed to check hazard status:', error);
    }
}

// =============================================================================
// EVENT LISTENERS
// =============================================================================

function setupEventListeners() {
    // JARVIS
    elements.btnJarvisSend.addEventListener('click', sendJarvisMessage);
    elements.jarvisInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendJarvisMessage();
    });
    elements.btnClearChat.addEventListener('click', () => {
        elements.jarvisMessages.innerHTML = '';
        addJarvisMessage('assistant', 'Chat cleared. How can I help you?');
    });
    
    // Voice - Click to talk (not hold)
    elements.btnJarvisVoice.addEventListener('click', () => {
        if (state.isRecording) {
            stopVoiceRecording();
        } else {
            startVoiceRecording();
        }
    });
    
    // Camera
    elements.btnCameraConnect.addEventListener('click', connectCamera);
    
    // Robot
    elements.btnRobotReset.addEventListener('click', robotReset);
    elements.btnHome.addEventListener('click', robotHome);
    elements.btnEmergencyStop.addEventListener('click', emergencyStop);
    
    // Tasks
    document.querySelectorAll('.task-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const task = btn.getAttribute('data-task');
            selectTask(task);
        });
    });
    
    // Family Contacts
    const btnCallMom = document.getElementById('btnCallMom');
    const btnCallDad = document.getElementById('btnCallDad');
    if (btnCallMom) btnCallMom.addEventListener('click', callMom);
    if (btnCallDad) btnCallDad.addEventListener('click', callDad);
    
    // Hazard Alert
    const btnDismissHazard = document.getElementById('btnDismissHazard');
    if (btnDismissHazard) btnDismissHazard.addEventListener('click', dismissHazardAlert);
    
    // Trajectories
    elements.btnRefreshTrajectories.addEventListener('click', updateTrajectories);
    elements.btnStopTrajectory.addEventListener('click', stopTrajectory);
    elements.btnUploadTrajectory.addEventListener('click', () => {
        elements.fileUploadTrajectory.click();
    });
    elements.fileUploadTrajectory.addEventListener('change', uploadTrajectory);
    
    // Settings
    elements.btnDarkMode.addEventListener('click', toggleDarkMode);
    elements.selectLanguage.addEventListener('change', (e) => {
        changeLanguage(e.target.value);
    });
    
    // Log
    elements.btnClearLog.addEventListener('click', () => {
        elements.logContainer.innerHTML = '';
        addLog('Log cleared', 'info');
    });
}

// =============================================================================
// INITIALIZATION
// =============================================================================

async function initialize() {
    addLog('Initializing system...', 'info');
    
    // Setup event listeners
    setupEventListeners();
    
    // Load initial data
    updateCameraStatus();
    updateRobotStatus();
    updateTrajectories();
    updateLogs();
    
    // Load settings
    const settings = await api.getSettings();
    if (settings) {
        state.language = settings.language || 'en';
        state.darkMode = settings.dark_mode || false;
        elements.selectLanguage.value = state.language;
        if (state.darkMode) {
            document.body.classList.add('dark-mode');
        }
    }
    
    // Start status polling
    setInterval(updateRobotStatus, 2000);
    setInterval(updateCameraStatus, 5000);
    setInterval(updateLogs, 3000);
    setInterval(checkHazardStatus, 2000); // Check for hazards every 2 seconds
    
    // Update footer time
    setInterval(() => {
        const now = new Date();
        elements.footerTime.textContent = now.toTimeString().split(' ')[0];
    }, 1000);
    
    // Start wake word detection
    setTimeout(() => {
        startWakeWordDetection();
    }, 2000); // Give 2 seconds for system to initialize
    
    updateSystemStatus('READY', true);
    addLog('System ready!', 'success');
    addLog('💬 Say "nurse" to start voice conversation', 'info');
}

// Start when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
} else {
    initialize();
}
