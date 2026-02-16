const fileInput = document.getElementById('fileInput');
const dropArea = document.getElementById('drop-area');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultSection = document.getElementById('resultSection');
const audioPlayer = document.getElementById('audioPlayer');
const recordBtn = document.getElementById('recordBtn');
const recordingStatus = document.getElementById('recordingStatus'); // Fixed ID
const visualizer = document.getElementById('visualizer');

let selectedFile = null;
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// Tab Switching
function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

    document.getElementById(`${tabName}-tab`).classList.add('active');
    document.querySelector(`button[onclick="switchTab('${tabName}')"]`).classList.add('active');

    // Reset inputs
    resetInputs();
}

function resetInputs() {
    selectedFile = null;
    fileInput.value = '';
    document.querySelector('.file-message').textContent = 'or drag and drop .wav/.mp3 here';
    analyzeBtn.disabled = true;
    analyzeBtn.classList.remove('ready');
}

// File Upload Logic
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
    }
});

dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.classList.add('drag-active');
});

dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('drag-active');
});

dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.classList.remove('drag-active');
    if (e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
    }
});

function handleFileSelection(file) {
    selectedFile = file;
    document.querySelector('.file-message').textContent = file.name;
    analyzeBtn.disabled = false;
    analyzeBtn.classList.add('ready');
}

// Recording Logic
recordBtn.addEventListener('click', async () => {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
});

let audioContext;
let processor;
let input;
let globalStream;

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        globalStream = stream;
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        input = audioContext.createMediaStreamSource(stream);

        // Create ScriptProcessor (bufferSize, inputChannels, outputChannels)
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        audioChunks = []; // Store raw float32 buffers

        processor.onaudioprocess = function (e) {
            const channelData = e.inputBuffer.getChannelData(0);
            audioChunks.push(new Float32Array(channelData));
        };

        input.connect(processor);
        processor.connect(audioContext.destination);

        isRecording = true;
        recordBtn.innerHTML = '<span class="mic-icon">⏹</span> Stop Recording';
        recordBtn.classList.add('recording');
        recordingStatus.textContent = "Recording...";
        visualizer.classList.add('active');

    } catch (err) {
        console.error("Error accessing microphone:", err);
        alert("Microphone access denied or not available.");
    }
}

function stopRecording() {
    if (processor && input && audioContext) {
        processor.disconnect();
        input.disconnect();
        if (globalStream) {
            globalStream.getTracks().forEach(track => track.stop());
        }
        audioContext.close();

        isRecording = false;
        recordBtn.innerHTML = '<span class="mic-icon">🎤</span> Start Recording';
        recordBtn.classList.remove('recording');
        recordingStatus.textContent = "Recording finished";
        visualizer.classList.remove('active');

        // Process chunks into WAV
        const wavBlob = exportWAV(audioChunks);
        const file = new File([wavBlob], "recorded_audio.wav", { type: 'audio/wav' });
        handleFileSelection(file);
    }
}

function exportWAV(chunks) {
    // Flatten chunks
    let length = 0;
    chunks.forEach(chunk => length += chunk.length);
    let pcmData = new Float32Array(length);
    let offset = 0;
    chunks.forEach(chunk => {
        pcmData.set(chunk, offset);
        offset += chunk.length;
    });

    // Downsample or keep? Let's just create 16-bit PCM WAV at current sample rate
    const sampleRate = 44100; // Assuming standard, or use audioContext.sampleRate

    const buffer = new ArrayBuffer(44 + pcmData.length * 2);
    const view = new DataView(buffer);

    // Write WAV Header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + pcmData.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, pcmData.length * 2, true);

    // Float to 16-bit PCM
    let index = 44;
    let volume = 1;
    for (let i = 0; i < pcmData.length; i++) {
        let s = Math.max(-1, Math.min(1, pcmData[i]));
        s = s < 0 ? s * 0x8000 : s * 0x7FFF;
        view.setInt16(index, s, true);
        index += 2;
    }

    return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// Analysis Request
analyzeBtn.addEventListener('click', () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    analyzeBtn.textContent = "Analyzing...";
    analyzeBtn.disabled = true;

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert("Error: " + data.error);
                return;
            }
            showResult(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert("An error occurred during analysis.");
        })
        .finally(() => {
            analyzeBtn.textContent = "Analyze Sound";
            analyzeBtn.disabled = false;
        });
});

// Visualizer Logic
const canvas = document.getElementById('frequencyCanvas');
const ctx = canvas.getContext('2d');
let audioContextVerify;
let analyser;
let source;

function setupVisualizer() {
    if (audioContextVerify) return;

    audioContextVerify = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContextVerify.createAnalyser();
    source = audioContextVerify.createMediaElementSource(audioPlayer);

    source.connect(analyser);
    analyser.connect(audioContextVerify.destination);

    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    // Resize canvas
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    function draw() {
        requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.2)'; // Trail effect
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLength) * 2.5;
        let barHeight;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            barHeight = dataArray[i] / 2;

            // Gradient Color based on height/danger
            const g = ctx.createLinearGradient(0, canvas.height, 0, 0);
            g.addColorStop(0, '#00f0ff');
            g.addColorStop(1, '#7000ff');

            ctx.fillStyle = g;
            ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

            x += barWidth + 1;
        }
    }
    draw();
}

audioPlayer.addEventListener('play', () => {
    if (!audioContextVerify) {
        setupVisualizer();
    }
    if (audioContextVerify.state === 'suspended') {
        audioContextVerify.resume();
    }
});

// Update Result Display
function showResult(data) {
    resultSection.classList.remove('hidden');
    resultSection.scrollIntoView({ behavior: 'smooth' });

    // Update Text
    document.getElementById('soundClass').textContent = data.class.replace(/_/g, ' ');
    const dangerBadge = document.getElementById('dangerLevel');
    dangerBadge.textContent = data.danger_level;

    const safetyBadge = document.getElementById('safetyLevel');
    safetyBadge.textContent = data.hearing_safety;

    // Update Badges Class
    dangerBadge.className = 'value badge'; // Reset
    safetyBadge.className = 'value badge';

    if (data.danger_level === 'DANGER') {
        dangerBadge.classList.add('badge-danger');
        safetyBadge.classList.add('badge-danger');
    }
    else if (data.danger_level === 'WARNING') {
        dangerBadge.classList.add('badge-warning');
        safetyBadge.classList.add('badge-warning');
    }
    else {
        dangerBadge.classList.add('badge-safe');
        safetyBadge.classList.add('badge-safe');
    }

    // Update Suggestion
    document.getElementById('suggestionText').textContent = data.suggestion;

    // Update Scene Description
    const sceneDesc = data.scene_description || "Scene analysis not available.";
    document.getElementById('sceneDescription').textContent = sceneDesc;

    // Update Audio Player
    audioPlayer.src = data.file_url;
    audioPlayer.load();
}

function resetApp() {
    resultSection.classList.add('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });
    resetInputs();
    if (audioPlayer) {
        audioPlayer.pause();
        audioPlayer.currentTime = 0;
    }
}
