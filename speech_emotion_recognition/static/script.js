// Global variables
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let recordedBlob = null;
let uploadedFile = null;

// DOM elements
const recordBtn = document.getElementById('recordBtn');
const recordText = document.getElementById('recordText');
const recordingIndicator = document.getElementById('recordingIndicator');
const analyzeBtn = document.getElementById('analyzeBtn');
const audioFileInput = document.getElementById('audioFile');
const fileName = document.getElementById('fileName');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultSection = document.getElementById('resultSection');

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    hideElements();
});

function initializeEventListeners() {
    // Record button click
    if (recordBtn) {
        recordBtn.addEventListener('click', toggleRecording);
    }
    
    // File input change
    if (audioFileInput) {
        audioFileInput.addEventListener('change', handleFileUpload);
    }
    
    // Analyze button click
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeEmotion);
    }
}

function hideElements() {
    if (recordingIndicator) recordingIndicator.style.display = 'none';
    if (loadingIndicator) loadingIndicator.style.display = 'none';
    if (resultSection) resultSection.style.display = 'none';
}

async function toggleRecording() {
    if (!isRecording) {
        await startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                sampleRate: 22050,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });
        
        mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });
        
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = () => {
            recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
            uploadedFile = null; // Clear uploaded file
            enableAnalyzeButton();
            
            // Stop all tracks to release microphone
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        isRecording = true;
        
        // Update UI
        recordText.textContent = 'Stop Recording';
        recordBtn.classList.add('recording');
        recordingIndicator.style.display = 'flex';
        
        // Auto-stop after 60 seconds (1 minute)
        setTimeout(() => {
            if (isRecording) {
                stopRecording();
            }
        }, 60000);
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Error accessing microphone. Please check permissions.');
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        
        // Update UI
        recordText.textContent = 'Start Recording';
        recordBtn.classList.remove('recording');
        recordingIndicator.style.display = 'none';
    }
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        uploadedFile = file;
        recordedBlob = null; // Clear recorded audio
        fileName.textContent = file.name;
        enableAnalyzeButton();
    }
}

function enableAnalyzeButton() {
    if (analyzeBtn) {
        analyzeBtn.disabled = false;
        analyzeBtn.classList.add('enabled');
    }
}

async function analyzeEmotion() {
    if (!recordedBlob && !uploadedFile) {
        alert('Please record audio or upload a file first.');
        return;
    }
    
    // Show loading
    loadingIndicator.style.display = 'block';
    resultSection.style.display = 'none';
    analyzeBtn.disabled = true;
    
    try {
        const formData = new FormData();
        
        if (recordedBlob) {
            // Convert webm to wav for better compatibility
            const wavBlob = await convertToWav(recordedBlob);
            formData.append('audio', wavBlob, 'recording.wav');
        } else if (uploadedFile) {
            formData.append('audio', uploadedFile);
        }
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResult(result);
        } else {
            throw new Error(result.error || 'Analysis failed');
        }
        
    } catch (error) {
        console.error('Error analyzing emotion:', error);
        alert('Error analyzing emotion: ' + error.message);
    } finally {
        loadingIndicator.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

async function convertToWav(webmBlob) {
    // For now, return the original blob
    // In a production app, you'd use a library like lamejs or similar
    return webmBlob;
}

function displayResult(result) {
    const emotionEmoji = document.getElementById('emotionEmoji');
    const emotionName = document.getElementById('emotionName');
    const confidenceScore = document.getElementById('confidenceScore');
    const confidenceFill = document.getElementById('confidenceFill');
    
    // Emotion emojis mapping
    const emojiMap = {
        'Happy': '😊',
        'Sad': '😢',
        'Angry': '😡',
        'Neutral': '😐',
        'Fear': '😨',
        'Disgust': '🤢',
        'Surprise': '😲'
    };
    
    // Update main result
    if (emotionEmoji) emotionEmoji.textContent = emojiMap[result.emotion] || '😐';
    if (emotionName) emotionName.textContent = result.emotion;
    if (confidenceScore) confidenceScore.textContent = result.confidence + '%';
    if (confidenceFill) confidenceFill.style.width = result.confidence + '%';
    
    // Update emotion breakdown if available
    if (result.probabilities) {
        updateEmotionBreakdown(result.probabilities);
    }
    
    // Show result section
    resultSection.style.display = 'block';
    
    // Scroll to result
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

function updateEmotionBreakdown(probabilities) {
    const emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'];
    
    emotions.forEach(emotion => {
        const fill = document.querySelector(`[data-emotion="${emotion}"] .fill`);
        const percentage = document.querySelector(`[data-emotion="${emotion}"] .percentage`);
        
        if (fill && percentage && probabilities[emotion] !== undefined) {
            const value = Math.round(probabilities[emotion] * 100);
            fill.style.width = value + '%';
            percentage.textContent = value + '%';
        }
    });
}

// Smooth scrolling function
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth' });
    }
}

// Navigation menu toggle for mobile
document.addEventListener('DOMContentLoaded', function() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', function() {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
        
        // Close menu when clicking on a link
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', function() {
                hamburger.classList.remove('active');
                navMenu.classList.remove('active');
            });
        });
    }
});