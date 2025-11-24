import os
from dotenv import load_dotenv
from flask import Flask, render_template_string, request, jsonify, Response
from flask_cors import CORS
import assemblyai as aai
from elevenlabs import ElevenLabs, VoiceSettings
from openai import OpenAI

# Load environment variables
load_dotenv()

ASSEMBLYAI_API_KEY = os.environ.get('ASSEMBLYAI_API_KEY')
ELEVEN_LABS_API_KEY = os.environ.get('ELEVEN_LABS_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize API clients
aai.settings.api_key = ASSEMBLYAI_API_KEY
openai_client = OpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)

# Load Q&A pairs from file
def load_qa_pairs():
    """Load Q&A pairs from q_&_a.txt"""
    qa_pairs = []
    try:
        qa_path = os.path.join("q_&_a.txt")
        with open(qa_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Parse Q&A pairs
            lines = content.strip().split('\n')
            current_q = None
            current_a = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Q:'):
                    if current_q and current_a:
                        qa_pairs.append({'question': current_q, 'answer': current_a})
                    current_q = line[2:].strip()
                    current_a = None
                elif line.startswith('A:'):
                    current_a = line[2:].strip()
            
            # Add last pair
            if current_q and current_a:
                qa_pairs.append({'question': current_q, 'answer': current_a})
        
        return qa_pairs
    except Exception as e:
        print(f"Error loading Q&A file: {e}")
        return []

qa_pairs = load_qa_pairs()

def check_relevancy(question):
    """Use OpenAI API to check if question is related to to-do list app"""
    try:
        prompt = f"""You are a classifier that determines if a user's question is related to a to-do list application.

A question is related to the to-do list app if it asks about:
- How to use the app features (adding tasks, completing tasks, deleting tasks, etc.)
- App functionality, buttons, inputs, or interface elements
- Task management, task status, or task operations
- App behavior, settings, or features

A question is NOT related if it asks about:
- General knowledge, facts, or information outside the app
- Other applications or software
- Personal advice, opinions, or unrelated topics
- Anything not directly about using the to-do list application

User's question: "{question}"

Respond with only "YES" if the question is related to the to-do list app, or "NO" if it is not related. Do not provide any other text."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful classifier that responds with only YES or NO."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        answer = response.choices[0].message.content.strip().upper()
        return answer == "YES"
    except Exception as e:
        print(f"Error in relevancy check: {e}")
        return True  # Default to allowing if check fails

def find_answer(question):
    """Find answer from Q&A pairs based on question"""
    question_lower = question.lower()
    
    # Simple keyword matching - look for similar questions
    for qa in qa_pairs:
        qa_question_lower = qa['question'].lower()
        # Check if key words match
        if any(word in qa_question_lower for word in question_lower.split() if len(word) > 3):
            return qa['answer']
    
    # If no direct match, use OpenAI to find best answer
    try:
        qa_context = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in qa_pairs])
        prompt = f"""Based on the following Q&A pairs about a to-do list application, answer the user's question.

Q&A Pairs:
{qa_context}

User's question: {question}

Provide a concise answer based on the Q&A pairs above. If the question is not directly covered, provide the closest relevant answer."""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about a to-do list application based on provided Q&A pairs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error finding answer: {e}")
        return "I apologize, but I couldn't find an answer to your question."

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Voice Chatbot - To-Do List Q&A</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .greeting {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }
        .transcription {
            background-color: #fff3e0;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            min-height: 50px;
        }
        .response {
            background-color: #f1f8e9;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            min-height: 50px;
        }
        .status {
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            border-radius: 5px;
            background-color: #e8f5e9;
        }
        .status.listening {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        .status.recording {
            background-color: #ffebee;
            color: #c62828;
        }
        .status.processing {
            background-color: #fff3e0;
            color: #ef6c00;
        }
        .recording {
            color: #f44336;
            font-weight: bold;
        }
        audio {
            width: 100%;
            margin-top: 10px;
        }
        .label {
            font-weight: bold;
            color: #666;
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Voice Chatbot - To-Do List Q&A</h1>
        
        <div class="greeting" id="greeting">
            <p>Hello! I'm your Q&A assistant for the to-do list app. Ask me anything about how to use the app, and I'll help you!</p>
        </div>
        
        <div class="status listening" id="status">
            <div id="statusText">🎤 Listening... Speak your question when ready.</div>
        </div>
        
        <div class="transcription">
            <div class="label">Your Question:</div>
            <div id="transcription">Waiting for your question...</div>
        </div>
        
        <div class="response">
            <div class="label">Bot Response:</div>
            <div id="response">Waiting for your question...</div>
            <audio id="responseAudio" controls style="display: none;"></audio>
        </div>
    </div>

    <script>
        let audioContext;
        let analyser;
        let microphone;
        let dataArray;
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let isProcessing = false;
        let greetingPlayed = false;
        let recordingStream = null;
        
        // Voice Activity Detection (VAD) parameters
        const SPEECH_THRESHOLD = 20; // Lower threshold for earlier detection (0-255)
        const SILENCE_DURATION = 1500; // 1.5 seconds of silence before processing
        const DETECTION_INTERVAL = 50; // Check every 50ms for fast response (was 500ms)
        const LOG_INTERVAL = 500; // Log to terminal every 0.5 seconds (not every detection)
        
        let speechDetected = false;
        let silenceStartTime = null;
        let detectionInterval = null;
        let lastLogTime = 0;

        // Play greeting audio on page load and start continuous listening
        window.onload = async function() {
            if (!greetingPlayed) {
                await playGreeting();
                greetingPlayed = true;
            }
            // Start continuous listening after greeting
            setTimeout(() => {
                startContinuousListening();
            }, 2000); // Wait 2 seconds after greeting
        };

        async function playGreeting() {
            try {
                const response = await fetch('/generate_audio', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        text: "Hello! I'm your Q&A assistant for the to-do list app. Ask me anything about how to use the app, and I'll help you!"
                    })
                });
                
                if (response.ok) {
                    const audioBlob = await response.blob();
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audio = new Audio(audioUrl);
                    await audio.play();
                }
            } catch (error) {
                console.error('Error playing greeting:', error);
            }
        }

        async function startContinuousListening() {
            try {
                // Get microphone access
                recordingStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });
                
                // Set up Web Audio API for voice activity detection
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 512; // Increased for better frequency resolution
                analyser.smoothingTimeConstant = 0.3; // Lower smoothing for faster response (was 0.8)
                
                microphone = audioContext.createMediaStreamSource(recordingStream);
                microphone.connect(analyser);
                
                dataArray = new Uint8Array(analyser.frequencyBinCount);
                
                updateStatus('listening', '🎤 Listening... Speak your question when ready.');
                
                // Log to terminal
                logDetection('listening', 0);
                
                // Start voice detection loop - runs every 0.5 seconds
                startVoiceDetection();
                
            } catch (error) {
                console.error('Error accessing microphone:', error);
                updateStatus('error', '❌ Error accessing microphone. Please check permissions.');
                alert('Error accessing microphone. Please check permissions and reload the page.');
            }
        }

        function startVoiceDetection() {
            // Clear any existing interval
            if (detectionInterval) {
                clearInterval(detectionInterval);
            }
            
            // Run detection every 50ms for fast response (was 500ms)
            detectionInterval = setInterval(() => {
                if (!isProcessing) {
                    detectVoice();
                }
            }, DETECTION_INTERVAL);
        }

        function detectVoice() {
            if (!analyser || isProcessing) {
                return;
            }
            
            // Get audio data
            analyser.getByteFrequencyData(dataArray);
            
            // Calculate average volume
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                sum += dataArray[i];
            }
            const averageVolume = sum / dataArray.length;
            
            // Log detection activity to terminal every 0.5 seconds (not every 50ms)
            const now = Date.now();
            if (now - lastLogTime >= LOG_INTERVAL) {
                logDetection('checking', averageVolume);
                lastLogTime = now;
            }
            
            // Detect speech start - IMMEDIATE response when threshold is crossed
            if (!speechDetected && averageVolume > SPEECH_THRESHOLD) {
                speechDetected = true;
                silenceStartTime = null;
                startRecording(); // Start recording immediately
                updateStatus('recording', '🔴 Recording... I can hear you speaking.');
                logDetection('speech_detected', averageVolume);
            }
            // Detect silence during speech
            else if (speechDetected && averageVolume < SPEECH_THRESHOLD) {
                if (silenceStartTime === null) {
                    silenceStartTime = Date.now();
                    logDetection('silence_detected', averageVolume);
                } else {
                    const silenceDuration = Date.now() - silenceStartTime;
                    // If silence detected for 1.5 seconds, process the audio
                    if (silenceDuration > SILENCE_DURATION) {
                        speechDetected = false;
                        stopRecordingAndProcess();
                    }
                }
            }
            // Reset silence timer if speech resumes
            else if (speechDetected && averageVolume >= SPEECH_THRESHOLD) {
                silenceStartTime = null;
            }
        }

        function startRecording() {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                return; // Already recording
            }
            
            audioChunks = [];
            
            // Create MediaRecorder for the actual audio capture
            mediaRecorder = new MediaRecorder(recordingStream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };
            
            // Start recording with smaller timeslice (50ms) to capture audio faster
            mediaRecorder.start(50); // Collect data every 50ms for lower latency
            isRecording = true;
        }

        async function stopRecordingAndProcess() {
            if (!mediaRecorder || mediaRecorder.state === 'inactive') {
                return;
            }
            
            isProcessing = true;
            updateStatus('processing', '⏳ Processing your question...');
            logDetection('processing', 0);
            
            mediaRecorder.stop();
            
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm;codecs=opus' });
                await processAudio(audioBlob);
                
                // Reset for next recording
                audioChunks = [];
                speechDetected = false;
                silenceStartTime = null;
                isProcessing = false;
                
                // Resume listening after processing
                setTimeout(() => {
                    updateStatus('listening', '🎤 Listening... Speak your question when ready.');
                    logDetection('listening', 0);
                }, 500);
            };
        }

        function updateStatus(type, message) {
            const statusDiv = document.getElementById('status');
            const statusText = document.getElementById('statusText');
            statusDiv.className = 'status ' + type;
            statusText.textContent = message;
        }

        async function logDetection(status, volume) {
            try {
                await fetch('/log_detection', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ status: status, volume: volume })
                });
            } catch (error) {
                // Silently fail - logging is not critical
            }
        }

        async function processAudio(audioBlob) {
            try {
                const formData = new FormData();
                formData.append('audio', audioBlob, 'recording.wav');

                // Transcribe audio
                const transcribeResponse = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });

                if (!transcribeResponse.ok) {
                    throw new Error('Transcription failed');
                }

                const transcribeData = await transcribeResponse.json();
                const question = transcribeData.transcription;
                
                if (!question || question.trim().length === 0) {
                    updateStatus('listening', '🎤 Listening... (No speech detected, try again)');
                    return;
                }
                
                // Display transcription
                document.getElementById('transcription').textContent = question;
                updateStatus('processing', '⏳ Checking relevancy and finding answer...');

                // Get response
                const responseResponse = await fetch('/get_response', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ question: question })
                });

                if (!responseResponse.ok) {
                    throw new Error('Response generation failed');
                }

                const responseData = await responseResponse.json();
                const botResponse = responseData.response;
                
                // Display response
                document.getElementById('response').textContent = botResponse;

                // Generate and play audio
                updateStatus('processing', '⏳ Generating audio response...');
                const audioResponse = await fetch('/generate_audio', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ text: botResponse })
                });

                if (audioResponse.ok) {
                    const audioBlob = await audioResponse.blob();
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audioElement = document.getElementById('responseAudio');
                    audioElement.src = audioUrl;
                    audioElement.style.display = 'block';
                    
                    // Wait for audio to finish before resuming listening
                    audioElement.onended = () => {
                        updateStatus('listening', '🎤 Listening... Speak your question when ready.');
                        logDetection('listening', 0);
                    };
                    
                    await audioElement.play();
                }
            } catch (error) {
                console.error('Error processing audio:', error);
                document.getElementById('transcription').textContent = 'Error processing your question. Please try again.';
                document.getElementById('response').textContent = 'An error occurred. Please try again.';
                updateStatus('listening', '🎤 Listening... (Error occurred, try again)');
                isProcessing = false;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe audio using AssemblyAI"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Get file extension to determine format
        filename = audio_file.filename
        file_ext = filename.split('.')[-1].lower() if '.' in filename else 'webm'
        
        # Save temporarily with appropriate extension
        temp_path = f'temp_audio.{file_ext}'
        audio_file.save(temp_path)
        
        print(f"[Transcription] Processing audio file: {temp_path}")
        
        # Transcribe using AssemblyAI
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if transcript.status == aai.TranscriptStatus.error:
            return jsonify({'error': 'Transcription failed'}), 500
        
        return jsonify({'transcription': transcript.text})
    except Exception as e:
        print(f"Transcription error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_response', methods=['POST'])
def get_response():
    """Get bot response based on question"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Check relevancy
        if not check_relevancy(question):
            response = "I'm a Q&A assistant designed specifically to help with questions about the to-do list application. I can only answer questions related to how to use the app, such as adding tasks, completing tasks, deleting tasks, and other app features. Please ask me something about the to-do list app!"
        else:
            # Find answer from Q&A pairs
            response = find_answer(question)
        
        return jsonify({'response': response})
    except Exception as e:
        print(f"Response generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/log_detection', methods=['POST'])
def log_detection():
    """Log voice detection activity to terminal"""
    try:
        data = request.json
        status = data.get('status', '')
        volume = data.get('volume', 0)
        
        if status == 'checking':
            print(f"[Voice Detection] Checking for voice... Volume level: {volume:.2f}")
        elif status == 'speech_detected':
            print(f"[Voice Detection] ✅ Speech detected! Volume: {volume:.2f} - Starting recording...")
        elif status == 'silence_detected':
            print(f"[Voice Detection] ⏸️  Silence detected (Volume: {volume:.2f}) - Counting silence duration...")
        elif status == 'processing':
            print(f"[Voice Detection] 🔄 Processing audio after 1.5s silence...")
        elif status == 'listening':
            print(f"[Voice Detection] 🎤 Listening mode active - Ready to detect speech...")
        
        return jsonify({'status': 'logged'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    """Generate audio using ElevenLabs and return as stream"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return Response(b'', mimetype='audio/mpeg'), 400
        
        # Generate audio using ElevenLabs
        audio_generator = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id="pNInz6obpgDQGcFmaJgB",
            output_format="mp3_22050_32",
            model_id="eleven_turbo_v2_5",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True,
                speed=1.0,
            ),
        )
        
        # Collect audio bytes
        audio_bytes = b""
        for chunk in audio_generator:
            audio_bytes += chunk
        
        return Response(audio_bytes, mimetype='audio/mpeg')
    except Exception as e:
        print(f"Audio generation error: {e}")
        return Response(b'', mimetype='audio/mpeg'), 500

if __name__ == '__main__':
    print("Starting Voice Chatbot...")
    print(f"Loaded {len(qa_pairs)} Q&A pairs")
    print("Web interface available at: http://127.0.0.1:5000/")
    app.run(host='127.0.0.1', port=5000, debug=True)

