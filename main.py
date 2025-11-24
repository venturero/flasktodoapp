import os
from dotenv import load_dotenv
from flask import Flask, render_template, redirect, url_for, request, jsonify, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
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

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///todo.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Initialize API clients
aai.settings.api_key = ASSEMBLYAI_API_KEY
openai_client = OpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)

# Todo Model
class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80))
    complete = db.Column(db.Boolean)

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

# Todo routes
@app.route("/")
def index():
    todos = Todo.query.all()
    return render_template("main.html", todos=todos)

@app.route("/complete/<string:id>")
def completeTodo(id):
    todo = Todo.query.filter_by(id=id).first()
    todo.complete = not todo.complete
    db.session.commit()
    return redirect(url_for("index"))

@app.route("/add", methods=["POST"])
def addTodo():
    title = request.form.get("title")
    newTodo = Todo(title=title, complete=False)
    db.session.add(newTodo)
    db.session.commit()
    return redirect(url_for("index"))

@app.route("/delete/<string:id>")
def deleteTodo(id):
    todo = Todo.query.filter_by(id=id).first()
    db.session.delete(todo)
    db.session.commit()
    return redirect(url_for("index"))

# Voice bot routes
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
    with app.app_context():
        db.create_all()
    main()

