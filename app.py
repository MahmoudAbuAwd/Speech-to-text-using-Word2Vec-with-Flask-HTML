import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Initialize Flask App
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio(filepath):
    """Transcribe the uploaded audio file using Wav2Vec2."""
    audio, sampling_rate = librosa.load(filepath, sr=16000)
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

@app.route("/", methods=["GET"])
def index():
    """Render the homepage."""
    return render_template("index.html")

@app.route("/", methods=["POST"])
def upload_audio():
    """Handle audio file upload and transcription."""
    if "audioFile" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["audioFile"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            # Transcribe the audio file
            transcription = transcribe_audio(filepath)
        except Exception as e:
            return jsonify({"error": f"Failed to process audio file: {str(e)}"}), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

        # Return the transcription as JSON
        return jsonify({"transcription": transcription})

    return jsonify({"error": "Invalid file type. Only .wav and .mp3 files are supported."}), 400

if __name__ == "__main__":
    app.run(debug=True)
