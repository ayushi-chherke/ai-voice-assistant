from dotenv import load_dotenv
load_dotenv()
import json
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import base64
import os
load_dotenv(dotenv_path=".env")
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=GROQ_API_KEY
)

MODEL_NAME = "llama-3.1-8b-instant"
SYSTEM_PROMPT = """
You are a friendly AI voice assistant.
Reply naturally like a human.
Keep replies short and conversational.
No markdown.
No bullet points.
"""

# ─────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────
app = Flask(__name__)

CORS(app)

# ─────────────────────────────────────────────
# AI CHAT FUNCTION
# ─────────────────────────────────────────────
def generate_reply(user_message, history=[]):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    for item in history:
        role = item.get("role", "user")
        content = item.get("parts", [""])[0]

        # ✅ Convert Gemini's "model" role → OpenAI's "assistant"
        if role == "model":
            role = "assistant"

        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()

# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────
@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "model": MODEL_NAME
    })

# ─────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────
import json
import tempfile
import base64

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # ── Voice (FormData) ──────────────────────────────
        if request.content_type and "multipart/form-data" in request.content_type:
            audio_file = request.files.get("audio")
            history_raw = request.form.get("history", "[]")
            tts = request.form.get("tts", "false").lower() == "true"
            history = json.loads(history_raw)

            if not audio_file:
                return jsonify({"error": "No audio file"}), 400

            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                audio_file.save(tmp.name)
                tmp_path = tmp.name

            with open(tmp_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=("recording.webm", f, "audio/webm"),
                )
            os.unlink(tmp_path)

            transcript = transcription.text.strip()
            if not transcript:
                return jsonify({"error": "Could not transcribe audio"}), 400

            reply = generate_reply(transcript, history)

            audio_b64 = generate_tts(reply) if tts else None

            return jsonify({
                "reply": reply,
                "transcript": transcript,
                "audio_b64": audio_b64
            })

        # ── Text (JSON) ───────────────────────────────────
        else:
            data = request.get_json(force=True)
            message = data.get("message", "").strip()
            history = data.get("history", [])
            tts = data.get("tts", False)

            if not message:
                return jsonify({"error": "Empty message"}), 400

            reply = generate_reply(message, history)

            audio_b64 = generate_tts(reply) if tts else None

            return jsonify({
                "reply": reply,
                "audio_b64": audio_b64
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# def generate_tts(text):
#     """Convert text to speech using Groq Orpheus TTS, return base64 wav."""
#     try:
#         # Orpheus has a 200 char limit — truncate if needed
#         if len(text) > 200:
#             text = text[:197] + "..."

#         response = client.audio.speech.create(
#             model="canopylabs/orpheus-v1-english",
#             voice="hannah",          # options: autumn, diana, hannah, austin, daniel, troy
#             input=text,
#             response_format="wav"    # MUST be wav — mp3 not supported
#         )
#         audio_bytes = response.read()
#         return base64.b64encode(audio_bytes).decode("utf-8")
#     except Exception as e:
#         print(f"TTS error: {e}")
#         return None

def generate_tts(text):
    try:
        if len(text) > 200:
            text = text[:197] + "..."

        print(f"TTS input ({len(text)} chars): {text}")  # debug

        response = client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice="hannah",
            input=text,
            response_format="wav"
        )
        audio_bytes = response.read()
        print(f"TTS success: {len(audio_bytes)} bytes")  # debug
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print(f"TTS error: {e}")  # this will tell us the real error
        return None
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("VOXA Backend Running")
    print("=" * 50)
    print("POST /chat")
    print("GET  /health")
    print("=" * 50)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )