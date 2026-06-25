import os
import traceback
import base64
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


from pdf_loader import load_and_split_pdf
from vector_store import get_vectorstore
from rag_chain import stream_answer
import os
from dotenv import load_dotenv

# 1. Load the environment variables from the .env file
load_dotenv()

# 2. Extract the keys securely using their variable names
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# 3. (Optional) Quick safety check to ensure they loaded properly
if not GOOGLE_API_KEY or not ELEVENLABS_API_KEY:
    print("❌ Error: One or both API keys failed to load. Check your .env file format.")
else:
    print("✅ API Keys successfully loaded from .env file!")

app = FastAPI(title="Voice/Text RAG Chatbot API")
load_dotenv()
# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⚠️ REMINDER: Revoke this API key string inside your ElevenLabs developer platform
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
PDF_PATH = "data/My.pdf"

print("Loading PDF...")
chunks = load_and_split_pdf(PDF_PATH)

print("Loading Vector Store...")
vectorstore = get_vectorstore(chunks)

print("RAG Ready!")


@app.get("/")
def home():
    return {"message": "Voice/Text RAG API Running"}


@app.post("/chat")
async def chat(
    file: UploadFile = File(None), 
    text_input: str = Form(None)
):
    """
    Accepts incoming user data via either text input or audio file, processes it
    through the local RAG pipeline, and returns both the text answer and base64 audio.
    """
    user_question = ""
    temp_audio_path = None

    try:
        # 1. Handle Input: Voice vs Text
        if file:
            temp_audio_path = f"temp_{file.filename}"
            with open(temp_audio_path, "wb") as buffer:
                buffer.write(await file.read())

            print("Sending audio to ElevenLabs STT...")
            stt_url = "https://api.elevenlabs.io/v1/speech-to-text"
            stt_headers = {"xi-api-key": ELEVENLABS_API_KEY}
            
            with open(temp_audio_path, "rb") as audio_file:
                stt_files = {"file": (file.filename, audio_file, "audio/webm")}
                stt_data = {"model_id": "scribe_v2"}
                stt_response = requests.post(stt_url, headers=stt_headers, files=stt_files, data=stt_data)
            
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

            if stt_response.status_code != 200:
                raise HTTPException(status_code=stt_response.status_code, detail=f"STT Error: {stt_response.text}")

            user_question = stt_response.json().get("text", "").strip()
        
        elif text_input:
            user_question = text_input.strip()
        
        else:
            raise HTTPException(status_code=400, detail="No input provided. Send either audio or text.")

        if not user_question:
            raise HTTPException(status_code=400, detail="Empty prompt processed.")

        print(f"Processed Input Question: {user_question}")

        # 2. Process through local RAG Pipeline
        full_response_text = ""
        for token in stream_answer(vectorstore, user_question):
            full_response_text += token

        print(f"RAG Generated Answer: {full_response_text}")

        # 3. Fetch active ElevenLabs voice identification
        voices_url = "https://api.elevenlabs.io/v1/voices"
        voices_headers = {"xi-api-key": ELEVENLABS_API_KEY}
        voices_response = requests.get(voices_url, headers=voices_headers)
        
        voice_id = "21m00Tcm4TlvDq8ikWAM" # Fallback
        
        if voices_response.status_code == 200:
            available_voices = voices_response.json().get("voices", [])
            if available_voices:
                voice_id = available_voices[0].get("voice_id")

        # 4. Convert output Answer Text to Speech (TTS)
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        tts_headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        tts_data = {
            "text": full_response_text,
            "model_id": "eleven_flash_v2_5",
            "output_format": "mp3_44100_128"
        }

        tts_response = requests.post(tts_url, headers=tts_headers, json=tts_data)

        if tts_response.status_code != 200:
            raise HTTPException(status_code=tts_response.status_code, detail=f"TTS Error: {tts_response.text}")

        # Encode the audio binary content to base64 safely passing structured JSON
        audio_base64 = base64.b64encode(tts_response.content).decode("utf-8")

        return {
            "user_question": user_question,
            "bot_response": full_response_text,
            "audio_data": f"data:audio/mpeg;base64,{audio_base64}"
        }

    except Exception as e:
        traceback.print_exc()
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=str(e))