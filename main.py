import os
import json
import re
from typing import List, Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

app = FastAPI(title="PTE Micro-Exam Engine - Groq Free Tier")

# Permitir solicitudes CORS desde archivos locales u otros orígenes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar cliente de Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


class WordResult(BaseModel):
    word: str
    status: Literal["correct", "incorrect", "missed"]

class PTEEvaluationResponse(BaseModel):
    transcription: str
    fluency_score: int
    accuracy_score: int
    status: Literal["PASS", "RETRY"]
    words_result: List[WordResult]


@app.post("/evaluate-audio", response_model=PTEEvaluationResponse)
async def evaluate_audio(
    audio_file: UploadFile = File(...),
    reference_text: str = Form(...)
):
    try:
        # 1. Leer audio y transcribir con Whisper en Groq
        audio_content = await audio_file.read()
        
        # Nombre de archivo asegurado con extensión compatible
        filename = audio_file.filename if audio_file.filename else "audio.m4a"
        if not (filename.endswith(".m4a") or filename.endswith(".wav") or filename.endswith(".webm") or filename.endswith(".mp3")):
            filename = "recording.m4a"

        transcription_response = client.audio.transcriptions.create(
            file=(filename, audio_content),
            model="whisper-large-v3-turbo",
            response_format="json",
            language="en"
        )
        transcribed_text = transcription_response.text.strip()

        # Si no se detectó voz
        if not transcribed_text:
            return PTEEvaluationResponse(
                transcription="",
                fluency_score=0,
                accuracy_score=0,
                status="RETRY",
                words_result=[
                    WordResult(word=w, status="missed") 
                    for w in reference_text.split()
                ]
            )

        # 2. Evaluación con Llama 3.3 en Groq
        prompt = f"""
You are a PTE Academic exam pronunciation evaluator.
Compare the transcribed text with the reference text.

Reference text: "{reference_text}"
User transcription: "{transcribed_text}"

Respond ONLY with a valid raw JSON object (no markdown formatting, no json  blocks).
Use this exact JSON structure:
{{
  "transcription": "{transcribed_text}",
  "fluency_score": 85,
  "accuracy_score": 90,
  "status": "PASS",
  "words_result": [
    {{"word": "word1", "status": "correct"}},
    {{"word": "word2", "status": "incorrect"}}
  ]
}}

Rules:
- fluency_score (0-100) and accuracy_score (0-100).
- status: "PASS" if both scores >= 65, else "RETRY".
- Map EVERY word from the original reference text with "correct", "incorrect", or "missed".
"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        raw_content = completion.choices[0].message.content.strip()
        
        # Limpiar bloques Markdown si el modelo los incluyó
        cleaned_json = re.sub(r"^(?:json)?\s*|\s*$", "", raw_content, flags=re.MULTILINE).strip()
        
        data = json.loads(cleaned_json)
        return PTEEvaluationResponse(**data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la evaluación: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok", "provider": "Groq Free Tier"}
