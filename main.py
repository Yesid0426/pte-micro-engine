import os
from typing import List, Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

app = FastAPI(title="PTE Micro-Exam Engine - Groq Free Tier")

# Permitir solicitudes CORS desde archivos locales
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cliente de Groq (lee la variable GROQ_API_KEY)
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
        # 1. Transcripción del audio con Whisper en Groq (Gratis)
        audio_content = await audio_file.read()
        transcription_response = client.audio.transcriptions.create(
            file=(audio_file.filename, audio_content),
            model="whisper-large-v3-turbo",
            response_format="json",
            language="en"
        )
        transcribed_text = transcription_response.text.strip()

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

        # 2. Evaluación del texto usando Llama-3 en Groq (Gratis)
        prompt = f"""
Eres un evaluador de pronunciación para el examen PTE Academic.
Compara la transcripción con la frase original.

Texto de referencia: "{reference_text}"
Transcripción del usuario: "{transcribed_text}"

Responde UNICAMENTE en formato JSON estricto con esta estructura exacta:
{{
  "transcription": "{transcribed_text}",
  "fluency_score": 85,
  "accuracy_score": 90,
  "status": "PASS",
  "words_result": [
    {{"word": "palabra1", "status": "correct"}},
    {{"word": "palabra2", "status": "incorrect"}}
  ]
}}
Reglas:
- fluency_score (0-100) y accuracy_score (0-100).
- status: "PASS" si ambos son >= 65, sino "RETRY".
- Mapea CADA palabra de la frase de referencia original indicando "correct", "incorrect" o "missed".
"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        response_content = completion.choices[0].message.content
        import json
        data = json.loads(response_content)
        return PTEEvaluationResponse(**data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la evaluación: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok", "provider": "Groq Free Tier"}
