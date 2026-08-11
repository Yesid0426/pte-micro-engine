import os
import json
import re
from typing import List, Literal, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

app = FastAPI(title="PTE Academic AI Tutor - Diana's Preparation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


class WordResult(BaseModel):
    word: str
    status: Literal["correct", "incorrect", "missed"]

class PTEEvaluationResponse(BaseModel):
    transcription: str
    fluency_score: int
    pronunciation_score: int
    overall_pte_score: int
    status: Literal["PASS", "RETRY"]
    words_result: List[WordResult]
    error_analysis: str
    correct_form: str
    actionable_tips: str


@app.post("/evaluate-audio", response_model=PTEEvaluationResponse)
async def evaluate_audio(
    audio_file: UploadFile = File(...),
    reference_text: str = Form(...),
    question_type: str = Form("Read Aloud")
):
    try:
        audio_content = await audio_file.read()
        
        filename = audio_file.filename if audio_file.filename else "audio.m4a"
        if not any(filename.endswith(ext) for ext in [".m4a", ".wav", ".webm", ".mp3"]):
            filename = "recording.m4a"

        # 1. Transcripción con Whisper
        transcription_response = client.audio.transcriptions.create(
            file=(filename, audio_content),
            model="whisper-large-v3-turbo",
            response_format="json",
            language="en"
        )
        transcribed_text = transcription_response.text.strip()

        if not transcribed_text:
            return PTEEvaluationResponse(
                transcription="",
                fluency_score=0,
                pronunciation_score=0,
                overall_pte_score=0,
                status="RETRY",
                words_result=[WordResult(word=w, status="missed") for w in reference_text.split()],
                error_analysis="No detectamos sonido claro en tu grabación, Diana.",
                correct_form=reference_text,
                actionable_tips="Asegúrate de hablar cerca del micrófono en un entorno silencioso y sin pausar al inicio."
            )

        # 2. Evaluación pedagógica especializada en PTE para Diana
        prompt = f"""
You are an expert PTE Academic AI Examiner creating a personalized evaluation report for student "Diana".
The exercise type is: {question_type}.

Reference Text / Model Answer: "{reference_text}"
Diana's Spoken Transcription: "{transcribed_text}"

Evaluate Diana strictly according to Pearson PTE Academic standards:
- Oral Fluency (0-90): Smooth rhythm, constant speed, zero self-corrections, no hesitation ("um", "eh").
- Pronunciation (0-90): Clarity of speech, correct word stress.
- Overall PTE Score (0-90): Weighted average.

Provide your response strictly in raw valid JSON (NO Markdown codeblocks, NO ```json):
{{
  "transcription": "{transcribed_text}",
  "fluency_score": 78,
  "pronunciation_score": 82,
  "overall_pte_score": 80,
  "status": "PASS",
  "words_result": [
    {{"word": "example", "status": "correct"}}
  ],
  "error_analysis": "Explicación detallada en español para Diana de dónde falló (palabras omitidas, pausas dudosas, errores de pronunciación).",
  "correct_form": "Muestra la frase u oración correcta en inglés enfatizando cómo debía leerse.",
  "actionable_tips": "Consejos tácticos específicos para que Diana suba su puntaje en el PTE Academic (ritmo, entonación, manejo del micrófono)."
}}

Rules:
- status: "PASS" if overall_pte_score >= 65, else "RETRY".
- words_result: Map EVERY word of the original reference text with "correct", "incorrect", or "missed".
- Write error_analysis, correct_form, and actionable_tips in supportive, professional Spanish addressed directly to Diana.
"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        raw_content = completion.choices[0].message.content.strip()
        cleaned_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.MULTILINE).strip()
        
        data = json.loads(cleaned_json)
        return PTEEvaluationResponse(**data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la evaluación: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok", "tutor": "PTE Academic - Diana Edition"}
