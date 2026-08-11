import json
from typing import List, Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(
    title="PTE Micro-Exam Engine",
    description="Backend minimalista para evaluación de fluidez y precisión PTE"
)

# Inicializar cliente de OpenAI (utiliza automáticamente OPENAI_API_KEY del entorno)
client = OpenAI()


# --- ESQUEMAS DE RESPUESTA (PYDANTIC) ---

class WordResult(BaseModel):
    word: str
    status: Literal["correct", "incorrect", "missed"]

class PTEEvaluationResponse(BaseModel):
    transcription: str = Field(..., description="Texto transcrito por Whisper")
    fluency_score: int = Field(..., description="Puntaje de fluidez de 0 a 100")
    accuracy_score: int = Field(..., description="Puntaje de precisión de 0 a 100")
    status: Literal["PASS", "RETRY"]
    words_result: List[WordResult]


# --- ENDPOINT PRINCIPAL ---

@app.post("/evaluate-audio", response_model=PTEEvaluationResponse)
async def evaluate_audio(
    audio_file: UploadFile = File(...),
    reference_text: str = Form(...)
):
    """
    Recibe un archivo de audio (m4a, mp3, wav, etc.) y el texto de referencia.
    1. Transcribe con Whisper.
    2. Compara y evalúa con GPT-4o.
    3. Devuelve JSON estructurado sin demoras.
    """
    try:
        # 1. Transcripción del audio usando OpenAI Whisper
        transcription_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=(audio_file.filename, audio_file.file, audio_file.content_type),
            language="en"
        )
        transcribed_text = transcription_response.text.strip()

        # Caso en que no se detecte voz en el audio
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

        # 2. Evaluación con GPT-4o
        system_prompt = (
            "Eres el algoritmo de evaluación de un examen PTE Academic.\n"
            "Compara el texto transcrito del estudiante con el texto de referencia original.\n\n"
            "REGLAS DE EVALUACIÓN:\n"
            "1. 'fluency_score' (0-100): Mide si la respuesta es fluida. Penaliza si hay muletillas (eh, um), "
            "repeticiones o inconsistencias severas en la estructura hablada.\n"
            "2. 'accuracy_score' (0-100): Porcentaje aproximado de palabras clave bien pronunciadas/capturadas.\n"
            "3. 'status': 'PASS' si ambos puntajes son >= 65, de lo contrario 'RETRY'.\n"
            "4. 'words_result': Mapea CADA palabra del texto de referencia e indica si fue 'correct', "
            "'incorrect' o 'missed'.\n"
            "Sé rápido, objetivo y no agregues ningún tipo de texto descriptivo ni explicaciones."
        )

        user_content = f"""
        Texto de referencia: "{reference_text}"
        Transcripción del usuario: "{transcribed_text}"
        """

        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format=PTEEvaluationResponse,
            temperature=0.1
        )

        evaluation_result = completion.choices[0].message.parsed
        evaluation_result.transcription = transcribed_text

        return evaluation_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento: {str(e)}")


# --- COMPROBACIÓN DE SALUD ---

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "PTE Micro-Exam Engine"}
