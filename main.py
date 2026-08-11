import os
import json
import re
import io
import random
from typing import List, Literal, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from gtts import gTTS

app = FastAPI(title="PTE Academic Robust Engine")

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
    grammar_vocab_score: int
    overall_pte_score: int
    status: Literal["PASS", "RETRY"]
    words_result: Optional[List[WordResult]] = []
    error_analysis: str
    correct_form: str
    actionable_tips: str

class GeneratedQuestionResponse(BaseModel):
    module: str
    question_type: str
    text: str


@app.post("/generate-question", response_model=GeneratedQuestionResponse)
async def generate_question(module: str = Form(...)):
    try:
        prompts = {
            "SPEAKING": "Generate a single official Pearson PTE Academic 'Read Aloud' paragraph (40-55 words) on an academic topic.",
            "WRITING": "Generate an official Pearson PTE Academic 'Write Essay' prompt (e.g. 'Some people argue that... Write 200-300 words').",
            "READING": "Generate an official Pearson PTE Academic 'Reading: Fill in the Blanks' passage (50-70 words) with key vocabulary words in brackets like [reduced] or [incentives].",
            "LISTENING": "Generate an official Pearson PTE Academic 'Write From Dictation' academic sentence (9-14 words)."
        }

        selected_prompt = prompts.get(module.upper(), prompts["SPEAKING"])

        system_instruction = (
            "You are an official Pearson PTE Academic test creator. "
            "Return ONLY a raw JSON object with fields 'module', 'question_type', and 'text'. "
            "Do NOT wrap in markdown formatting."
        )

        user_content = f"{selected_prompt}\nReturn JSON format: {{\"module\": \"{module.upper()}\", \"question_type\": \"OFFICIAL_PTE\", \"text\": \"YOUR_GENERATED_QUESTION_HERE\"}}"

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7
        )

        raw_content = completion.choices[0].message.content.strip()
        cleaned_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.MULTILINE).strip()
        data = json.loads(cleaned_json)
        return GeneratedQuestionResponse(**data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating question: {str(e)}")


@app.get("/get-audio")
async def get_audio(text: str):
    try:
        accents = [
            {'tld': 'us', 'lang': 'en'},
            {'tld': 'co.uk', 'lang': 'en'},
            {'tld': 'com.au', 'lang': 'en'}
        ]
        chosen_accent = random.choice(accents)

        tts = gTTS(text=text, lang=chosen_accent['lang'], tld=chosen_accent['tld'], slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        return Response(content=mp3_fp.read(), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")


@app.post("/evaluate-audio", response_model=PTEEvaluationResponse)
async def evaluate_audio(
    audio_file: UploadFile = File(...),
    reference_text: str = Form(...),
    question_type: str = Form("Read Aloud")
):
    try:
        audio_content = await audio_file.read()
        filename = audio_file.filename if audio_file.filename else "recording.m4a"

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
                grammar_vocab_score=0,
                overall_pte_score=0,
                status="RETRY",
                words_result=[WordResult(word=w, status="missed") for w in reference_text.split()],
                error_analysis="CERO PUNTOS: No se detectó voz clara en tu respuesta, Diana.",
                correct_form=reference_text,
                actionable_tips="En el examen oficial de PTE, si hay 3 segundos de silencio el micrófono se apaga automáticamente."
            )

        prompt = f"""You are a RIGOROUS Pearson PTE Academic AI Examiner assessing Diana's Speaking.
Question Type: {question_type}
Reference Text: "{reference_text}"
Diana's Audio Transcription: "{transcribed_text}"

Evaluate strictly according to Pearson PTE Score Matrix (0-90):
1. Oral Fluency (0-90)
2. Pronunciation (0-90)
3. Overall Score: Target >= 79.

Return ONLY raw JSON:
{{
  "transcription": "{transcribed_text}",
  "fluency_score": 70,
  "pronunciation_score": 75,
  "grammar_vocab_score": 80,
  "overall_pte_score": 73,
  "status": "RETRY",
  "words_result": [
    {{"word": "example", "status": "correct"}}
  ],
  "error_analysis": "Análisis riguroso en español directo para Diana.",
  "correct_form": "Lectura modelo con pausa adecuada (chunking).",
  "actionable_tips": "Estrategia técnica para alcanzar 79+."
}}
Map EVERY word of reference text in words_result with 'correct', 'incorrect', or 'missed'.
"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        cleaned_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", completion.choices[0].message.content.strip(), flags=re.MULTILINE).strip()
        data = json.loads(cleaned_json)
        return PTEEvaluationResponse(**data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate-text", response_model=PTEEvaluationResponse)
async def evaluate_text(
    user_response: str = Form(...),
    reference_text: str = Form(...),
    question_type: str = Form("Write Essay")
):
    try:
        prompt = f"""You are a Pearson PTE Academic AI Examiner evaluating Diana.
Question Type: {question_type}
Reference Text: {json.dumps(reference_text)}
Diana's Input: {json.dumps(user_response)}

Rules for READING:
1. Diana only types the ordered words separated by commas or spaces.
2. Extract the bracketed words [word] in order from Reference Text.
3. Compare Diana's input words with expected bracketed words.
4. If words match in order, assign score 90 for all metric fields.

Return ONLY a valid JSON object matching this key structure:
{{
  "transcription": {json.dumps(user_response)},
  "fluency_score": 90,
  "pronunciation_score": 90,
  "grammar_vocab_score": 90,
  "overall_pte_score": 90,
  "status": "PASS",
  "words_result": [],
  "error_analysis": "Explicación en español si hay errores o felicitación por acierto.",
  "correct_form": {json.dumps(reference_text)},
  "actionable_tips": "Consejo técnico para PTE Academic Reading."
}}
"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        raw = completion.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        data = json.loads(raw)

        return PTEEvaluationResponse(
            transcription=str(data.get("transcription", user_response)),
            fluency_score=int(data.get("fluency_score", 90)),
            pronunciation_score=int(data.get("pronunciation_score", 90)),
            grammar_vocab_score=int(data.get("grammar_vocab_score", 90)),
            overall_pte_score=int(data.get("overall_pte_score", 90)),
            status="PASS" if data.get("status") == "PASS" else "RETRY",
            words_result=data.get("words_result", []),
            error_analysis=str(data.get("error_analysis", "Evaluación completada.")),
            correct_form=str(data.get("correct_form", reference_text)),
            actionable_tips=str(data.get("actionable_tips", "Continúa practicando."))
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok", "mode": "Safe JSON Parsing Enabled"}
